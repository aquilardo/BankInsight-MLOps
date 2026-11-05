from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_URI = (PROJECT_ROOT / "mlruns").as_uri()
REMOTE_URI = "http://localhost:5000"

EXPERIMENT_NAME = "BankInsight-Classification"
SOURCE_TAG = "migration.source_run_id"

local = MlflowClient(tracking_uri=LOCAL_URI)
remote = MlflowClient(tracking_uri=REMOTE_URI)


def migrate_run(source_summary, remote_experiment_id):
    source_id = source_summary.info.run_id
    source = local.get_run(source_id)

    run_name = source.data.tags.get(
        "mlflow.runName",
        "unnamed_run"
    )

    print(f"\nMigrando: {run_name}")
    print("ID local:", source_id)

    if source.info.status != "FINISHED":
        raise RuntimeError(
            f"La ejecución local {source_id} no está FINISHED."
        )

    # Detectar migraciones anteriores de esta misma ejecución.
    existing_runs = remote.search_runs(
        experiment_ids=[remote_experiment_id],
        filter_string=f"tags.`{SOURCE_TAG}` = '{source_id}'",
        max_results=100
    )

    for existing in existing_runs:
        if existing.info.status == "FINISHED":
            print("Ya migrada correctamente. Se omite.")
            return "SKIPPED"

        print(
            "Eliminando migración incompleta:",
            existing.info.run_id
        )
        remote.delete_run(existing.info.run_id)

    # Crear la ejecución remota.
    destination = remote.create_run(
        experiment_id=remote_experiment_id,
        start_time=source.info.start_time,
        tags={
            **source.data.tags,
            SOURCE_TAG: source_id,
            "migration.source_tracking_uri": "local_mlruns"
        }
    )

    destination_id = destination.info.run_id

    try:
        # 1. Migrar parámetros.
        for key, value in source.data.params.items():
            remote.log_param(destination_id, key, value)

        # 2. Migrar historial completo de métricas.
        for metric_name in source.data.metrics:
            history = local.get_metric_history(
                source_id,
                metric_name
            )

            for item in history:
                remote.log_metric(
                    destination_id,
                    metric_name,
                    item.value,
                    timestamp=item.timestamp,
                    step=item.step
                )

        # 3. Descargar artefactos desde el almacenamiento local.
        with TemporaryDirectory() as temp_dir:
            mlflow.set_tracking_uri(LOCAL_URI)

            artifact_dir = local.download_artifacts(
                run_id=source_id,
                path="",
                dst_path=temp_dir
            )

            artifact_path = Path(artifact_dir)

            # 4. Subir artefactos al servidor MLflow y SILO.
            mlflow.set_tracking_uri(REMOTE_URI)

            if artifact_path.is_dir():
                for item in artifact_path.iterdir():
                    if item.is_dir():
                        remote.log_artifacts(
                            destination_id,
                            str(item),
                            artifact_path=item.name
                        )
                    else:
                        remote.log_artifact(
                            destination_id,
                            str(item)
                        )

        # 5. Verificar artefactos en el servidor.
        mlflow.set_tracking_uri(REMOTE_URI)

        remote_artifacts = remote.list_artifacts(destination_id)

        local_artifacts = local.list_artifacts(source_id)

        if local_artifacts and not remote_artifacts:
            raise RuntimeError(
                "No se encontraron artefactos en el servidor."
            )

        # 6. Marcar como finalizada únicamente después de verificar.
        remote.set_terminated(
            destination_id,
            status="FINISHED",
            end_time=source.info.end_time
        )

        print("MIGRADA CORRECTAMENTE")
        print("ID remoto:", destination_id)
        print("Artefactos raíz:", len(remote_artifacts))

        return "SUCCESS"

    except Exception:
        remote.set_terminated(
            destination_id,
            status="FAILED"
        )
        raise


def main():
    print("\n=== 1. PREPARANDO EXPERIMENTOS ===")

    mlflow.set_tracking_uri(REMOTE_URI)

    local_exp = local.get_experiment_by_name(EXPERIMENT_NAME)

    if local_exp is None:
        raise RuntimeError("No existe el experimento local.")

    remote_exp = remote.get_experiment_by_name(EXPERIMENT_NAME)

    if remote_exp is None:
        remote_exp_id = remote.create_experiment(EXPERIMENT_NAME)
        remote_exp = remote.get_experiment(remote_exp_id)

    print("Experimento local:", local_exp.experiment_id)
    print("Experimento remoto:", remote_exp.experiment_id)

    source_runs = local.search_runs(
        experiment_ids=[local_exp.experiment_id],
        max_results=1000
    )

    print("Ejecuciones locales:", len(source_runs))

    successful = 0
    skipped = 0

    print("\n=== 2. MIGRANDO EJECUCIONES ===")

    for source_summary in source_runs:
        result = migrate_run(
            source_summary,
            remote_exp.experiment_id
        )

        if result == "SUCCESS":
            successful += 1
        else:
            skipped += 1

    print("\n=== 3. VERIFICACIÓN FINAL ===")

    mlflow.set_tracking_uri(REMOTE_URI)

    final_runs = remote.search_runs(
        experiment_ids=[remote_exp.experiment_id],
        max_results=1000
    )

    finished = sum(
        1 for run in final_runs
        if run.info.status == "FINISHED"
    )

    failed = sum(
        1 for run in final_runs
        if run.info.status == "FAILED"
    )

    print("Migraciones nuevas:", successful)
    print("Ejecuciones omitidas:", skipped)
    print("Total remoto:", len(final_runs))
    print("FINISHED:", finished)
    print("FAILED:", failed)

    if len(final_runs) != len(source_runs) or failed:
        raise RuntimeError(
            "La migración requiere revisión. "
            "No se considera completada."
        )

    print("\nMIGRACIÓN COMPLETADA CORRECTAMENTE.")


if __name__ == "__main__":
    main()
