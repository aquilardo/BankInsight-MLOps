# BankInsight MLOps

**Plataforma de aprendizaje automático para la predicción de suscripción a depósitos bancarios, con ciclo de vida MLOps y despliegue en Kubernetes.**

## 1. Descripción del proyecto

BankInsight MLOps es un proyecto académico que implementa un ciclo de vida completo de Machine Learning en un entorno reproducible y desplegable.

La solución utiliza datos históricos de campañas de marketing bancario para entrenar modelos de clasificación binaria capaces de estimar la probabilidad de que un cliente contrate un depósito a plazo.

El proyecto integra experimentación, seguimiento de modelos, registro de versiones, servicio de inferencia, contenedorización, orquestación, monitorización y mecanismos de reentrenamiento.

## 2. Objetivo general

Diseñar e implementar una arquitectura MLOps que permita gestionar de forma reproducible el ciclo de vida de un modelo de clasificación, desde la preparación de los datos hasta su despliegue, monitorización y eventual actualización en producción.

## 3. Objetivos específicos

- Preparar y validar el conjunto de datos.
- Entrenar y comparar diferentes modelos e hiperparámetros.
- Registrar experimentos, métricas, parámetros y artefactos mediante MLflow.
- Gestionar versiones de modelos y seleccionar una versión para producción.
- Implementar una API de inferencia utilizando FastAPI.
- Construir imágenes Docker reproducibles.
- Desplegar el servicio en Kubernetes con múltiples réplicas.
- Implementar monitorización de data drift y concept drift.
- Definir criterios medibles para activar el reentrenamiento.
- Desarrollar una interfaz para consultar predicciones y resultados de monitorización.

## 4. Dataset y problema de Machine Learning

**Dataset:** Bank Marketing — UCI Machine Learning Repository.

**Tipo de problema:** clasificación binaria supervisada.

**Variable objetivo:** `y`.

- `yes`: el cliente contrató un depósito a plazo.
- `no`: el cliente no contrató un depósito a plazo.

Se excluirá la variable `duration` del entrenamiento principal porque representa la duración de la llamada y no está disponible antes de realizar el contacto con el cliente. Su utilización podría introducir fuga de información (*data leakage*).

## 5. Arquitectura tecnológica

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.12 |
| Procesamiento de datos | Pandas y NumPy |
| Machine Learning | Scikit-learn |
| Seguimiento de experimentos | MLflow |
| Base de datos de metadatos | PostgreSQL |
| Almacenamiento de artefactos | MinIO |
| API de inferencia | FastAPI |
| Contenedores | Docker |
| Orquestación | Kubernetes |
| Monitorización estadística | SciPy y métricas propias |
| Interfaz de usuario | Streamlit |
| Pruebas | Pytest |
| Control de versiones | Git |

## 6. Ciclo de vida MLOps

1. Ingesta y validación de datos.
2. Preparación y transformación de variables.
3. Entrenamiento y experimentación.
4. Evaluación y comparación de modelos.
5. Registro y versionado en MLflow.
6. Despliegue del modelo mediante FastAPI.
7. Construcción y ejecución de contenedores Docker.
8. Despliegue y escalamiento en Kubernetes.
9. Monitorización de datos y rendimiento.
10. Detección de degradación y reentrenamiento.

## 7. Monitorización

Se implementarán dos mecanismos diferenciados:

- **Data drift:** identificación de cambios en la distribución de las variables de entrada respecto a los datos de referencia.
- **Concept drift:** simulación y seguimiento de cambios en la relación entre las variables de entrada y la variable objetivo, evaluando el rendimiento del modelo en lotes sucesivos.

Los criterios de alerta y reentrenamiento se documentarán y justificarán mediante resultados experimentales.

## 8. Estructura del repositorio

```text
bankinsight-mlops/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   └── evaluation/
├── api/
├── monitoring/
│   ├── data_drift/
│   ├── concept_drift/
│   └── reports/
├── tests/
├── docker/
├── kubernetes/
├── ui/
├── docs/
│   ├── architecture/
│   ├── evidence/
│   ├── methodology/
│   └── defense/
├── scripts/
├── requirements.txt
└── README.md
```

## 9. Reproducibilidad

El proyecto utilizará dependencias con versiones definidas, control de versiones Git, configuraciones mediante variables de entorno y servicios contenedorizados.

Las instrucciones de instalación, ejecución, despliegue y pruebas se documentarán progresivamente.

## 10. Alcance académico

Este proyecto se desarrolla con fines académicos como implementación práctica de conceptos de MLOps, Machine Learning en producción, monitorización y despliegue de servicios escalables.

