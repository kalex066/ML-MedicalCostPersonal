# 🏥 Detección de Fumadores — Medical Cost Personal Dataset

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Status](https://img.shields.io/badge/Status-Completo-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📋 Descripción

Este proyecto implementa un **pipeline completo de Machine Learning** para detectar si un asegurado es fumador a partir de sus datos médicos y de costos históricos, **sin depender de su declaración explícita**.

El problema tiene valor directo en la industria aseguradora: un fumador que declara no serlo obtiene una prima más baja de forma fraudulenta. El modelo actúa como **sistema de alerta temprana** para declaraciones inconsistentes, no como reemplazo del proceso de verificación médica.

### ¿Por qué `smoker` como target y no `charges`?

`charges` no es una variable natural del paciente — es una **decisión de la aseguradora**, calculada directamente a partir de `smoker`, `age`, `bmi` y `region`. Predecir `charges` usando esas variables equivale a invertir una fórmula actuarial, no a extraer conocimiento nuevo.

`smoker`, en cambio, es una característica real del individuo que el modelo debe *inferir* a partir de patrones, lo que sí representa un problema de clasificación genuino con aplicación de negocio:

```
age, sex, bmi, children, region, charges → predice → smoker (yes/no)
```
## 🗂️ Descripción del Dataset

| Campo | Descripción |
|---|---|
| **Fuente** | [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| **Registros** | 1,338 asegurados |
| **Features** | 6 (age, sex, bmi, children, region, charges) |
| **Target** | `smoker` → binario (1 = fumador, 0 = no fumador) |
| **Desbalance** | 79.5% no fumadores / 20.5% fumadores |

### Variables

| Variable | Tipo | Descripción |
|:---|:---|:---|
| `age` | int64 | Edad del asegurado |
| `sex` | object | Sexo (male / female) |
| `bmi` | float64 | Índice de masa corporal |
| `children` | int64 | Número de hijos cubiertos |
| `region` | object | Región geográfica (4 valores) |
| `charges` | float64 | Costo médico anual cobrado por la aseguradora — **feature**, no target |
| `smoker` | object | Hábito tabáquico → **TARGET** (convertido a int: yes=1, no=0) |

### Decisión sobre tipos de datos en limpieza

Solo `smoker` cambia de tipo (`object` → `int`) porque es el target y scikit-learn lo requiere numérico. Las columnas categóricas `sex` y `region` se dejan como `object` — el `ColumnTransformer` con `OneHotEncoder` las transforma dentro del Pipeline, evitando data leakage.

---

## 🔄 Flujo del Pipeline

```
Carga del dataset
      ↓
Limpieza
  · str.strip().str.lower() en sex y region
  · smoker → smoker_bin (int)
      ↓
Train/Test Split 80/20 estratificado (stratify=y)
      ↓
ColumnTransformer
  · Numéricas (age, bmi, children, charges): SimpleImputer(median) + StandardScaler
  · Categóricas (sex, region):               SimpleImputer(mode)   + OneHotEncoder
      ↓
Benchmark inicial — 4 modelos con CV-5 estratificada
      ↓
Optimización de hiperparámetros
  · GridSearchCV    
  · RandomizedSearchCV
      ↓
Evaluación final en test set
  · Classification Report + Matriz de Confusión + Curvas ROC
```

---

## 📊 Resultados del Benchmark

### Benchmark inicial (sin tunear)

| Modelo | Accuracy | Precision | Recall | F1 | AUC-Test |
|---|---|---|---|---|---|
| 🏆 **Random Forest (200)** | 0.966 | 0.911 | **0.927** | 0.919 | **0.994** |
| Decision Tree (d=5) | **0.970** | **0.961** | 0.891 | **0.925** | 0.961 |
| Logistic Regression | 0.929 | 0.909 | 0.727 | 0.808 | 0.991 |
| KNN (k=11) | 0.787 | 0.333 | 0.036 | 0.066 | 0.653 |

### ¿Por qué gana Random Forest?

**1. AUC-Test = 0.994** — el más alto. Mide la capacidad de separar clases en todos los umbrales posibles, no solo con el umbral por defecto (0.5). Si se toma un fumador y un no-fumador al azar, el modelo los ordena correctamente el 99.4% de las veces.

**2. Recall = 0.927** — crítico para este problema. Un fumador no detectado (FN) implica que la aseguradora le cobra prima de no-fumador, generando pérdida directa. Random Forest captura 3.6 puntos porcentuales más de fumadores reales que Decision Tree.

**3. Sin overfitting** — AUC-CV = 0.990 ≈ AUC-Test = 0.994. El modelo generaliza correctamente.

**4. Decision Tree tiene mejor F1 (0.925 vs 0.919)** pero lo logra siendo más conservador (Precision alta, Recall bajo). En detección de fraude, el FN es el error más costoso — ese tradeoff es desfavorable.

**5. KNN colapsó** — Recall = 0.036, F1 = 0.066. Con clases desbalanceadas (79/21), el voto de k=11 vecinos tiende a la clase mayoritaria. La Accuracy de 0.787 es una ilusión: el modelo simplemente predice "no fumador" casi siempre.

---

## 🚀 Instrucciones para Ejecutar

### Prerrequisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Pasos

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/<tu-usuario>/MedicalCostInsurance.git
   cd MedicalCostInsurance
   git checkout feature/insurance-ml-pipeline
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Subir el dataset** a la ruta correcta:
   ```
   /data/insurance.csv
   ```
   O descargarlo directamente desde [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).

4. **Ejecutar el notebook:**
   ```bash
   jupyter notebook notebooks/Insurance_ML_Pipeline.ipynb
   ```

5. **Ajustar la ruta del dataset** en la celda de carga (sección 1.1):
   ```python
   # En Colab
   df_raw = pd.read_csv('/content/sample_data/insurance.csv')

   # En local
   df_raw = pd.read_csv('../data/insurance.csv')
   ```
---

## 🔑 Decisiones Técnicas Clave

### 1. Target: `smoker`, no `charges`
`charges` es una consecuencia calculada de las demás variables — predecirla sería invertir la fórmula actuarial. `smoker` es una característica real del individuo con aplicación de negocio directa.

### 2. Datos sucios simulados
El dataset original es limpio. Se introdujeron deliberadamente: nulos (~8% en variables numéricas), outliers extremos (BMI=85, age=150, charges=500,000) y errores de capitalización ("MALE", "FEMALE") para practicar técnicas reales de limpieza.

### 3. Winsorización en lugar de eliminación
Los outliers se capan al percentil 1–99 en lugar de eliminar filas. Con solo 1,338 registros, cada fila vale.

### 4. `OneHotEncoder` dentro del Pipeline
Las columnas `sex` y `region` se codifican dentro del `ColumnTransformer`, no antes del split. Hacerlo manualmente antes del `train_test_split` introduciría data leakage porque el encoder vería la distribución completa de categorías antes de separar train y test.

### 5. Métricas: F1 y AUC, no Accuracy
Con 79.5% de no-fumadores, un modelo que predice siempre "no fumador" alcanza 79.5% de accuracy — completamente inútil. F1 y AUC son las métricas correctas.

---

## ⚠️ Limitaciones

- **Solo 1,338 registros**: el modelo puede ser frágil ante cambios en la distribución demográfica de nuevos asegurados.
- **`charges` como feature**: en un escenario real, la aseguradora fija `charges` *después* de conocer `smoker`. En producción, esta variable no estaría disponible en el momento de la predicción para nuevos clientes. El modelo debe reentrenarse sin `charges` o con un valor estimado.
- **No causalidad**: el modelo detecta patrones estadísticos, no causa. Un BMI alto no *causa* que alguien sea fumador — simplemente coocurre en los datos.
- **Sin SMOTE**: el desbalance 79/21 se maneja via métricas correctas (F1, AUC) pero no se aplicó oversampling. Agregar `class_weight='balanced'` o SMOTE podría mejorar el Recall de la clase minoritaria.

---

## 👥 Autores

| Nombre | Rol |
|---|---|
| [Karen Herrera] | Data Scientist — Pipeline completo, análisis e interpretación |

---

## 📄 Licencia

Este proyecto es realizado con objetivos educativos

---

## 🔗 Referencias

- Dataset: [Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- scikit-learn Pipelines: https://scikit-learn.org/stable/modules/pipeline.html
- scikit-learn ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html
- Interpretación AUC-ROC: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
