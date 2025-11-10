import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from packaging import version
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from pathlib import Path
import logging

#----------------------------
# 0) Paths (ajustar según sea necesario)
# ---------------------------

BASE_DIR = Path("C:/dataBmwSales").resolve()
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUT_DIR = BASE_DIR / "data" / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
MODELS_DIR = OUT_DIR / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)



# logging simple config
#pendiente ajustar codificación de formato -> Preguntarle a Juan Utf-8 me dio igual las salidas.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("Inicio del programa")

def load_csv(path, encoding="latin-1", on_bad_lines="skip", **kwargs):
    """
    Carga un CSV con fallback de encoding y control de líneas malas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe")
    try:
        return pd.read_csv(path, encoding=encoding, on_bad_lines=on_bad_lines, **kwargs)
    except UnicodeDecodeError:
        logging.warning("UnicodeDecodeError con %s; reintentando con latin1", encoding)
        return pd.read_csv(path, encoding="latin-1", on_bad_lines=on_bad_lines, **kwargs)

def save_eda_outputs(df, out_dir=PLOTS_DIR):
    """
    Guarda plots y resúmenes (CSV + JSON) de EDA en out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    try:
        # resumen numérico y categórico
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        df[num_cols].describe().to_csv(out_dir / "numeric_summary.csv")
        df[cat_cols].describe().to_csv(out_dir / "categorical_summary.csv")
        meta["shape"] = df.shape
        meta["num_cols"] = num_cols
        meta["cat_cols"] = cat_cols

        # histogramas solo si existe la columna target
        if "Price_USD" in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df["Price_USD"].dropna(), bins=40, kde=True, color="#59a14f")
            plt.title("Distribución de Price_USD")
            plt.tight_layout()
            plt.savefig(out_dir / "target_distribution.png")
            plt.close()

            plt.figure(figsize=(8,6))
            sns.boxplot(x=df["Price_USD"].dropna())
            plt.title("Boxplot Price_USD")
            plt.tight_layout()
            plt.savefig(out_dir / "target_boxplot.png")
            plt.close()
        else:
            logging.warning("No existe columna 'Price_USD' para generar plots.")

        # correlación si hay suficientes numéricas
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
            plt.title("Matriz de correlación")
            plt.tight_layout()
            plt.savefig(out_dir / "corr_heatmap.png")
            plt.close()

        # Guardar metadatos
        with open(out_dir / "eda_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        logging.info("EDA guardado en %s", out_dir)
    except Exception as e:
        logging.exception("Error al guardar EDA: %s", e)


def preparar_datos(df, target_col="Price_USD", test_size=0.2, random_state=42):
    """
    Versión reforzada: chequea columnas, crea features de forma segura (.copy()), devuelve X/y.
    """
    if target_col not in df.columns:
        raise KeyError(f"No existe la columna objetivo {target_col}")
    if "Year" not in df.columns:
        logging.warning("No existe 'Year' para calcular 'Car_Age' — se omitirá esa feature.")
        df = df.copy()
    else:
        # evitar SettingWithCopy
        current_year = int(df["Year"].max())
        df = df.copy()
        df["Car_Age"] = current_year - df["Year"]

    features = [c for c in df.columns if c != target_col]
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    X = df[features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, num_cols, cat_cols


# ---------------------------
# 1) Carga de datos
# Programación Funcional
# ---------------------------

df = load_csv(DATA_PATH)

columnas_esperadas = ["Price_USD", "Year"]  # ajustar según dataset

def limpiar_y_verificar_data(df, columnas_esperadas):
    """
    Limpia un DataFrame quitando espacios en headers y valores string, 
    y verifica la presencia de columnas esperadas.

    Args:
        df (pd.DataFrame): El DataFrame a limpiar y verificar.
        columnas_esperadas (list): Una lista de nombres de columnas que se esperan en el DataFrame.

    Returns:
        pd.DataFrame: El DataFrame limpio.
    """
    # Limpieza ligera: quita espacios en headers y valores tipo string
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Revisa columnas esperadas
    faltantes = set(columnas_esperadas) - set(df.columns)
    if faltantes:
        print(f"[ADVERTENCIA] Faltan columnas: {faltantes}. El pipeline seguirá con las disponibles.")
    return df

def transformar_datos(df, columnas_a_mantener):
    """
    Transforma un DataFrame eliminando duplicados, valores nulos y columnas no deseadas.

    Args:
        df (pd.DataFrame): El DataFrame a transformar.
        columnas_a_mantener (list): Una lista de nombres de columnas que se deben mantener.

    Returns:
        pd.DataFrame: El DataFrame transformado.
    """
    # Seleccionar solo las columnas a mantener
    df = df[columnas_a_mantener]
    df = df.drop_duplicates()
    df = df.dropna()        
    return df

df = limpiar_y_verificar_data(df, columnas_esperadas)

# ---------------------------
# 1.1) ESTADISTICA DESCRIPTIVA
# ---------------------------

print("\n== EDA ==")
print("Shape:", df.shape)
print("Primeras filas:\n", df.head(3), "\n")
print("Tipos de datos:\n", df.dtypes, "\n")
print("Nulos por columna:\n", df.isna().sum().sort_values(ascending=False), "\n")

# Estadísticos numéricos
num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
print("Resumen numérico:\n", df[num_cols_all].describe().T, "\n")

#Distribución del target (conteo por clase)
target_col = 'Sales_Classification'
counts = df[target_col].value_counts()
print(counts)
plt.figure(figsize=(6,4))
counts.plot(kind='bar')
plt.title('Distribution of Sales_Classification')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_distribution_count.png"))
plt.close()


# Estadísticos categóricos
cat_cols_all = df.select_dtypes(include=["object"]).columns.tolist()
print("Resumen categórico:\n", df[cat_cols_all].describe().T, "\n")

#
# Histograma de precio
plt.figure(figsize=(6,4))
plt.hist(df["Price_USD"], bins=40, color="#4e79a7", edgecolor="white")
plt.title("Distribución de Price_USD")
plt.xlabel("Precio en USD")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"))
plt.close()

# Ventas Por Color
"""
plt.subplots(figsize=(8,6))
df_dhhist = pd.DataFrame({
    x_label: grp['Color'].value_counts()
    for x_label, grp in df.groupby('Region')
})
sns.heatmap(df_dhhist,cmap='seismic')
plt.title("Ventas por Color y Región")
plt.tight_layout("Distribución de Ventas por Color y Región")
plt.xlabel('Region')
plt.ylabel('Color')
plt.savefig(os.path.join(PLOTS_DIR, "sales_by_color_region.png"))
plt.close()
"""


# Histograma para visualizar la distribución y la tendencia central (media, mediana)
plt.figure(figsize=(8,6))
sns.histplot(df["Price_USD"], bins=30, kde=True, color="#59a14f")
plt.axvline(df["Price_USD"].mean(), color='red', linestyle='dashed', label='Media')
plt.axvline(df["Price_USD"].median(), color='blue', linestyle='dashed', label='Mediana')
plt.title("Distribución de Precios con Media y Mediana(en USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_distribution_with_stats.png"))
plt.close()


# Precio promedio de los vehículos es de alrededor de $75,000 con un rango
# significativo en los precios como lo indican la desviación estándar y # la
# diferencia entre los valores mínimo y máximo. La mediana, al estar cerca de
# la media, sugiere una distribución de precios relativamente equilibrada.

print("Estadísticas descriptivas para Precio en USD:")
print(df['Price_USD'].describe())
print("\nPromedio de Precio en USD:")
# El resultado del promedio también será un entero (si los valores son enteros) o un flotante,
# por lo que lo redondeamos al imprimirlo si queremos el promedio como entero.
print(f"{df['Price_USD'].mean():.0f}") # Redondea el promedio al entero más cercano para la impresión


# Boxplot para visualizar la mediana, cuartiles y posibles valores atípicos
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Price_USD'])
plt.title('Boxplot de Price_USD')
plt.xlabel('Price in USD')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_boxplot.png"))      
plt.close()


# Heatmap de correlación (numéricas)
if len(num_cols_all) > 1:
    corr = df[num_cols_all].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Matriz de correlación (numéricas)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "corr_heatmap.png"))
    plt.close()

# Histograma del target
plt.figure(figsize=(6,7))
plt.hist(df["Price_USD"], bins=5, color="#4e79a7", edgecolor="white")
plt.title("Distribución del Precio (USD)")
plt.xlabel("Precio en USD")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_histogram.png"))
plt.close()

print("EDA completado. Gráficos guardados en:", PLOTS_DIR)


# ---------------------------
# 2) Preparación de datos para modelado
# ---------------------------

def preparar_datos(df, target_col="Price_USD", test_size=0.2, random_state=42):
    """
    Prepara los datos para el modelado, incluyendo la creación de features, 
    la selección de columnas, la detección de tipos de datos y la división 
    en conjuntos de entrenamiento y prueba.

    Args:
        datos (pd.DataFrame): El DataFrame que contiene los datos.
        target_col (str): El nombre de la columna objetivo. Por defecto es "Price_USD".
        test_size (float): La proporción del conjunto de datos a incluir en el conjunto de prueba. Por defecto es 0.2.
        random_state (int): La semilla para la generación de números aleatorios, para la reproducibilidad. Por defecto es 42.

    Returns:
        tuple: Una tupla que contiene:
            - X_train (pd.DataFrame): Conjunto de entrenamiento de las features.
            - X_test (pd.DataFrame): Conjunto de prueba de las features.
            - y_train (pd.Series): Conjunto de entrenamiento de la variable objetivo.
            - y_test (pd.Series): Conjunto de prueba de la variable objetivo.
            - num_cols (list): Lista de nombres de columnas numéricas.
            - cat_cols (list): Lista de nombres de columnas categóricas.
    """
    assert target_col in df.columns, f"No existe la columna objetivo {target_col}"

    # Puedes crear algunas features derivadas (opcional):
    current_year = df["Year"].max()
    df["Car_Age"] = current_year - df["Year"]

    # Selección de features (excluye la etiqueta)
    features = [c for c in df.columns if c != target_col]

    # Detectar numéricas y categóricas
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    print("Numéricas:", num_cols)
    print("Categóricas:", cat_cols)

    X = df[features].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, num_cols, cat_cols

# Ejemplo de uso:
# Supongamos que tienes un DataFrame llamado 'datos'

# Llamar a la función
X_train, X_test, y_train, y_test, num_cols, cat_cols = preparar_datos(df)

# Ahora puedes usar X_train, X_test, y_train, y_test, num_cols y cat_cols para tu modelado

if __name__ == "__main__":
    # carga usando la función con encoding y on_bad_lines por defecto
    df = load_csv(DATA_PATH)
    # limpieza ligera y verificación
    columnas_esperadas = ["Price_USD", "Year"]  # ajustar según dataset
    df = limpiar_y_verificar_data(df, columnas_esperadas)
    save_eda_outputs(df, out_dir=PLOTS_DIR)
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preparar_datos(df)



