import os, json, warnings
warnings.filterwarnings("ignore")

import joblib
from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split


from data.loader import load_csv
from utils.cleaner import limpiar_y_verificar_data
from analysis.eda import preparar_datos
from data.transformer import save_eda_outputs





def main():
    # Load the data
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "dataBmwSales/data/archivo.csv"  # Ajusta esto a la ruta real de tu archivo
    df = load_csv(DATA_PATH)

    # Clean and verify the data
    expected_columns = ["Price_USD", "Year"]  # Adjust according to your dataset
    df = limpiar_y_verificar_data(df, expected_columns)

    # Prepare the data for modeling
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preparar_datos(df)

    # Save exploratory data analysis outputs
    save_eda_outputs(df)

if __name__ == "__main__":
    main()





"""


BASE_DIR = Path("C:/dataBmwSales").resolve()
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUT_DIR = BASE_DIR / "data" / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
MODELS_DIR = OUT_DIR / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
"""
