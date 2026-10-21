import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def preparar_datos(
    df: pd.DataFrame,
    target_col: str = "Price_USD",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create modeling features, identify their types, and split X/y."""
    if target_col not in df.columns:
        raise KeyError(f"No existe la columna objetivo {target_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")
    if len(df) < 2:
        raise ValueError("Se necesitan al menos dos filas para dividir los datos.")

    data = df.copy()
    if "Year" in data.columns:
        years = pd.to_numeric(data["Year"], errors="coerce")
        if years.notna().any():
            data["Car_Age"] = years.max() - years
        else:
            logger.warning("La columna 'Year' no contiene valores numéricos; se omite Car_Age.")
    else:
        logger.warning("No existe 'Year' para calcular 'Car_Age'; se omitirá esa feature.")

    features = [column for column in data.columns if column != target_col]
    X = data[features].copy()
    y = data[target_col].copy()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [column for column in features if column not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, num_cols, cat_cols
