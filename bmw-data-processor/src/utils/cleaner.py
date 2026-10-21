import logging

import pandas as pd


logger = logging.getLogger(__name__)


def limpiar_y_verificar_data(df: pd.DataFrame, columnas_esperadas: list[str]) -> pd.DataFrame:
    """
    Limpia un DataFrame quitando espacios en headers y valores string, 
    y verifica la presencia de columnas esperadas.

    Args:
        df (pd.DataFrame): El DataFrame a limpiar y verificar.
        columnas_esperadas (list): Una lista de nombres de columnas que se esperan en el DataFrame.

    Returns:
        pd.DataFrame: El DataFrame limpio.
    """
    df = df.copy()
    df.columns = [column.strip() if isinstance(column, str) else column for column in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    faltantes = set(columnas_esperadas) - set(df.columns)
    if faltantes:
        logger.warning(
            "Faltan columnas esperadas: %s. El pipeline continuará con las disponibles.",
            sorted(faltantes),
        )
    return df