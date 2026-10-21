import pandas as pd


def transformar_datos(df: pd.DataFrame, columnas_a_mantener: list[str]) -> pd.DataFrame:
    """
    Transforma un DataFrame eliminando duplicados, valores nulos y columnas no deseadas.

    Args:
        df (pd.DataFrame): El DataFrame a transformar.
        columnas_a_mantener (list): Una lista de nombres de columnas que se deben mantener.

    Returns:
        pd.DataFrame: El DataFrame transformado.
    """
    if not columnas_a_mantener:
        return pd.DataFrame()

    df = df[columnas_a_mantener].copy()
    df = df.drop_duplicates()
    df = df.dropna()        
    return df