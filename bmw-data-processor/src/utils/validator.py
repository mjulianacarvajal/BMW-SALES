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