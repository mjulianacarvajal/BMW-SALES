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