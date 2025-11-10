def test_limpiar_y_verificar_data():
    import pandas as pd
    from src.utils.cleaner import limpiar_y_verificar_data

    # Test case 1: DataFrame with expected columns
    df = pd.DataFrame({
        'Price_USD': [20000, 30000, 25000],
        'Year': [2015, 2016, 2017]
    })
    expected_columns = ['Price_USD', 'Year']
    cleaned_df = limpiar_y_verificar_data(df, expected_columns)
    assert cleaned_df.shape == (3, 2)
    assert 'Price_USD' in cleaned_df.columns
    assert 'Year' in cleaned_df.columns

    # Test case 2: DataFrame missing expected columns
    df_missing = pd.DataFrame({
        'Price_USD': [20000, 30000],
    })
    cleaned_df_missing = limpiar_y_verificar_data(df_missing, expected_columns)
    assert cleaned_df_missing.shape == (2, 1)  # Only Price_USD should remain
    assert 'Year' not in cleaned_df_missing.columns

    # Test case 3: DataFrame with whitespace in headers and values
    df_whitespace = pd.DataFrame({
        ' Price_USD ': [20000, 30000],
        ' Year ': [2015, 2016]
    })
    cleaned_df_whitespace = limpiar_y_verificar_data(df_whitespace, expected_columns)
    assert cleaned_df_whitespace.columns.tolist() == ['Price_USD', 'Year']
    assert cleaned_df_whitespace['Price_USD'].iloc[0] == 20000
    assert cleaned_df_whitespace['Year'].iloc[0] == 2015

    print("All tests passed!")