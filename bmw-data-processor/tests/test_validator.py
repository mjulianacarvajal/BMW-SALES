import pandas as pd
import pytest
from src.utils.validator import transformar_datos

def test_transformar_datos():
    # Test case 1: Basic functionality
    df = pd.DataFrame({
        'Price_USD': [20000, 25000, 30000],
        'Year': [2015, 2016, 2017],
        'Color': ['Red', 'Blue', 'Green']
    })
    columnas_a_mantener = ['Price_USD', 'Year', 'Color']
    transformed_df = transformar_datos(df, columnas_a_mantener)
    
    assert transformed_df.shape[0] == 3  # No duplicates or NaNs
    assert set(transformed_df.columns) == set(columnas_a_mantener)

    # Test case 2: Handling duplicates
    df_with_duplicates = pd.DataFrame({
        'Price_USD': [20000, 20000, 25000],
        'Year': [2015, 2015, 2016],
        'Color': ['Red', 'Red', 'Blue']
    })
    transformed_df = transformar_datos(df_with_duplicates, columnas_a_mantener)
    
    assert transformed_df.shape[0] == 2  # Duplicates should be removed

    # Test case 3: Handling NaN values
    df_with_nans = pd.DataFrame({
        'Price_USD': [20000, None, 30000],
        'Year': [2015, 2016, 2017],
        'Color': ['Red', 'Blue', None]
    })
    transformed_df = transformar_datos(df_with_nans, columnas_a_mantener)
    
    assert transformed_df.shape[0] == 2  # Rows with NaNs should be dropped
    assert 'Color' in transformed_df.columns  # Color column should still be present

    # Test case 4: No columns to keep
    df_empty = pd.DataFrame({
        'Price_USD': [20000, 25000],
        'Year': [2015, 2016],
        'Color': ['Red', 'Blue']
    })
    transformed_df = transformar_datos(df_empty, [])
    
    assert transformed_df.shape[0] == 0  # No columns to keep should result in empty DataFrame