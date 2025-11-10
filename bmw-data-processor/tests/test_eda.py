import unittest
import pandas as pd
from src.analysis.eda import preparar_datos

class TestEDAFunctions(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'Price_USD': [20000, 30000, 25000, 40000],
            'Year': [2015, 2016, 2017, 2018]
        })

    def test_preparar_datos(self):
        X_train, X_test, y_train, y_test, num_cols, cat_cols = preparar_datos(self.df)

        # Check if the shapes are correct
        self.assertEqual(X_train.shape[0], 3)
        self.assertEqual(X_test.shape[0], 1)

        # Check if the target column is separated correctly
        self.assertIn('Price_USD', y_train.name)
        self.assertIn('Price_USD', y_test.name)

        # Check if numerical columns are identified correctly
        self.assertEqual(num_cols, ['Year'])
        self.assertEqual(cat_cols, [])

if __name__ == '__main__':
    unittest.main()