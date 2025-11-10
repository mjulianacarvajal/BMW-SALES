import unittest
import pandas as pd
from src.data.loader import load_csv

class TestLoadCSV(unittest.TestCase):

    def test_load_csv_valid(self):
        df = load_csv('path/to/valid_data.csv')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_load_csv_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_csv('path/to/invalid_data.csv')

    def test_load_csv_invalid_encoding(self):
        with self.assertRaises(UnicodeDecodeError):
            load_csv('path/to/invalid_encoding.csv')

    def test_load_csv_bad_lines(self):
        df = load_csv('path/to/data_with_bad_lines.csv', on_bad_lines='warn')
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()