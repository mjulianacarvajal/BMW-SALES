import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.loader import load_csv

class TestLoadCSV(unittest.TestCase):

    def test_load_csv_valid(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "valid.csv"
            path.write_text("Model,Year\nX5,2020\n", encoding="utf-8")
            df = load_csv(path)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(df.loc[0, "Model"], "X5")

    def test_load_csv_invalid_path(self):
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                load_csv(Path(temp_dir) / "missing.csv")

    def test_load_csv_encoding_fallback(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latin1.csv"
            path.write_bytes("Model\nCafé\n".encode("latin-1"))
            df = load_csv(path, encoding="utf-8")
            self.assertEqual(df.loc[0, "Model"], "Café")

    def test_load_csv_bad_lines(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_lines.csv"
            path.write_text("Model,Year\nX5,2020\nbad,line,extra\n", encoding="utf-8")
            df = load_csv(path, on_bad_lines="skip")
            self.assertEqual(len(df), 1)

if __name__ == '__main__':
    unittest.main()