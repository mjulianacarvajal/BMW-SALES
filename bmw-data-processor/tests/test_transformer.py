import os
import unittest
import pandas as pd
from src.data.transformer import save_eda_outputs

class TestSaveEdaOutputs(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Price_USD': [20000, 30000, 25000],
            'Year': [2015, 2016, 2017]
        })
        self.out_dir = 'test_outputs'

    def test_save_eda_outputs(self):
        save_eda_outputs(self.df, out_dir=self.out_dir)
        # Check if the output files are created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'numeric_summary.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'categorical_summary.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'eda_meta.json')))
        # Add more assertions as needed to validate the content of the files

    def tearDown(self):
        # Clean up the output directory after tests
        import shutil
        shutil.rmtree(self.out_dir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()