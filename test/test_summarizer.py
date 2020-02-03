import unittest
import glob

import pandas as pd

from gurih.data.summarizer import Summarizer

class SummarizerTest(unittest.TestCase):
    """
    Test if the summarizer returns the right outputs
    """

    def setUp(self):
        self.dir = "test_data/test_summarizer"
        self.X = glob.glob(f"{self.dir}/*.json")
        self.summarizer = Summarizer(output_dir=self.dir)

    def test_summarizer_output(self):
        """
        Test if Summarizer returns the correct output.
        """
        summarized_df = self.summarizer.transform(self.X)


        # Check whether Summarizer correctly writes the output
        csv_output = f"{self.dir}/statistical_summary.csv"
        csvs = glob.glob(f"{self.dir}/*.csv")

        print(csv_output)
        print(csvs)

        self.assertEqual(type(summarized_df), pd.DataFrame)
        self.assertEqual(summarized_df.shape[1], 11)
        self.assertGreater(summarized_df.shape[0], 1)
        self.assertTrue(csv_output in csvs)

if __name__ == "__main__":
    unittest.main()

        