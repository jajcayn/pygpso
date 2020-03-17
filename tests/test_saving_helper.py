"""
Tests for saving helper.
"""

import os
import unittest

import numpy as np
import pandas as pd
from gpso.saving_helper import TableSaver
from gpso.utils import H5_EXT
from tables import open_file


class TestTableSaver(unittest.TestCase):

    FILENAME = "test"
    EXTRAS = {
        "a": "b",
        "c": 12,
        "d": 0.54e-4,
        "e": [1, 2, 3],
        "f": np.array([1, 2, 3]),
    }

    def test_init(self):
        saver = TableSaver(filename=self.FILENAME)
        saver.close()
        self.assertTrue(os.path.exists(self.FILENAME + H5_EXT))
        os.remove(self.FILENAME + H5_EXT)

    def test_write_extras(self):
        saver = TableSaver(filename=self.FILENAME, extras=self.EXTRAS)
        saver.close()
        # test saved extras
        saved = open_file(self.FILENAME + H5_EXT)
        for key, value in self.EXTRAS.items():
            saved_val = saved.root.extras[key].read()
            if isinstance(saved_val, bytes):
                saved_val = saved_val.decode()
            if isinstance(saved_val, np.ndarray):
                np.testing.assert_equal(value, saved_val)
            else:
                self.assertEqual(value, saved_val)
        # proper exit
        saved.close()
        os.remove(self.FILENAME + H5_EXT)

    def test_write_single_result(self):
        np.random.seed(42)
        ARRAY = np.random.rand(12, 3)
        PARAMS = {"a": 1.0, "b": 0.1}
        saver = TableSaver(filename=self.FILENAME)
        saver.save_runs(ARRAY, PARAMS)
        saver.close()
        # test saved run
        saved = open_file(self.FILENAME + H5_EXT)
        # check parameters
        for key, value in PARAMS.items():
            saved_val = saved.root.runs["run_0"]["params"][key].read()
            self.assertEqual(saved_val, value)
        # check result itself
        np.testing.assert_equal(
            ARRAY, saved.root.runs["run_0"]["result"]["result"].read()
        )
        # proper exit
        saved.close()
        os.remove(self.FILENAME + H5_EXT)

    def test_write_multiple_df_results(self):
        np.random.seed(42)
        PD_COLUMNS = ["a", "b", "c"]
        DFS = [
            pd.DataFrame(np.random.rand(12, 3), columns=PD_COLUMNS),
            pd.DataFrame(
                np.random.normal(0, 1, size=(12, 3)), columns=PD_COLUMNS
            ),
        ]
        PARAMS = {"a": 1.0, "b": 0.1}
        saver = TableSaver(filename=self.FILENAME)
        saver.save_runs(DFS, PARAMS)
        saver.close()
        # test saved run
        saved = open_file(self.FILENAME + H5_EXT)
        # check parameters
        for key, value in PARAMS.items():
            saved_val = saved.root.runs["run_0"]["params"][key].read()
            self.assertEqual(saved_val, value)
        # check results
        for idx, df in enumerate(DFS):
            group = saved.root.runs["run_0"]["result"][f"result_{idx}"]
            # recreate dataframe
            df_saved = pd.DataFrame(
                group["pd_data"].read(),
                columns=[c.decode() for c in group["pd_columns"].read()],
                index=group["pd_index"].read(),
            )
            pd.testing.assert_frame_equal(df, df_saved)
        # proper exit
        saved.close()
        os.remove(self.FILENAME + H5_EXT)
