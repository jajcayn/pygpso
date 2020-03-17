"""
Tests for saving helper.
"""

import os
import unittest

import numpy as np
import pandas as pd
from gpso.saving_helper import (
    ALL_RUNS_KEY,
    EXTRAS_KEY,
    RUN_PREFIX,
    TableSaver,
    table_reader,
)
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
            saved_val = saved.root[EXTRAS_KEY][key].read()
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
        SCORE = 0.8
        saver = TableSaver(filename=self.FILENAME)
        saver.save_runs(ARRAY, SCORE, PARAMS)
        saver.close()
        # test saved run
        saved = open_file(self.FILENAME + H5_EXT)
        # check parameters
        for key, value in PARAMS.items():
            saved_val = saved.root[ALL_RUNS_KEY][f"{RUN_PREFIX}0"]["params"][
                key
            ].read()
            self.assertEqual(saved_val, value)
        # check result itself
        np.testing.assert_equal(
            ARRAY,
            saved.root[ALL_RUNS_KEY][f"{RUN_PREFIX}0"]["result"][
                "result"
            ].read(),
        )
        self.assertEqual(
            SCORE,
            saved.root[ALL_RUNS_KEY][f"{RUN_PREFIX}0"]["result"][
                "score"
            ].read(),
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
        SCORES = [0.7, 0.1]
        saver = TableSaver(filename=self.FILENAME)
        saver.save_runs(DFS, SCORES, PARAMS)
        saver.close()
        # test saved run
        saved = open_file(self.FILENAME + H5_EXT)
        # check parameters
        for key, value in PARAMS.items():
            saved_val = saved.root[ALL_RUNS_KEY][f"{RUN_PREFIX}0"]["params"][
                key
            ].read()
            self.assertEqual(saved_val, value)
        # check results
        for idx, df in enumerate(DFS):
            group = saved.root[ALL_RUNS_KEY][f"{RUN_PREFIX}0"]["result"][
                f"result_{idx}"
            ]
            # recreate dataframe
            df_saved = pd.DataFrame(
                group["pd_data"].read(),
                columns=[c.decode() for c in group["pd_columns"].read()],
                index=group["pd_index"].read(),
            )
            pd.testing.assert_frame_equal(df, df_saved)
            self.assertEqual(SCORES[idx], group["score"].read())
        # proper exit
        saved.close()
        os.remove(self.FILENAME + H5_EXT)


class TestTableReader(unittest.TestCase):
    FILENAME_1 = "pd_extras_multiple"
    FILENAME_2 = "np_single"
    # 1
    EXTRAS = {
        "a": "b",
        "c": 12,
        "d": 0.54e-4,
        "e": [1, 2, 3],
        "f": np.array([1, 2, 3]),
    }
    PD_COLUMNS = ["a", "b", "c"]
    DFS = [
        pd.DataFrame(np.random.rand(12, 3), columns=PD_COLUMNS),
        pd.DataFrame(np.random.normal(0, 1, size=(12, 3)), columns=PD_COLUMNS),
    ]
    PARAMS = {"a": 1.0, "b": 0.1}
    SCORES = [0.7, 0.1]

    # 2
    ARRAY = np.random.rand(12, 3)
    SCORE = 0.8

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # 1
        saver1 = TableSaver(filename=cls.FILENAME_1, extras=cls.EXTRAS)
        saver1.save_runs(cls.DFS, cls.SCORES, cls.PARAMS)
        saver1.close()
        # 2
        saver2 = TableSaver(filename=cls.FILENAME_2, extras=None)
        saver2.save_runs(cls.ARRAY, cls.SCORE, cls.PARAMS)
        saver2.close()

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.FILENAME_1 + H5_EXT)
        os.remove(cls.FILENAME_2 + H5_EXT)

    def test_table_reader_multiple(self):
        results, scores, parameters, extras = table_reader(self.FILENAME_1)

        # test length - we should have one result
        self.assertEqual(len(results), 1)
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(parameters), 1)

        # test results themselves
        for df_expected, df_result in zip(self.DFS, results[0]):
            pd.testing.assert_frame_equal(df_expected, df_result)
        for score_expected, score_result in zip(self.SCORES, scores[0]):
            self.assertEqual(score_expected, score_result)
        self.assertDictEqual(self.PARAMS, parameters[0])
        for v_exp, v_res in zip(self.EXTRAS.values(), extras.values()):
            if isinstance(v_exp, np.ndarray):
                np.testing.assert_equal(v_exp, v_res)
            else:
                self.assertEqual(v_exp, v_res)

    def test_table_reader_single(self):
        results, scores, parameters, extras = table_reader(self.FILENAME_2)

        # test length - we should have one result
        self.assertEqual(len(results), 1)
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(parameters), 1)

        # test results themselves
        np.testing.assert_equal(self.ARRAY, results[0])
        self.assertEqual(self.SCORE, scores[0])
        self.assertDictEqual(self.PARAMS, parameters[0])
        self.assertEqual(extras, None)


if __name__ == "__main__":
    unittest.main()
