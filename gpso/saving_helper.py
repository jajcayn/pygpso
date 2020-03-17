"""
Helper for saving intermediate results from optimisation.
"""
import logging
import os

import pandas as pd
from tables import open_file

from .utils import H5_EXT


class TableSaver:
    """
    Saver handles HDF5 files and can save intermediate results to them.
    """

    def __init__(self, filename, extras=None):
        """
        :param filename: filename for the HDF file
        :type filename: str
        :param extras: extras to write as a group into HDF file, e.g. default
            parameters of the model, which are not subject to optimisaion
        :type extras: dict|None
        """
        if not filename.endswith(H5_EXT):
            filename += H5_EXT
        if os.path.exists(filename):
            logging.warning(f"{filename} already exists, will be overwritten.")
        self.file = open_file(filename, mode="w")
        self.filename = filename
        # default group
        self.group = self.file.create_group(
            "/", "runs", "Individual exploration runs"
        )
        if extras is not None:
            extra_group = self.file.create_group(
                "/", "extras", "Extra parameters of the model"
            )
            assert isinstance(extras, dict)
            self._write_dict(extra_group, extras)

        self.results_counter = 0

    def save_runs(self, result, parameters):
        """
        Save runs of the model / objective function to file.

        :param result: result(s) of the run, can be single result or multiple
            results (same parameters, different results, typically valid for
            stochastic systems)
        :type result: any pytables supported + pd.DataFrame|list of thereof
        :param parameters: parameters for this particular run(s)
        :type parameters: dict
        """
        run_group = self.file.create_group(
            self.group,
            f"run_{self.results_counter}",
            f"Results for run no. {self.results_counter}",
        )
        params_group = self.file.create_group(
            run_group, "params", "Parameters for the run"
        )
        self._write_dict(params_group, parameters)
        result_group = self.file.create_group(
            run_group, "result", "Results for the run(s)"
        )
        # we have multiple runs with the same parameters
        if isinstance(result, (list, tuple)):
            for idx, ind_result in enumerate(result):
                ind_group = self.file.create_group(
                    result_group, f"result_{idx}", f"Result for the run {idx}"
                )
                self._write_result(ind_group, ind_result)
        else:
            self._write_result(result_group, result)

        self.file.flush()
        self.results_counter += 1

    def close(self):
        """
        Clean exit.
        """
        self.file.close()

    def _write_result(self, group, result):
        """
        Write single result to given group.

        :param group: group to write to
        :type group: `tables.group.Group`
        :param result: result to write
        :type result: any pytables supported + pd.DataFrame
        """
        if isinstance(result, pd.DataFrame):
            result_dict = {
                "pd_data": result.values,
                "pd_columns": list(result.columns),
                "pd_index": list(result.index),
            }
            self._write_dict(group, result_dict)
        else:
            self.file.create_array(group, "result", result)

    def _write_dict(self, group, dict_data):
        """
        Write dictionary to HDF file.

        :param group: group to write to
        :type group: `tables.group.Group`
        :param dict_data: dictionary to save
        :type dict_data: dict
        """
        for rkey, rval in dict_data.items():
            # string to bytes if necessary
            if isinstance(rval, str):
                rval = rval.encode()
            self.file.create_array(group, rkey, rval)
