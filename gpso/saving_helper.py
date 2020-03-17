"""
Helper for saving intermediate results from optimisation.
"""
import logging
import os

import pandas as pd
from tables import open_file

from .utils import H5_EXT

# keys in HDF file hierarchy
ALL_RUNS_KEY = "runs"
EXTRAS_KEY = "extras"
RUN_PREFIX = "run_"


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
            "/", ALL_RUNS_KEY, "Individual exploration runs"
        )
        if extras is not None:
            extra_group = self.file.create_group(
                "/", EXTRAS_KEY, "Extra parameters of the model"
            )
            assert isinstance(extras, dict)
            self._write_dict(extra_group, extras)

        self.results_counter = 0

    def save_runs(self, result, score, parameters):
        """
        Save runs of the model / objective function to file.

        :param result: result(s) of the run, can be single result or multiple
            results (same parameters, different results, typically valid for
            stochastic systems)
        :type result: any pytables supported + pd.DataFrame|list of thereof
        :param score: score(s) of the run
        :type score: float|list[float]
        :param parameters: parameters for this particular run(s)
        :type parameters: dict
        """
        run_group = self.file.create_group(
            self.group,
            f"{RUN_PREFIX}{self.results_counter}",
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
            assert isinstance(score, (list, tuple))
            assert len(score) == len(result)
            for idx, ind_result in enumerate(result):
                ind_group = self.file.create_group(
                    result_group, f"result_{idx}", f"Result for the run {idx}"
                )
                self._write_result(ind_group, ind_result, score[idx])
        else:
            self._write_result(result_group, result, score)

        self.file.flush()
        self.results_counter += 1

    def close(self):
        """
        Clean exit.
        """
        self.file.close()

    def _write_result(self, group, result, score):
        """
        Write single result to given group.

        :param group: group to write to
        :type group: `tables.group.Group`
        :param result: result to write
        :type result: any pytables supported + pd.DataFrame
        :param score: score to write
        :type score: float
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
        self.file.create_array(group, "score", score)

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


def table_reader(filename):
    """
    Read saved table from the optimisation and return results, scores and
    parameters of individual runs.

    :param filename: filename of the HDF file
    :type filename: str
    :return: results, scores, parameters and extras if present; if mutliple
        runs per parameter set, that item is a list itself
    :rtype: (list,list,list,dict|None)
    """
    if not filename.endswith(H5_EXT):
        filename += H5_EXT
    logging.info(f"Loading {filename}...")
    loaded = open_file(filename, mode="r")

    def _read_params(group):
        parameters = {}
        for key in loaded.walk_nodes(group["params"], "Array"):
            parameters[key.name] = key.read()
        return parameters

    def _read_result(group):
        score = group["score"].read()
        if "result" in group:
            result = group["result"].read()
        elif "pd_data" in group:
            result = pd.DataFrame(
                group["pd_data"].read(),
                columns=[c.decode() for c in group["pd_columns"].read()],
                index=group["pd_index"].read(),
            )
        return result, score

    assert ALL_RUNS_KEY in loaded.root
    # get list of all groups with runs
    results_groups = loaded.list_nodes(f"/{ALL_RUNS_KEY}")
    all_results = []
    all_scores = []
    all_parameters = []
    for group in results_groups:
        all_parameters.append(_read_params(group))
        # if we have more results per run
        ind_results_groups = list(loaded.walk_groups(group["result"]))
        if len(ind_results_groups) >= 2:
            results_run = []
            scores_run = []
            for individual_group in ind_results_groups[1:]:
                result, score = _read_result(individual_group)
                results_run.append(result)
                scores_run.append(score)
            all_results.append(results_run)
            all_scores.append(scores_run)
        else:
            result, score = _read_result(group["result"])
            all_results.append(result)
            all_scores.append(score)

    assert len(all_parameters) == len(results_groups)
    assert len(all_results) == len(results_groups)
    assert len(all_scores) == len(results_groups)

    # try to read extras
    if EXTRAS_KEY in loaded.root:
        extras = {}
        for key in loaded.walk_nodes(f"/{EXTRAS_KEY}", "Array"):
            value = key.read()
            extras[key.name] = (
                value.decode() if isinstance(value, bytes) else value
            )
    else:
        extras = None
    loaded.close()

    return all_results, all_scores, all_parameters, extras
