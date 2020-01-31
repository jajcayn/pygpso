"""
Set of tests for parameter space, partition tree and ternary splits.
"""

import os
import unittest

import numpy as np
from anytree import PreOrderIter
from gpso.param_space import NORM_PARAMS_BOUNDS, LeafNode, ParameterSpace
from gpso.utils import PKL_EXT
from sklearn.preprocessing import MinMaxScaler


class TestParameterSpace(unittest.TestCase):

    TEMP_FILENAME = "test"
    BEST_SCORE = 20.1
    ROOT_SCORE = 10.2
    parameter_names = ["sigma", "xi"]
    parameter_bounds = [[1.0, 6.5], [-0.3, 0.3]]

    def _create_space(self):
        self.space = ParameterSpace(
            parameter_names=self.parameter_names,
            parameter_bounds=self.parameter_bounds,
        )

    def _create_and_split_space(self):
        self._create_space()
        self.space.score = self.ROOT_SCORE
        # split first time
        split1 = self.space.ternary_split()
        split1[0].score = np.random.rand()
        split1[1].score = np.random.rand()
        split1[2].score = np.random.rand()
        # split first child leaf second time
        split11 = split1[0].ternary_split()
        split11[0].score = np.random.rand()
        split11[1].score = np.random.rand()
        split11[2].score = self.BEST_SCORE

    def test_basic(self):
        self._create_space()
        self.assertEqual(self.space.name, "full_domain")
        self.assertEqual(self.space.parent, None)
        self.assertEqual(self.space.depth, 0)
        self.assertEqual(self.space.ndim, len(self.parameter_bounds))
        self.assertEqual(
            self.space.norm_bounds,
            [NORM_PARAMS_BOUNDS for _ in range(self.space.ndim)],
        )

    def test_scaler(self):
        self._create_space()
        self.assertTrue(isinstance(self.space.scaler, MinMaxScaler))
        self.assertEqual(self.space.scaler.n_samples_seen_, 2)
        np.testing.assert_equal(
            np.array(list(map(lambda x: np.max(x), self.parameter_bounds))),
            self.space.scaler.data_max_,
        )
        np.testing.assert_equal(
            np.array(list(map(lambda x: np.min(x), self.parameter_bounds))),
            self.space.scaler.data_min_,
        )
        np.testing.assert_equal(
            np.array(list(map(lambda x: x[1] - x[0], self.parameter_bounds))),
            self.space.scaler.data_range_,
        )

    def test_max_depth(self):
        self._create_and_split_space()
        self.assertEqual(self.space.max_depth, 2)

    def test_get_best_score_leaf(self):
        self._create_and_split_space()
        best_root = self.space.get_best_score_leaf(depth=0)
        self.assertTrue(isinstance(best_root, LeafNode))
        self.assertEqual(best_root.score, self.ROOT_SCORE)
        best_child = self.space.get_best_score_leaf(depth=2)
        self.assertTrue(isinstance(best_child, LeafNode))
        self.assertEqual(best_child.score, self.BEST_SCORE)

    def test_center_list(self):
        self._create_space()
        # normed coordinates
        np.testing.assert_equal(
            self.space.get_center_as_list(normed=True),
            [np.mean(NORM_PARAMS_BOUNDS) for _ in range(self.space.ndim)],
        )
        # real coordinates
        np.testing.assert_equal(
            self.space.get_center_as_list(normed=False),
            [np.mean(single_bounds) for single_bounds in self.parameter_bounds],
        )

    def test_center_dict(self):
        self._create_space()
        # normed coordinates
        self.assertDictEqual(
            self.space.get_center_as_dict(normed=True),
            {
                key: value
                for key, value in zip(
                    self.parameter_names,
                    [
                        np.mean(NORM_PARAMS_BOUNDS)
                        for _ in range(self.space.ndim)
                    ],
                )
            },
        )
        # real coordinates
        self.assertDictEqual(
            self.space.get_center_as_dict(normed=False),
            {
                key: value
                for key, value in zip(
                    self.parameter_names,
                    [
                        np.mean(single_bounds)
                        for single_bounds in self.parameter_bounds
                    ],
                )
            },
        )

    def test_ternary_split(self):
        self._create_space()
        children_leaves = self.space.ternary_split()
        # we need 3 leaves
        self.assertEqual(len(children_leaves), 3)
        # all should be `LeafNode` class
        self.assertTrue(
            all(isinstance(leaf, LeafNode) for leaf in children_leaves)
        )
        # depth should be 1 for all children from root
        self.assertTrue(all(leaf.depth == 1 for leaf in children_leaves))
        # parent has to be root space
        self.assertTrue(
            all(leaf.parent == self.space for leaf in children_leaves)
        )
        # scaler has to be the same for all children and root
        self.assertTrue(
            all(leaf.scaler == self.space.scaler for leaf in children_leaves)
        )
        # assert correct normed coords in childrens
        coord_delta = (NORM_PARAMS_BOUNDS[1] - NORM_PARAMS_BOUNDS[0]) / 3
        for i, leaf in enumerate(children_leaves):
            splitted_coord = (i * coord_delta, (i + 1) * coord_delta)
            self.assertListEqual(
                leaf.norm_bounds, [splitted_coord, NORM_PARAMS_BOUNDS]
            )

    def test_save_load(self):
        self._create_and_split_space()
        self.space.save(self.TEMP_FILENAME)
        loaded_space = ParameterSpace.from_file(self.TEMP_FILENAME)
        self.assertEqual(self.space.__class__, loaded_space.__class__)
        # compare attributes as name, score, label, etc. for all nodes
        self.assertTrue(
            all(
                all(
                    getattr(orig_node, attr) == getattr(loaded_node, attr)
                    for attr in self.space.COMPARE_ATTRS
                )
                for orig_node, loaded_node in zip(
                    PreOrderIter(self.space), PreOrderIter(loaded_space)
                )
            )
        )
        os.remove(self.TEMP_FILENAME + PKL_EXT)

    def test_sample_uniformly(self):
        N_POINTS = 50
        self._create_and_split_space()
        child = self.space[0][2]
        sampled = child.sample_uniformly(N_POINTS)
        # assert shape
        self.assertTupleEqual(sampled.shape, (N_POINTS, child.ndim))
        # assert sampled points are within bounds
        for dim in range(child.ndim):
            self.assertTrue(
                np.all(child.norm_bounds[dim][0] <= sampled[:, dim])
                and np.all(child.norm_bounds[dim][1] >= sampled[:, dim])
            )

    def test_grow(self):
        GROW_DEPTH = 4
        self._create_and_split_space()
        child = self.space[0][2]
        sampled = child.grow(depth=GROW_DEPTH)
        n_points = sum([np.power(3, i) for i in range(GROW_DEPTH)])
        # assert shape
        self.assertTupleEqual(sampled.shape, (n_points, child.ndim))
        # assert no children were actually created
        self.assertEqual(len(child.children), 0)
        self.assertEqual(self.space.max_depth, 2)
        # assert sampled points are within bounds
        for dim in range(child.ndim):
            self.assertTrue(
                np.all(child.norm_bounds[dim][0] <= sampled[:, dim])
                and np.all(child.norm_bounds[dim][1] >= sampled[:, dim])
            )

    def test_norm_denorm(self):
        TEST_COORDS = np.array([[5.2, 0.2]])
        self._create_space()
        normed_coords = self.space.normalise_coords(TEST_COORDS)
        orig_coords = self.space.denormalise_coords(normed_coords)
        np.testing.assert_equal(orig_coords, TEST_COORDS)


if __name__ == "__main__":
    unittest.main()
