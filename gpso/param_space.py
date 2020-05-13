"""
Parameter space and its parcellations.
"""

import pickle
from collections import OrderedDict
from itertools import tee

import numpy as np
from anytree import NodeMixin, PreOrderIter
from anytree.exporter import DictExporter
from anytree.importer import DictImporter
from sklearn.preprocessing import MinMaxScaler

from .utils import PKL_EXT, PointLabels

NORM_PARAMS_BOUNDS = (0, 1)


class LeafNode(NodeMixin):
    """
    Represents a leaf node in the partition tree of the parameter space. Each
    partition is n-orthotopic / hyperrectanglular. Internally works with
    normalised coordinates.
    """

    COMPARE_ATTRS = [
        "norm_bounds",
        "parameter_names",
        "name",
        "ndim",
        "depth",
        "score",
        "sampled",
        "label",
    ]

    INIT_ATTRS = [
        "label",
        "name",
        "norm_bounds",
        "parameter_names",
        "sampled",
        "scaler",
        "score",
        "children",
    ]

    @staticmethod
    def _validate_single_bound(single_bound):
        """
        Validate single bound.

        :param single_bound: single coordinate bound to validate
        :type single_bound: list|tuple
        """
        assert isinstance(single_bound, (list, tuple))
        assert len(single_bound) == 2
        assert single_bound[1] > single_bound[0]

    def _validate_param_bounds(self, param_bounds):
        """
        Validate param bounds.

        :param param_bounds: parameter bounds to validate
        :type param_bounds: list|None
        """
        assert param_bounds is not None
        assert isinstance(param_bounds, (list, tuple))
        [
            self._validate_single_bound(single_bound)
            for single_bound in param_bounds
        ]

    def __init__(
        self,
        norm_bounds,
        scaler,
        parameter_names,
        score=0.0,
        sampled=False,
        label=PointLabels.not_assigned,
        name="",
        parent=None,
        children=None,
    ):
        """
        :param norm_bounds: bounds for this leaf in normalised coordinates
        :type norm_bounds: list[list[float]]
        :param scaler: already trained MinMaxScaler
        :type scaler: `sklearn.preprocessing.MinMaxScaler`
        :param parameter_names: parameter names, must have the same order as
            norm_bounds
        :type parameter_names: list[str]
        :param score: score of the leaf, either UCB or evaluated obj. function
        :type score: float
        :param sampled: whether leaf was already sampled
        :type sampled: bool
        :param label: label of the leaf
        :type label: `gpso.gp_surrogate.PointLabels`
        :param name: name of the leaf
        :type name: str
        :param parent: parent of this leaf
        :type parent: `LeafNode`|None
        :param children: children of this leaf
        :type children: list[`LeafNode`]|None
        """
        assert isinstance(
            scaler, MinMaxScaler
        ), "Scaler must be sklearn's `MinMaxScaler`"
        self.scaler = scaler

        self.name = name
        self.score = score
        self.sampled = sampled
        self.label = label
        # define tree properties
        self.parent = parent
        if children:
            self.children = children

        self._validate_param_bounds(norm_bounds)
        assert len(norm_bounds) == self.ndim
        self.norm_bounds = norm_bounds

        assert len(parameter_names) == self.ndim
        assert all(
            isinstance(param_name, str) for param_name in parameter_names
        )
        self.parameter_names = parameter_names

    def __str__(self):
        """
        String representation
        """
        return (
            f"Leaf node `{self.name}`: score {self.score}; center at "
            f"{self.get_center_as_dict(normed=True)}; depth {self.depth}"
        )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, pos):
        """
        Overload __getitem__ to get the children.
        """
        return self.children[pos]

    @property
    def ndim(self):
        """
        Return dimension of the parameter space.
        """
        return self.scaler.data_max_.shape[0]

    def sample_uniformly(self, n_points, seed=None):
        """
        Return n_points uniformly sampled within the leaf node domain.

        :param n_points: number of point to sample within the domain
        :type n_points: int
        :param seed: seed for the random number generator
        :type seed: int|None
        :return: normalised coordinates of sampled points [n_points x ndim]
        :rtype: np.ndarray
        """
        np.random.seed(seed)
        return np.random.uniform(
            low=[bound[0] for bound in self.norm_bounds],
            high=[bound[1] for bound in self.norm_bounds],
            size=(n_points, self.ndim),
        )

    def grow(self, depth):
        """
        Recursively grow tree from this leaf up to depth, returning coordinates
        of all the centers. Do not save children though, in the end set
        children to None!

        :param depth: depth of the subtree to grow
        :type depth: int
        :return: normalised coordinates of the child leafs within the subtree
        :rtype: np.ndarray
        """
        leaves = [self]
        normed_coords = []
        for _ in range(depth):
            # save coords from current level
            normed_coords.append(
                np.array(
                    [leaf.get_center_as_list(normed=True) for leaf in leaves]
                )
            )
            # split all leaves in the current level and sum them into one big
            # list
            leaves = sum((leaf.ternary_split() for leaf in leaves), [])
        # reset growth, i.e. set children to empty list
        self.children = list()
        return np.vstack(normed_coords)

    def get_center_as_list(self, normed=False):
        """
        Return center point of the domain associated with this leaf as list of
        coordinates.

        :param normed: whether to return normalized coordinates or original
        :type normed: bool
        :return: center point for each coordinate
        :rtype: list
        """
        centers = [np.mean(dim_bounds) for dim_bounds in self.norm_bounds]
        if not normed:
            centers = np.around(
                self.scaler.inverse_transform(np.array([centers])), decimals=5
            )[0].tolist()
        return centers

    def get_center_as_dict(self, normed=False):
        """
        Return center point of the domain associated with this leaf as dict
        with key as parameter name and its value as the center.

        :param normed: whether to return normalized coordinates or original
        :type normed: bool
        :return: center point for each coordinate associated with its name
        :rtype: dict
        """
        centers = [np.mean(dim_bounds) for dim_bounds in self.norm_bounds]
        if not normed:
            centers = np.around(
                self.scaler.inverse_transform(np.array([centers])), decimals=5
            )[0].tolist()
        return {
            param_name: centre
            for param_name, centre in zip(self.parameter_names, centers)
        }

    def _replace_normed_coord(self, index, new_coord):
        """
        Return list of normed coordinates with one coordinate replaced.

        :param index: index of which coordinate to replace in the list
        :type index: int
        :param new_coord: new coordinate for the normed coordinate
        :type new_coord: list|tuple
        :return: list of normed coordinates with replaced coordinate
        :rtype: list[list[float]]
        """
        assert index < self.ndim
        self._validate_single_bound(new_coord)
        return [
            item if idx != index else new_coord
            for idx, item in enumerate(self.norm_bounds)
        ]

    def ternary_split(self):
        """
        Parcellate this leaf using the ternary split function along the largest
        dimension. For discussion why ternary is the best choise see the paper.

        Hadida, J., Sotiropoulos, S. N., Abeysuriya, R. G., Woolrich, M. W., &
            Jbabdi, S. (2018). Bayesian Optimisation of Large-Scale Biophysical
            Networks. NeuroImage, 174, 219-236.

        :return: list of child leafs according to the ternary split function
        :rtype: list[LeafNode]
        """
        # compute coordinate differences for each dimension for selecting the
        # largest dimension
        coord_diffs = list(map(lambda x: x[1] - x[0], self.norm_bounds))
        largest_dim = np.argmax(coord_diffs)
        delta_coord = coord_diffs[largest_dim] / 3  # ternary split
        children_coords = [
            self.norm_bounds[largest_dim][0] + i * delta_coord for i in range(4)
        ]

        def pairwise(iterable):
            """
            Generate pairwise chunks as s -> (s0,s1), (s1,s2), (s2, s3), ...
            """
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        children_leaves = [
            LeafNode(
                norm_bounds=self._replace_normed_coord(
                    largest_dim, splitted_coord
                ),
                scaler=self.scaler,
                parameter_names=self.parameter_names,
                name=self.name + "->" + idx,
                parent=self,
                children=None,
            )
            for splitted_coord, idx in zip(
                pairwise(children_coords), ["l", "c", "r"]
            )
        ]
        # middle leaf has the same center
        middle_center = children_leaves[1].get_center_as_list(normed=True)
        parent_center = self.get_center_as_list(normed=True)
        np.testing.assert_allclose(middle_center, parent_center)

        return children_leaves


class ParameterSpace(LeafNode):
    """
    Represents full parameter space of the optimisation problem. Allows for
    ternary partition of the space into leafs.
    """

    @classmethod
    def from_file(cls, filename):
        """
        Load parameter space as a ternary tree from pickle file.
        """
        # load dict from pickle
        if not filename.endswith(PKL_EXT):
            filename += PKL_EXT
        with open(filename, "rb") as f:
            loaded_dict = pickle.load(f)

        def _sanitize_dict(raw_dict, keys_to_delete):
            """
            Remove keys from dict - possibly nested.
            """
            sanit_dict = {}
            for key, value in raw_dict.items():
                if key not in keys_to_delete:
                    if isinstance(value, (list, tuple)):
                        sanit_dict[key] = [
                            _sanitize_dict(list_val, keys_to_delete)
                            if isinstance(list_val, dict)
                            else list_val
                            for list_val in value
                        ]
                    elif isinstance(value, dict):
                        sanit_dict[key] = _sanitize_dict(value, keys_to_delete)
                    else:
                        sanit_dict[key] = value
            return sanit_dict

        # sanitise dict for correct init
        keys_to_delete = set(loaded_dict.keys()) - set(cls.INIT_ATTRS)
        sanitised_dict = _sanitize_dict(loaded_dict, keys_to_delete)
        # import as `LeafNode` class
        importer = DictImporter(nodecls=LeafNode)
        root_leaf_node = importer.import_(sanitised_dict)
        # force `ParameterSpace` class which extends `LeafNode` with some
        # useful methods
        root_leaf_node.__class__ = ParameterSpace

        return root_leaf_node

    def __init__(self, parameter_bounds, parameter_names):
        """
        Initialise ternary tree.

        :param parameter_bounds: list of parameter bounds for the optimisation
            problem
        :type parameter_bounds: list[list[float]]
        :param parameter_names: list of parameter names for the reference
        :type parameter_names: list[str]
        """
        # train MinMaxScaler
        self._validate_param_bounds(parameter_bounds)
        scaler = MinMaxScaler(feature_range=NORM_PARAMS_BOUNDS)
        scaler.fit(np.array(parameter_bounds).T)

        parameter_names = parameter_names or ["" for _ in range(self.ndim)]
        assert len(parameter_names) == len(parameter_bounds)

        # normalised coordinates for the full domain
        norm_bounds_full = [
            NORM_PARAMS_BOUNDS for _ in range(len(parameter_bounds))
        ]

        # initialise leaf node for the full domain
        super().__init__(
            norm_bounds=norm_bounds_full,
            scaler=scaler,
            parameter_names=parameter_names,
            name="full_domain",
            parent=None,
            children=None,
        )

    @property
    def max_depth(self):
        """
        Return depth of the tree.
        """
        return np.max([node.depth for node in PreOrderIter(self)])

    def get_best_score_leaf(self, depth, only_not_sampled=True):
        """
        Return leaf with best score in a given level.

        :param depth: level at which the best scored leaf is seek
        :type depth: int
        :param only_not_sampled: whether to search for leaves that were not
            sampled yet
        :type only_not_sampled: bool
        :return: leaf with best score
        :rtype: `LeafNode`|None
        """
        try:
            return sorted(
                PreOrderIter(
                    self,
                    filter_=lambda node: (node.depth == depth)
                    and not (node.sampled and only_not_sampled),
                ),
                key=lambda node: node.score,
                reverse=True,
            )[0]
        except IndexError:
            return None

    def normalise_coords(self, orig_coords):
        """
        Normalise original coordinates.

        :param orig_coords: original coordinates in the parameter space as
            [n points x ndim]
        :type orig_coords: np.ndarray
        :return: normalised coordinates
        :rtype: np.ndarray
        """
        assert orig_coords.ndim == 2
        assert orig_coords.shape[1] == self.ndim
        return self.scaler.transform(orig_coords)

    def denormalise_coords(self, normed_coords):
        """
        Deormalise normed coordinates.

        :param normed_coords: normed coordinates in the parameter space as
            [n points x ndim]
        :type normed_coords: np.ndarray
        :return: denormalised coordinates
        :rtype: np.ndarray
        """
        assert normed_coords.ndim == 2
        assert normed_coords.shape[1] == self.ndim
        return self.scaler.inverse_transform(normed_coords)

    def save(self, filename):
        """
        Save tree with all its attributes and computed scores to binary pickle
        file.

        :param filename: filename of the pickle
        :type filename: str
        """
        # export to OrderedDict
        exporter = DictExporter(dictcls=OrderedDict, attriter=sorted)
        exported_dict = exporter.export(self)
        if not filename.endswith(PKL_EXT):
            filename += PKL_EXT
        # serialize to pickle
        with open(filename, "wb") as f:
            pickle.dump(exported_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
