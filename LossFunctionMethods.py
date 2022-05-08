import collections

# %%Import statements and setting taxa data
from io import StringIO

import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow.experimental.numpy as tnp
from skbio import TreeNode
from skbio.tree import MissingNodeError, DuplicateNodeError
import collections.abc
import skbio.diversity
tnp.experimental_enable_numpy_behavior(prefer_float32=False)



def create_otu_ids():
    projects = ['COAD', 'READ', 'HNSC', 'STAD']
    dframes = []
    for project in projects:
        fpath = '/Users/remycross/Downloads/tcma_data 2/{}/WGS/'.format(project)
        fname = 'bacteria.unambiguous.tissue.sample.reads.txt'
        df = pd.read_table(fpath + fname, index_col=0).transpose()
        # print(assay,project,df.shape)
        dframes.append(df)
    d = pd.concat(dframes).fillna(0.0)
    return [str(x) for x in d.columns]


def count(u, v, otu_ids, tree):
    tree_index = tree.to_array(nan_length_value=0.0)
    u_counts = count_list(u, otu_ids, tree_index)
    v_counts = count_list(v, otu_ids, tree_index)

    final_list = [list(x) for x in zip(u_counts, v_counts)]

    return tnp.asarray(final_list), tree_index


def count_list(u, otu_ids, tree_index):
    values_list = [u[otu_ids.index(tree_index['id_index'][i].name)] \
                       if tree_index['id_index'][i].name in otu_ids \
                       else 0 for i in range(len(tree_index['id_index']))]

    children_nodes_structure = tree_index['child_index']
    for i in range(len(children_nodes_structure)):
        row = children_nodes_structure[i]
        parent = row[0]
        children = set(row[1:])
        values_list[parent] = sum([values_list[c] for c in children])
    return values_list


def _vectorize_counts_and_tree(u_counts, v_counts, otu_ids, tree):
    """ Index tree and convert counts to np.array in corresponding order
    """

    counts_by_node, tree_index = count(u_counts, v_counts, otu_ids, tree)
    branch_lengths = tree_index['length']

    return counts_by_node.T, tree_index, branch_lengths


def _setup_pairwise_unifrac(u_counts, v_counts, otu_ids, tree, validate,
                            normalized, unweighted):
    if validate:
        _validate(u_counts, v_counts, otu_ids, tree)

    # temporarily store u_counts and v_counts in a 2-D array as that's what
    # _vectorize_counts_and_tree takes
    u_counts = tnp.asarray(u_counts)
    v_counts = tnp.asarray(v_counts)
    counts_by_node, tree_index, branch_lengths = \
        _vectorize_counts_and_tree(u_counts, v_counts, otu_ids, tree)
    # unpack counts vectors for single pairwise UniFrac calculation
    u_node_counts = counts_by_node[0]
    v_node_counts = counts_by_node[1]

    u_total_count = K.sum(u_counts)
    v_total_count = K.sum(v_counts)

    return (u_node_counts, v_node_counts, u_total_count, v_total_count,
            tree_index)


def _validate(u_counts, v_counts, otu_ids, tree):
    _validate_counts_matrix([u_counts, v_counts], suppress_cast=True)
    _validate_otu_ids_and_tree(counts=u_counts, otu_ids=otu_ids, tree=tree)


def _validate_counts_matrix(counts, ids=None, suppress_cast=False):
    results = []

    # handle case of where counts is a single vector by making it a matrix.
    # this has to be done before forcing counts into an ndarray because we
    # don't yet know that all of the entries are of equal length
    if isinstance(counts, pd.core.frame.DataFrame):
        if ids is not None and len(counts.index) != len(ids):
            raise ValueError(
                "Number of rows in ``counts``"
                " must be equal to number of provided ``ids``."
            )
        return tnp.asarray(counts)
    else:

        if len(counts) == 0 or not isinstance(counts[0],
                                              collections.abc.Iterable):
            counts = [counts]
        counts = tnp.asarray(counts)
        if counts.ndim > 2:
            raise ValueError(
                "Only 1-D and 2-D array-like objects can be provided "
                "as input. Provided object has %d dimensions." %
                counts.ndim)

        if ids is not None and len(counts) != len(ids):
            raise ValueError(
                "Number of rows in ``counts`` must be equal "
                "to number of provided ``ids``."
            )

        lens = []
        for v in counts:
            results.append(_validate_counts_vector(v, suppress_cast))
            lens.append(len(v))
        if len(set(lens)) > 1:
            raise ValueError(
                "All rows in ``counts`` must be of equal length."
            )
        return tnp.asarray(results)


def _validate_counts_vector(counts, suppress_cast=False):
    """Validate and convert input to an acceptable counts vector type.
    Note: may not always return a copy of `counts`!
    """
    counts = tnp.asarray(counts)
    try:
        if not tnp.all(tnp.isreal(counts)):
            raise Exception
    except Exception:
        raise ValueError("Counts vector must contain real-valued entries.")
    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    elif tnp.any((counts < 0)):
        raise ValueError("Counts vector cannot contain negative values.")

    return counts


def _validate_otu_ids_and_tree(counts, otu_ids, tree):
    len_otu_ids = len(otu_ids)
    set_otu_ids = set(otu_ids)
    if len_otu_ids != len(set_otu_ids):
        raise ValueError("``otu_ids`` cannot contain duplicated ids.")

    if len(counts) != len_otu_ids:
        raise ValueError("``otu_ids`` must be the same length as ``counts`` "
                         "vector(s).")

    if len(tree.root().children) == 0:
        raise ValueError("``tree`` must contain more than just a root node.")

    if len(tree.root().children) > 2:
        # this is an imperfect check for whether the tree is rooted or not.
        # can this be improved?
        raise ValueError("``tree`` must be rooted.")

    # all nodes (except the root node) have corresponding branch lengths
    # all tip names in tree are unique
    # all otu_ids correspond to tip names in tree
    branch_lengths = []
    tip_names = []
    for e in tree.traverse():
        if not e.is_root():
            branch_lengths.append(e.length)
        if e.is_tip():
            tip_names.append(e.name)
    set_tip_names = set(tip_names)
    if len(tip_names) != len(set_tip_names):
        raise DuplicateNodeError("All tip names must be unique.")

    if tnp.any(tnp.array([branch is None for branch in branch_lengths])):
        raise ValueError("All non-root nodes in ``tree`` must have a branch "
                         "length.")
    missing_tip_names = set_otu_ids - set_tip_names
    if missing_tip_names != set():
        n_missing_tip_names = len(missing_tip_names)
        raise MissingNodeError("All ``otu_ids`` must be present as tip names "
                               "in ``tree``. ``otu_ids`` not corresponding to "
                               "tip names (n=%d): %s" %
                               (n_missing_tip_names,
                                " ".join(missing_tip_names)))


def weighted_unifrac(u_counts, v_counts, otu_ids, tree,
                     normalized=False,
                     validate=False):
    u_node_counts, v_node_counts, u_total_count, v_total_count, tree_index = \
        _setup_pairwise_unifrac(u_counts, v_counts, otu_ids, tree, validate,
                                normalized=normalized, unweighted=False)

    branch_lengths = tree_index['length']

    if normalized:
        tip_indices = _get_tip_indices(tree_index)
        node_to_root_distances = skbio.diversity._phylogenetic._tip_distances(branch_lengths, tree,
                                                tip_indices)
        return _weighted_unifrac_normalized(u_node_counts, v_node_counts,
                                            u_total_count, v_total_count,
                                            branch_lengths,
                                            node_to_root_distances)
    else:
        return _weighted_unifrac(u_node_counts, v_node_counts,
                                 u_total_count, v_total_count,
                                 branch_lengths)[0]


def _get_tip_indices(tree_index):
    tip_indices = np.array([n.id for n in tree_index['id_index'].values()
                             if n.is_tip()])
    return tip_indices


def _weighted_unifrac_normalized(u_node_counts, v_node_counts, u_total_count,
                                 v_total_count, branch_lengths,
                                 node_to_root_distances):
    if u_total_count == 0.0 and v_total_count == 0.0:
        # handle special case to avoid division by zero
        return 0.0
    u, u_node_proportions, v_node_proportions = _weighted_unifrac(
        u_node_counts, v_node_counts, u_total_count, v_total_count,
        branch_lengths)

    c = _weighted_unifrac_branch_correction(
        node_to_root_distances, u_node_proportions, v_node_proportions)

    return u / c


def _weighted_unifrac_branch_correction(node_to_root_distances,
                                        u_node_proportions,
                                        v_node_proportions):
    return float(K.sum((node_to_root_distances.ravel() *
                        (u_node_proportions + v_node_proportions))))


def _weighted_unifrac(u_node_counts, v_node_counts, u_total_count,
                      v_total_count, branch_lengths):
    if u_total_count > 0:
        # convert to relative abundances if there are any counts
        u_node_proportions = u_node_counts / u_total_count
    else:
        # otherwise, we'll just do the computation with u_node_counts, which
        # is necessarily all zeros
        u_node_proportions = 1.0 * u_node_counts

    if v_total_count > 0:
        v_node_proportions = v_node_counts / v_total_count
    else:
        v_node_proportions = 1.0 * v_node_counts

    wu = float(K.sum((branch_lengths *
                      tnp.absolute(u_node_proportions - v_node_proportions))))
    return wu, u_node_proportions, v_node_proportions

