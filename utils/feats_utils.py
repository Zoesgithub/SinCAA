from typing import Iterable, List, Sequence
from openfold.data import msa_pairing
from openfold.data.msa_pairing import CHAIN_FEATURES, MSA_FEATURES, MSA_PAD_VALUES, SEQ_FEATURES
from openfold.data.templates import Mapping
import numpy as np
from openfold.data.data_pipeline import feature_processing_multimer, MutableMapping
import collections
from utils.rigid_utils import RobustRigid
import torch


def empty_template_feats():
    return {
        "template_nodes_int_feats": np.zeros([2, 3]).astype(int),
        "template_nodes_float_feats": np.zeros([2, 4]).astype(float),
        "template_nodes_residue_index": np.zeros([2]).astype(int),
        "template_gt_positions": np.zeros([2, 3]).astype(float),
        "template_gt_position_exists": np.zeros([2]).astype(int),
        "template_edges": np.zeros([2, 2]).astype(int),
        "template_edge_attrs": np.zeros(2).astype(int),
        "template_projection": np.zeros([2, 3]).astype(int),

    }


TEMPLATE_FEATURES = (
    "template_nodes_int_feats",
    "template_nodes_float_feats",
    "template_nodes_residue_index",
    "template_gt_positions",
    "template_gt_position_exists",
    "template_edges",
    "template_edge_attrs",
    "template_projection",

)
GRAPH_FEATURES = (
    "nodes_int_feats",
    "nodes_float_feats",
    "nodes_chain_length",
    "nodes_residue_index",
    "gt_positions",
    "gt_position_exists",
    "edges",
    "edge_attrs",
    "nodes_asym_id",
    "nodes_sym_id",
    # the chain id from graph data, which might be different from asym id, one-base
    "chain_asym_id",
    # feats related to structure
    "defined_torsion_angle",
    "current_torsion_angles",
    "torsion_angle_is_180_symmetric",
    "default_positions",
    "pseudo_backbone_atoms",

    # feats related to coordinates
    "nodes_affected_by_torsion_angles",
    "nodes_reindex_when_alt_torsion_angles",
    "gt_torsion_angles",
    "gt_torsion_angle_exists",
    "gt_pseudo_backbone_exists",
    "gt_pseudo_backbone",
    "edge_lengths"
)


def get_transform_from_three_points(
    p_neg_x_axis: np.array,
    origin: np.array,
    p_xy_plane: np.array,
    return_rigid=False,
    is_numpy=False
):
    if is_numpy:
        rigid = RobustRigid.from_3_points(torch.tensor(
            p_neg_x_axis), torch.tensor(origin), torch.tensor(p_xy_plane))
    else:
        rigid = RobustRigid.from_3_points(p_neg_x_axis, origin, p_xy_plane)
    if return_rigid:
        return rigid
    if is_numpy:
        return rigid.to_tensor_4x4().detach().numpy()
    else:
        return rigid.to_tensor_4x4()


def get_torsion_angle_from_four_points(
    p_neg_x_axis: np.array,
    origin: np.array,
    p_xy_plane: np.array,
    p_angle: np.array,
    is_numpy=False
):
    local_rigid = get_transform_from_three_points(
        p_neg_x_axis=p_neg_x_axis,
        origin=origin,
        p_xy_plane=p_xy_plane,
        return_rigid=True,
        is_numpy=is_numpy
    )

    p_angle_in_local = local_rigid.invert_apply(
        torch.tensor(p_angle))
    if is_numpy:
        p_angle_in_local = p_angle_in_local.detach().numpy()
    return p_angle_in_local[[2, 1]]  # sin, cos


def process_final(
    np_example: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    "rewrite version of process final, do not filter features"
    """Final processing steps in data pipeline, after merging and pairing."""
    np_example = feature_processing_multimer._correct_msa_restypes(np_example)
    np_example = feature_processing_multimer._make_seq_mask(np_example)
    np_example = feature_processing_multimer._make_msa_mask(np_example)
    return np_example


def _merge_features_from_multiple_chains(
        chains: Sequence[Mapping[str, np.ndarray]],
        pair_msa_sequences: bool) -> Mapping[str, np.ndarray]:
    """Merge features from multiple chains.

    Args:
      chains: A list of feature dictionaries that we want to merge.
      pair_msa_sequences: Whether to concatenate MSA features along the
        num_res dimension (if True), or to block diagonalize them (if False).

    Returns:
      A feature dictionary for the merged example.
    """
    merged_example = {}

    for c in chains:
        if "template_nodes_int_feats" not in c:
            for k in list(c.keys()):
                if "template" in k:
                    c.pop(k)
            c.update(empty_template_feats())

    for feature_name in chains[0]:
        feats = [x[feature_name] for x in chains]
        feature_name_split = feature_name.split('_all_seq')[0]
        if feature_name_split in MSA_FEATURES:
            if pair_msa_sequences or '_all_seq' in feature_name:
                merged_example[feature_name] = np.concatenate(feats, axis=1)
            else:
                merged_example[feature_name] = msa_pairing.block_diag(
                    *feats, pad_value=MSA_PAD_VALUES[feature_name])
        elif feature_name_split in SEQ_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=0)
        elif feature_name_split in TEMPLATE_FEATURES+GRAPH_FEATURES:
            if feature_name_split == "template_projection":

                asym_id = [
                    x["asym_id"] for x in chains]
                for v, a in zip(feats, asym_id):
                    unique_asym_id = np.unique(a)
                    if len(unique_asym_id) == 1:
                        v[..., 2] = unique_asym_id
            elif feature_name_split == "template_edges":
                n = 0
                template_nodes_int_feats = [
                    x["template_nodes_int_feats"] for x in chains]
                for v, a in zip(feats, template_nodes_int_feats):
                    v += n
                    n += len(a)
            merged_example[feature_name] = np.concatenate(feats, axis=0)
        elif feature_name_split in msa_pairing.TEMPLATE_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=1)
        elif feature_name_split in CHAIN_FEATURES:
            merged_example[feature_name] = np.sum(
                x for x in feats).astype(np.int32)
        else:
            merged_example[feature_name] = feats[0]

    return merged_example


def _merge_homomers_dense_msa(
        chains: Iterable[Mapping[str, np.ndarray]]) -> Sequence[Mapping[str, np.ndarray]]:
    """Merge all identical chains, making the resulting MSA dense.

    Args:
      chains: An iterable of features for each chain.

    Returns:
      A list of feature dictionaries.  All features with the same entity_id
      will be merged - MSA features will be concatenated along the num_res
      dimension - making them dense.
    """
    entity_chains = collections.defaultdict(list)
    for chain in chains:
        entity_id = chain['entity_id'][0]
        entity_chains[entity_id].append(chain)

    grouped_chains = []
    for entity_id in sorted(entity_chains):
        chains = entity_chains[entity_id]
        grouped_chains.append(chains)
    chains = [
        _merge_features_from_multiple_chains(chains, pair_msa_sequences=True)
        for chains in grouped_chains]
    return chains


def merge_chain_features(np_chains_list: List[Mapping[str, np.ndarray]],
                         pair_msa_sequences: bool,
                         max_templates: int) -> Mapping[str, np.ndarray]:
    """Merges features for multiple chains to single FeatureDict.

    Args:
      np_chains_list: List of FeatureDicts for each chain.
      pair_msa_sequences: Whether to merge paired MSAs.
      max_templates: The maximum number of templates to include.

    Returns:
      Single FeatureDict for entire complex.
    """
    np_chains_list = msa_pairing._pad_templates(
        np_chains_list, max_templates=max_templates)
    np_chains_list = _merge_homomers_dense_msa(np_chains_list)
    # Unpaired MSA features will be always block-diagonalised; paired MSA
    # features will be concatenated.
    np_example = _merge_features_from_multiple_chains(
        np_chains_list, pair_msa_sequences=False)
    if pair_msa_sequences:
        np_example = msa_pairing._concatenate_paired_and_unpaired_features(
            np_example)
    np_example = msa_pairing._correct_post_merged_feats(
        np_example=np_example,
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences)

    return np_example


def pair_and_merge(
    all_chain_features: MutableMapping[str, Mapping[str, np.ndarray]],
) -> Mapping[str, np.ndarray]:
    """Runs processing on features to augment, pair and merge.

    Args:
      all_chain_features: A MutableMap of dictionaries of features for each chain.

    Returns:
      A dictionary of features.
    """
    feature_processing_multimer.process_unmerged_features(all_chain_features)

    np_chains_list = list(all_chain_features.values())

    pair_msa_sequences = not feature_processing_multimer._is_homomer_or_monomer(
        np_chains_list)

    if pair_msa_sequences:
        np_chains_list = msa_pairing.create_paired_features(
            chains=np_chains_list
        )
        np_chains_list = msa_pairing.deduplicate_unpaired_sequences(
            np_chains_list)
    np_chains_list = feature_processing_multimer.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing_multimer.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing_multimer.MAX_TEMPLATES
    )

    np_example = merge_chain_features(
        np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing_multimer.MAX_TEMPLATES
    )
    np_example = process_final(np_example)
    return np_example
