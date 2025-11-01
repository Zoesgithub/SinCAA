from peptide.amino_acid import AminoAcid
from openfold.np.residue_constants import restype_3to1
import numpy as np
from openfold.data.mmcif_parsing import MmcifObject
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from peptide.graph_utils import add_node_num_feats, build_conn_from_distance


class Chain():
    def __init__(
        self,
        chain_id: str,
        mmcif_seq_res_to_strcture: list,
        mmcif_chain_to_seqres: list = None,
        mmcif_structure: object = None,
        mmcif_object: MmcifObject = None,
        emb_path="vocab_emb"
    ) -> None:
        self.chain_id = chain_id
        self.emb_path=emb_path
        self.connections = None

        if mmcif_seq_res_to_strcture is not None:
            assert mmcif_chain_to_seqres is not None, "mmcif_inter_conn must be set when mmcif file is used"
            assert mmcif_structure is not None, "mmcif_chain_positions must be set when mmcif file is used"
            self.parse_mmcif_parsed_chain(mmcif_seq_res_to_strcture=mmcif_seq_res_to_strcture,
                                          mmcif_chain_to_seqres=mmcif_chain_to_seqres, mmcif_structure=mmcif_structure, mmcif_object=mmcif_object)


    def get_exist_atom_name(self, mmcif_object: MmcifObject, same_seq_chain_ids: set, aa_idx: int, aa_name: str):
        ret = None
        ori_non_H = 0
        for chain_id in same_seq_chain_ids:
            aa = mmcif_object.seqres_to_structure[chain_id][aa_idx]
            if aa.is_missing or aa_name != aa.name:
                continue
            structure = mmcif_object.structure[chain_id][(
                aa.hetflag,  aa.position.residue_number, aa.position.insertion_code)]
            atom_names = [atom.get_name() for atom in structure]
            num_non_H = sum([1 if not _.upper().startswith(
                "H") else 0 for _ in atom_names])
            if num_non_H > ori_non_H:
                ori_non_H = num_non_H
                ret = set(atom_names)
        return ret

    def parse_mmcif_parsed_chain(
            self,
            mmcif_seq_res_to_strcture: list,
            mmcif_chain_to_seqres: list,
            mmcif_structure: object,
            mmcif_object: MmcifObject,
    ) -> None:

        assert not hasattr(self, "_amino_acid_seq")
        assert not hasattr(self, "_non_standard_num")
        assert not hasattr(self, "_dna_rna_num")

        self._amino_acid_seq = {}
        self._dna_rna_num = 0
        self._non_standard_num = 0

        num_res = len(mmcif_chain_to_seqres)
        same_seq_chain_ids = set()
        for k in mmcif_object.chain_to_seqres:
            if mmcif_object.chain_to_seqres[k] == mmcif_chain_to_seqres:
                same_seq_chain_ids.add(k)
        
        init_residue_index=list(range(num_res))
        
        for idx in init_residue_index:
            aa = mmcif_seq_res_to_strcture[idx]
            if aa.is_missing:
                structure = None
            else:
                structure = mmcif_structure[(
                    aa.hetflag, aa.position.residue_number, aa.position.insertion_code)]

            self._amino_acid_seq[idx] = AminoAcid(
                aa_name=aa.name,
                aa_idx=idx,
                structure=structure,
                include_default_oh_as_exist_atom=False,
                is_missing=aa.is_missing,
                init_exist_atoms=self.get_exist_atom_name(
                    mmcif_object=mmcif_object, same_seq_chain_ids=same_seq_chain_ids, aa_idx=idx, aa_name=aa.name),
                #emb_path=self.emb_path # unused 

            )
            if aa.name not in restype_3to1:
                self._non_standard_num += 1
           

    def __getitem__(self, idx) -> AminoAcid:
        return self._amino_acid_seq[idx]

    def get_num_non_standard(self) -> int:
        return self._non_standard_num

    def get_graph_with_gt(self, residue_index=None) -> dict:
        if residue_index is None:
            residue_index = sorted(self._amino_acid_seq.keys())
        ret = None
        num_nodes = 0

        default_CO_and_N = {}
       
        for idx in residue_index:
            aa = self[idx]
            assert aa.index == idx, f'{aa.index} {idx}'
            aa_graph, default_CO, default_N = aa.get_graph_with_gt()
    
            if ret is None:
                ret = aa_graph
            else:
                for k in list(ret.keys()):
                    if k in add_node_num_feats:
                        v= aa_graph[k]+num_nodes
                        v[aa_graph[k]<0]=-1
                        ret[k] = np.concatenate([
                            ret[k], v
                        ], 0)
                    else:
                        ret[k] = np.concatenate([ret[k], aa_graph[k]], 0)
            
            if default_CO is not None and default_N is not None:
                default_CO_and_N[idx] = [
                    default_CO+num_nodes, default_N+num_nodes]
            elif default_N is not None:
                default_CO_and_N[idx] = [
                    None, default_N+num_nodes]

            num_nodes = len(ret["nodes_int_feats"])
            
        
        updated_graph = build_conn_from_distance(
            init_graph=ret, connections=self.connections, amino_acid_seq=self._amino_acid_seq, default_CO_and_N=default_CO_and_N)
        for k in updated_graph:
            if k not in ret:
                ret[k] = np.zeros((len(ret["edges"]),) +
                                  updated_graph[k].shape[1:])
            ret[k] = np.concatenate([ret[k], updated_graph[k]], 0)
        ret["chain_names"] = np.array([self.chain_id for _ in ret["aa_names"]])
        ret["nodes_chain_length"] = np.array(
            [len(self._amino_acid_seq) for _ in ret["nodes_int_feats"]])
        
        return ret