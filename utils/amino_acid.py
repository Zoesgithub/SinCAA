from rdkit import Chem
import os
import numpy as np
import utils.align_utils as autils
import utils.data_constants as ud_constants
import scipy
from utils.similarity_utils import aa_pattern
import rdkit

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE
    ]
}

class AminoAcid():
    def __init__(
        self,
        aa_name: str = None,
        aa_idx: int = None,
        ligand_path: str = None,
        smiles: str = None,
        include_default_oh_as_exist_atom: bool = False,
        is_missing: bool = False,
        init_exist_atoms: set = None,
        mol=None
    ) -> None:
        self.ligand_path = ligand_path
        self.smiles = smiles

        self.aa_name = aa_name
        self.index = aa_idx
        self.include_default_oh_as_exist_atom = include_default_oh_as_exist_atom
        self.init_exist_atoms = init_exist_atoms
        self.is_missing = is_missing

        self._exist_atoms = []
        self._pseudo_backbones = []
        self.is_ion = False
        self.has_conf = False
        if smiles is not None:
            self.init_from_smiles(smiles)
        elif mol is not None:
            self.init_from_mol(mol)
        else:
            self.init_from_sdf_files()

    def _load_from_smiles(self, smiles):
        # load mol from smiles
        self.mol = Chem.MolFromSmiles(
            smiles,)

        assert self.mol is not None
        
    def _load_from_mol(self, mol):
        self.mol=mol
        assert self.mol is not None

    def _load_from_sdf(self):
        try:
            sdf_path = os.path.join(self.ligand_path, f"{self.aa_name}.mol")
            self.mol = Chem.MolFromMolFile(
                sdf_path,  sanitize=True, removeHs=False)
        except:
            sdf_path = os.path.join(self.ligand_path, f"{self.aa_name}.sdf")
            self.mol = Chem.MolFromMolFile(
                sdf_path,  sanitize=True, removeHs=False)
        # load mol from sdf
        
        self.has_conf = True
        assert self.mol is not None, sdf_path
        
    def _init_atom_group_pe(self):
        self.exist_atoms = sorted([a.GetIdx()
                                   for a in self.mol.GetAtoms() if a.GetSymbol() != "H"])
        self.non_H_atoms = len(self.exist_atoms)
        self.pe, self.permutation=autils.get_atom_groups(self.mol, self.exist_atoms)
        if self.has_conf:
            for v in self.mol.GetConformers():
                conf = v
                self._default_positions=conf.GetPositions()

    def _init_gt_positions(self):
        # node feats
        num_atoms=self.non_H_atoms
        self._map_from_inner_idx_to_output_idx = {}
        self._out_default_positions = np.zeros((num_atoms, 3))
        self._node_int_feats = np.zeros((num_atoms, 3), dtype=int)
        self._node_float_feats = np.zeros((num_atoms, 4), dtype=float)
        self._out_pe=np.zeros((num_atoms, 1), dtype=int)
        self._out_permutation=np.zeros((num_atoms, 20), dtype=int)-1
        match_pattern = self.mol.GetSubstructMatches(Chem.MolFromSmarts(aa_pattern))
        self.isaa=False
        if len(match_pattern)>0:
            self.isaa=True
            N, CA, CO = match_pattern[0][0], match_pattern[0][1], match_pattern[0][2]
            self._map_from_inner_idx_to_output_idx[match_pattern[0][0]] =0
            self._map_from_inner_idx_to_output_idx[match_pattern[0][-1]]=self.non_H_atoms-1
            
        # edge feats
        edges = []
        edge_attrs = []

        # init node feats
        
        exists_atom_idx = self.exist_atoms
        assert len(exists_atom_idx) == self.non_H_atoms
        for i in range(self.non_H_atoms):
            inner_idx = exists_atom_idx[i]
            if inner_idx not in self._map_from_inner_idx_to_output_idx:
                self._map_from_inner_idx_to_output_idx[inner_idx] = len(self._map_from_inner_idx_to_output_idx)
            i=self._map_from_inner_idx_to_output_idx[inner_idx] 
            if self.has_conf:
                self._out_default_positions[i] = self._default_positions[inner_idx]

            atom = self.mol.GetAtomWithIdx(inner_idx)
            self._node_int_feats[i, 0] = atom.GetAtomicNum()
            self._node_int_feats[i,
                                 1] = ud_constants.CIPCodeDict[self.cip_info[inner_idx]]
            self._node_int_feats[i,
                                 2] = atom.GetFormalCharge()+10

            self._node_float_feats[i, 0] = ud_constants.PeriodicTable.GetAtomicWeight(
                atom.GetAtomicNum())
            self._node_float_feats[i, 1] = ud_constants.PeriodicTable.GetRvdw(
                atom.GetAtomicNum())
            self._node_float_feats[i, 2] = ud_constants.PeriodicTable.GetRb0(
                atom.GetAtomicNum())
            self._node_float_feats[i, 3] = ud_constants.PeriodicTable.GetNOuterElecs(
                atom.GetAtomicNum())
            self._out_pe[i, 0]=self.pe[inner_idx]
            assert self.pe[inner_idx]>-1, f"encounter error in pe {self.pe}"
            
        '''for i in range(self.non_H_atoms):
            inner_idx = exists_atom_idx[i]
            assert self._map_from_inner_idx_to_output_idx[inner_idx]==i
            inner_permutation=self.permutation[inner_idx]
            for j in range(min(self._out_permutation.shape[-1], inner_permutation.shape[-1])):
                if inner_permutation[j] in self._map_from_inner_idx_to_output_idx:
                    self._out_permutation[i, j]=self._map_from_inner_idx_to_output_idx[inner_permutation[j]]'''
                    
        assert self._out_pe.min()>-1, f"encounter error in check out pe {self._out_pe}"
        # init edge feats
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if start in self._map_from_inner_idx_to_output_idx and end in self._map_from_inner_idx_to_output_idx:
                start_save_idx = self._map_from_inner_idx_to_output_idx[start]
                end_save_idx = self._map_from_inner_idx_to_output_idx[end]
                edges.append([start_save_idx, end_save_idx])
                edges.append([end_save_idx, start_save_idx])
                edge_attrs.extend(
                    [[ud_constants.MolBondDict[str(bond.GetBondType())], allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())], [ud_constants.MolBondDict[str(bond.GetBondType())], allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]])
                
        # add pseudo edges
        '''for i in range(self.non_H_atoms):
            edges.append([i, self.non_H_atoms])
            edges.append([self.non_H_atoms, i])
            edge_attrs.extend([ud_constants.MolBondDict["PSEUDO"],
                              ud_constants.MolBondDict["PSEUDO"]])'''
        self._edges = np.array(edges, dtype=int).reshape(-1, 2)
        self._edge_attrs = np.array(edge_attrs, dtype=int)#.reshape([-1, 2])
        
        # centralization of coordinates
        self._out_default_positions=self._out_default_positions-self._out_default_positions.mean(0, keepdims=True)
        
        # aug with random rotation
        random_rot=scipy.spatial.transform.Rotation.random()
        self._out_default_positions=random_rot.apply(self._out_default_positions)

    def init_from_smiles(self, smiles):
        self._load_from_smiles(smiles=smiles)
        self.cip_info = {atom.GetIdx(): atom.GetPropsAsDict()[
            '_CIPCode'] if '_CIPCode' in atom.GetPropsAsDict() else "N" for atom in self.mol.GetAtoms()}
        self._init_atom_group_pe()
        
    def init_from_mol(self, mol):
        self._load_from_mol(mol=mol)
        self.cip_info = {atom.GetIdx(): atom.GetPropsAsDict()[
            '_CIPCode'] if '_CIPCode' in atom.GetPropsAsDict() else "N" for atom in self.mol.GetAtoms()}
        self._init_atom_group_pe()

    def init_from_sdf_files(self):
        self._load_from_sdf()
        self.cip_info = {atom.GetIdx(): atom.GetPropsAsDict()[
            '_CIPCode'] if '_CIPCode' in atom.GetPropsAsDict() else "N" for atom in self.mol.GetAtoms()}
        self._init_atom_group_pe()

    def get_graph_with_gt(
        self
    ) -> dict:
        self._init_gt_positions()
        ret= {
            "nodes_int_feats": self._node_int_feats,
            "nodes_float_feats": self._node_float_feats,
            "edges": self._edges,
            "edge_attrs": self._edge_attrs#.reshape(-1, 2),
        }
        return ret
