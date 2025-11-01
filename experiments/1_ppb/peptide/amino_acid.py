from rdkit import Chem
import os
import numpy as np
from gemmi import cif
import wget
import torch
#from utils.align_utils import get_atom_groups
PeriodicTable = Chem.GetPeriodicTable()

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

MolBondDict = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "DISULF": 3,
    "AROMATIC": 4,
    'METALC': 5,
    "TRIPLE": 7,
    "COVALE": 1,
    "DATIVE": 1,
    "PSEUDO": 8,
    "AABOND": 9
}

class AminoAcid():
    def __init__(
        self,
        aa_name: str,
        aa_idx: int,
        structure: object,
        precomputed_prefix: str = None,
        include_default_oh_as_exist_atom: bool = False,
        is_missing: bool = False,
        init_exist_atoms: set = None,
        use_default_conf: bool = False
    ) -> None:
        self.precomputed_prefix = precomputed_prefix
        self.use_default_conf = use_default_conf

        self.aa_name = aa_name
        self.index = aa_idx
        self.include_default_oh_as_exist_atom = include_default_oh_as_exist_atom
        self.init_exist_atoms = init_exist_atoms
        self.is_missing = is_missing
        self.sdf_path="raw_data/vocab/"

        self._exist_atoms = []
        self._pseudo_backbones = []
        self.is_ion = False

        self.init_from_cif_and_sdf_files(
            structure=structure)
        

    def _load_from_sdf(self, structure):
        sdf_path = os.path.join(
            self.sdf_path, f"{self.aa_name}_ideal.sdf")

        # load mol from sdf
        self.mol = Chem.MolFromMolFile(
            sdf_path,  sanitize=True, removeHs=False)

        if self.mol is None or self.aa_name in ["UNL", "U5G", "BVA"]:
            self.is_missing = True
            self.aa_name = "GLY"
            # print("Warning: use wild card to replace unknown for ", sdf_path)
            sdf_path = os.path.join(
               self.sdf_path, f"{self.aa_name}_ideal.sdf")
            self.mol = Chem.MolFromMolFile(
                sdf_path,  sanitize=True, removeHs=False)
            structure = None
        return structure
    
    #def _init_atom_group_pe(self):
    #    self.exist_atoms = list(self.name_to_atom_dict.values())
    #    self.pe, self.permutation=get_atom_groups(self.mol, self.exist_atoms)

    def _load_from_cif(self):
        cif_path = os.path.join(
            self.sdf_path, f"{self.aa_name}.cif")
        if not os.path.exists(
                cif_path):
            wget.download(
                "https://files.rcsb.org/ligands/download/{}.cif".format(self.aa_name), cif_path)
        parser = cif.read_file(cif_path)
        block = parser.sole_block()
        self.name_to_atom_idx = {}
        self.alt_name_to_atom_idx = {}
        for aid, aaid, atype, aord in zip(block.find_loop("_chem_comp_atom.atom_id"), block.find_loop("_chem_comp_atom.alt_atom_id"), block.find_loop("_chem_comp_atom.type_symbol"), block.find_loop("_chem_comp_atom.pdbx_ordinal")):
            alt_aid = aaid.replace("\"", "")
            aid = aid.replace("\"", "")
            try:
                aord = int(aord)
            except:
                print(aord)
                continue

            atom_in_mol = self.mol.GetAtomWithIdx(aord-1)
            if not atom_in_mol.GetSymbol().upper() == atype.upper():
                print("Warning: find inconsistant atoms",
                      f'{atom_in_mol.GetSymbol()} {atype} {self.aa_name}')
            self.name_to_atom_idx[aid] = atom_in_mol.GetIdx()
            self.alt_name_to_atom_idx[alt_aid] = atom_in_mol.GetIdx()
        assert len(self.name_to_atom_idx) == len(self.mol.GetAtoms()) or len(
            self.mol.GetAtoms()) == 1, f"{self.name_to_atom_idx} {len(self.mol.GetAtoms())} {self.aa_name}"

        if len(block.find_loop("_chem_comp_atom.atom_id")) == 0 and len(self.mol.GetAtoms()) == 1:  # ions
            atom_in_mol = self.mol.GetAtomWithIdx(0)
            self.is_ion = True
            for atom in self.mol.GetAtoms():
                self.name_to_atom_idx[atom.GetSymbol().upper()
                                      ] = atom_in_mol.GetIdx()
                self.alt_name_to_atom_idx[atom.GetSymbol(
                ).upper()] = atom_in_mol.GetIdx()

    def _parse_structure(self, structure):
        self.positions = {}
        use_alt = False
        self.exist_atom_names = None
        if self.init_exist_atoms is not None:
            non_H = 0
            for atom_name in self.init_exist_atoms:
                if not atom_name.upper().startswith("H") and atom_name not in self.name_to_atom_idx:
                    use_alt = True
                if not atom_name.upper().startswith("H"):
                    non_H += 1
            if non_H > 0:
                self.exist_atom_names = self.init_exist_atoms
        if structure is not None:
            non_H = 0
            for atom in structure:
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                self.positions[atom_name] = [x, y, z]
                if not atom_name.upper().startswith("H") and atom_name not in self.name_to_atom_idx:
                    use_alt = True
                if not atom_name.upper().startswith("H"):
                    non_H += 1
            if self.exist_atom_names is None:
                if non_H == 0:
                    self.exist_atom_names = set(self.name_to_atom_idx.keys())
                else:
                    self.exist_atom_names = set(self.positions.keys())
        else:
            if self.exist_atom_names is None:
                self.exist_atom_names = set(self.name_to_atom_idx.keys())
        if use_alt:
            self.name_to_atom_dict = self.alt_name_to_atom_idx
        else:
            self.name_to_atom_dict = self.name_to_atom_idx
        # remove not exist atoms & H
        for n in list(self.name_to_atom_dict.keys()):
            if n not in self.exist_atom_names:
                self.name_to_atom_dict.pop(n)
            else:
                atom = self.mol.GetAtomWithIdx(self.name_to_atom_dict[n])
                if atom.GetSymbol() == "H":
                    self.name_to_atom_dict.pop(n)
                    self.exist_atom_names.remove(n)

        # get default cooh and nh2
        self.default_CO = None
        self.default_N = None
        self.default_CA=-1
        if "C" in self.name_to_atom_dict:
            self.default_CO = self.name_to_atom_idx["C"]
        if "OXT" in self.name_to_atom_dict and not self.include_default_oh_as_exist_atom and (self.init_exist_atoms is None or "OXT" not in self.init_exist_atoms):
            self.exist_atom_names.remove("OXT")
            self.name_to_atom_dict.pop("OXT")

        if "N" in self.name_to_atom_dict:
            self.default_N = self.name_to_atom_dict["N"]
        if "CA" in self.name_to_atom_dict:
            self.default_CA=self.name_to_atom_dict["CA"]
        
        self.non_H_atoms = len(self.name_to_atom_dict)
        assert self.non_H_atoms > 0, f"{self.non_H_atoms} {self.aa_name} {self.init_exist_atoms}"
        self.atom_to_name_dict = {v: k for k,
                                  v in self.name_to_atom_dict.items()}
        


    def _init_default_position(self):
        for v in self.mol.GetConformers():
            conf=v
            self._default_positions=conf.GetPositions()

    def _init_gt_positions_and_default_positions(self):
        # node feats
        #self._init_atom_group_pe()
        self._has_gt_positions = np.zeros(self.non_H_atoms)
        self._gt_positions = np.zeros((self.non_H_atoms, 3))
        self._out_default_positions = np.zeros((self.non_H_atoms, 3))

        self._node_int_feats = np.zeros((self.non_H_atoms, 3), dtype=int)
        self._node_float_feats = np.zeros(
            (self.non_H_atoms, 4), dtype=float)
        
        self._out_pe=np.zeros((self.non_H_atoms, 1), dtype=int)
        self._out_permutation=np.zeros((self.non_H_atoms, 20), dtype=int)-1

        self._residue_index = np.zeros(self.non_H_atoms, dtype=int)+self.index
        self.ret_atom_names = []
        self.ret_atom_types = []

        # edge feats
        edges = []
        edge_attrs = []

        # init node feats
        self._map_from_inner_idx_to_output_idx = {}
        exists_atom_idx = sorted(list(self.atom_to_name_dict.keys()))
        assert len(exists_atom_idx) == self.non_H_atoms
        
        for i in range(self.non_H_atoms):
            inner_idx = exists_atom_idx[i]
            self._map_from_inner_idx_to_output_idx[inner_idx] = i

        for i in range(self.non_H_atoms):
            inner_idx = exists_atom_idx[i]
            atom_name = self.atom_to_name_dict[inner_idx]

            if atom_name in self.positions:
                self._has_gt_positions[i] = 1
                self._gt_positions[i] = np.array(self.positions[atom_name])

            self._out_default_positions[i] = self._default_positions[inner_idx]
            atom = self.mol.GetAtomWithIdx(inner_idx)
            self._node_int_feats[i, 0] = atom.GetAtomicNum()
            self._node_int_feats[i,
                                 1] = allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
            self._node_int_feats[i, 2]=atom.GetFormalCharge()+10

            self._node_float_feats[i, 0] = PeriodicTable.GetAtomicWeight(
                atom.GetAtomicNum())
            self._node_float_feats[i, 1] = PeriodicTable.GetRvdw(
                atom.GetAtomicNum())
            self._node_float_feats[i, 2] = PeriodicTable.GetRb0(
                atom.GetAtomicNum())
            self._node_float_feats[i, 3] =PeriodicTable.GetNOuterElecs(
                atom.GetAtomicNum())
            
            #self._out_pe[i, 0]=self.pe[inner_idx]
            #assert self.pe[inner_idx]>-1, f"encounter error in pe {self.pe}"
            self.ret_atom_names.append(atom_name)
            self.ret_atom_types.append(atom.GetSymbol())
            
        '''for i in range(self.non_H_atoms):
            inner_idx = exists_atom_idx[i]
            assert self._map_from_inner_idx_to_output_idx[inner_idx]==i
            inner_permutation=self.permutation[inner_idx]
            for j in range(min(self._out_permutation.shape[-1], inner_permutation.shape[-1])):
                if inner_permutation[j] in self._map_from_inner_idx_to_output_idx:
                    self._out_permutation[i, j]=self._map_from_inner_idx_to_output_idx[inner_permutation[j]]
        '''
        # init edge feats
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if start in self._map_from_inner_idx_to_output_idx and end in self._map_from_inner_idx_to_output_idx:
                start_save_idx = self._map_from_inner_idx_to_output_idx[start]
                end_save_idx = self._map_from_inner_idx_to_output_idx[end]
                edges.append([start_save_idx, end_save_idx])
                edges.append([end_save_idx, start_save_idx])
                edge_attrs.extend(
                    [[MolBondDict[str(bond.GetBondType())], allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())], [MolBondDict[str(bond.GetBondType())], allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]])
        self._edges = np.array(edges, dtype=int).reshape(-1, 2)
        self._edge_attrs = np.array(edge_attrs, dtype=int)

        
    def init_from_cif_and_sdf_files(
        self,
        structure: object
    ) -> None:
        # parse mol from sdf
        structure = self._load_from_sdf(structure=structure)

        # load mol from cif, merge cif and sdf
        self._load_from_cif()

        # record chirality info
        self.cip_info = {atom.GetIdx(): atom.GetPropsAsDict()[
            '_CIPCode'] if '_CIPCode' in atom.GetPropsAsDict() else "N" for atom in self.mol.GetAtoms()}

        # parse_structures
        self._parse_structure(structure=structure)
        
        # init default position
        self._init_default_position()    
            

    def get_graph_with_gt(
        self
    ) -> dict:
        self._init_gt_positions_and_default_positions()
        if self.default_CA>-1:
            default_CA=self._map_from_inner_idx_to_output_idx[self.default_CA]
        else:
            default_CA=self.default_CA
        ret = {
            "nodes_int_feats": self._node_int_feats,
            "nodes_float_feats": self._node_float_feats,
            "nodes_residue_index": self._residue_index,

            "gt_positions": self._gt_positions,
            #"default_positions": self._out_default_positions,
            "gt_position_exists": self._has_gt_positions,
            "edges": self._edges,
            "edge_attrs": self._edge_attrs.reshape(-1, 2),

            "atom_names": self.ret_atom_names,
            "atom_types": self.ret_atom_types,
            "aa_names": [self.aa_name for _ in self.ret_atom_names],
            "last_atom_idx":np.array(0).reshape(-1),
            "default_CA":np.array(default_CA).reshape(-1),
            
            #"pe":self._out_pe,
            #"permutation":self._out_permutation,

        }
        return ret, self.default_CO, self.default_N