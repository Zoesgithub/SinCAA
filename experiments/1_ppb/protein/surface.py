# -*- coding: utf-8 -*-
"""
@File   :  surface.py
@Time   :  2024/02/05 15:06
@Author :  Yufan Liu
@Desc   :  Generate protein surface and vertices fingerprints
"""


import Bio.PDB as biopdb
import numpy as np
import torch
from Bio.PDB import Model, Structure
from scipy.spatial import KDTree
from torch_geometric.data import Data

from peptide.mmcif_parsing import parse as mmcif_parse
from protein.chemical_feature import ele2num, generate_charge
from protein.computeSurface import extractPDB, generate_surface, protonate
from protein.geomesh import GeoMesh
from protein.geometric_feature import generate_ddc, generate_shapeindex
from Bio.SeqUtils import seq1

def faces_to_edge(faces):
    edge_index = []
    for face in faces:
        edge_index.append([face[0], face[1]])
        edge_index.append([face[0], face[2]])
        edge_index.append([face[1], face[2]])
        edge_index.append([face[1], face[0]])
        edge_index.append([face[2], face[0]])
        edge_index.append([face[2], face[1]])
    edge_index = np.array(edge_index).T
    edge_index = np.unique(edge_index, axis=1)
    return edge_index


class ChainRemover(biopdb.Select):
    def __init__(self, chain_to_remove):
        self.chain_to_remove = chain_to_remove

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_to_remove:
            return 0
        else:
            return 1


def mmcif_to_surface(args, mmcif_path, file_id, chain, path):
    """This function generate protein surface from mmcif-formated file

    Args:
        args: from arguments file
        mmcif_path: input mmcif file of protein
        file_id: i.e. the input file name
        chain: receptor chains of the complex
        path: path for store processed files

    Returns:
        A PyG graph of generated surface (mesh)
    """
    process_path = path
    chain_split = chain.split("-")
    with open(mmcif_path, "r") as f:
        cif_data = mmcif_parse(file_id=file_id, mmcif_string="".join(f.readlines()))
    cif_data = cif_data.mmcif_object
    chain_to_seqres=cif_data.chain_to_seqres
    seqres_to_structure=cif_data.seqres_to_structure
    cif_struct = cif_data.structure
    select_struct = Structure.Structure("New")
    select_model = Model.Model(0)
    select_struct.add(select_model)

    for ch in cif_struct:
        if ch.get_id() in chain_split:
            select_model.add(ch)

    cif_chains = [x.id for x in select_model]
    trainsit_map=[f"TRAINS{i}" for  i in range(len(cif_chains))]
    chains_map = [chr(i) for i in range(ord("A"), ord("Z") + 1)] + [chr(i) for i in range(ord("a"), ord("z") + 1)][
        : len(cif_chains)
    ]
    map_id_to_idx={}
    seqs=[]
    struc_map=[]
    struc=[]
    for c in select_model:
        idx=cif_chains.index(c.id)
        seqs.append([chain_to_seqres[c.id], c.id])
        struc_map.append(seqres_to_structure[c.id])
        struc.append(cif_data.structure[c.id])
        c.id = trainsit_map[idx]
        map_id_to_idx[c.id]=idx
    for c in select_model:
        c.id = chains_map[map_id_to_idx[c.id]]
    # lyf here delete a sanity check (the chain id cannot be changed casually) in Biopython, be very careful.

    new_chain = []
    
    for c in chain_split:
        new_chain.append(chains_map[cif_chains.index(c)])
    new_chain = "-".join(new_chain)

    pdbio = biopdb.PDBIO()
    # here, only save selected protein chains

    pdbio.set_structure(select_model)
    pdbio.save(f"{process_path}/{file_id}.pdb")

    protonate(args, f"{process_path}/{file_id}.pdb", f"{process_path}/{file_id}_pro.pdb")
    extractPDB(
        f"{process_path}/{file_id}_pro.pdb",
        f"{process_path}/{file_id}.pdb",  # cover the old one from cif
        new_chain,
    )
    vertices, faces, normals, names, areas = generate_surface(
        args, f"{process_path}/{file_id}.pdb", f"{process_path}/{file_id}.xyzrn", cache=False
    )

   
    # downsampling mesh and update features
    mesh = GeoMesh(vertex_matrix=vertices, face_matrix=faces)
    face_number = mesh.current_mesh().face_number()
    mesh.meshing_decimation_quadric_edge_collapse(targetfacenum=int(args.collapse_rate * face_number))
    mesh.meshing_repair_non_manifold_edges()
    mesh.meshing_remove_unreferenced_vertices()
    mesh.apply_coord_taubin_smoothing()
    new_vertices = mesh.vertices
    new_faces = mesh.faces
    mesh_edge_index = faces_to_edge(new_faces)
    atom_dists, atom_types = generate_charge(f"{process_path}/{file_id}.pdb", new_vertices)
    mesh.set_attribute("atom_type", atom_types)
    mesh.set_attribute("atom_dist", atom_dists)
    
    mesh = generate_shapeindex(mesh)
    mesh, mesh_edge_index, mesh_edge_attr = generate_ddc(args, mesh, mesh_edge_index)

    edge_index = torch.from_numpy(mesh_edge_index).to(torch.float32)
    x_feature = torch.from_numpy(mesh.get_attribute("input_feature")).to(torch.float32)
    edge_attr = torch.from_numpy(mesh_edge_attr).to(torch.float32)
    
    # get nearest residue
    atom_coords = []
    atom_names=[]
    residue_indices = []  
    reindex_residue_indices = []  
    residue_type=[]
    chain_ids=[]
    map_from_chain_id_to_seq={}
    reindex_ri=0
    for s, chain, st, chain_id in zip(seqs, struc_map, struc, select_model):
        chain_id=chain_id.id
        map_from_chain_id_to_seq[chain_id]=s
        s=s[0]
        for j in chain:
            residue=chain[j] # zero-base
            assert seq1(residue.name)==s[j], f"{seq1(residue.name)} {s[j]} {j} {s}"
            if not residue.is_missing: 
                structure = st[(residue.hetflag,  residue.position.residue_number, residue.position.insertion_code)]
                for atom in structure:
                    atom_coords.append(atom.get_coord())
                    atom_names.append(atom.name)
                    residue_indices.append(j) 
                    reindex_residue_indices.append(reindex_ri)
                    residue_type.append(residue.name)
                    chain_ids.append(chain_id)
                reindex_ri+=1
    
    atom_coords = np.array(atom_coords)
    residue_indices = np.array(residue_indices)
    reindex_residue_indices=np.array(reindex_residue_indices)
    assert len(atom_coords)>0
    
    tree = KDTree(atom_coords)
    nearest_dist, nearest_atoms = tree.query(new_vertices) 
    nearest_residues = reindex_residue_indices[nearest_atoms] 
    # save
    mesh_graph = Data(x=x_feature, edge_index=edge_index, edge_attr=edge_attr, mesh_vertex=new_vertices,nearest_residue=nearest_residues, nearest_dist=nearest_dist )
    mesh_graph = surface_vertex_feature(f"{process_path}/{file_id}", mesh_graph)
    protein_info=Data(x=atom_coords,reindex_residue_indices=reindex_residue_indices, residue_indices=residue_indices, chain_ids=chain_ids, map_from_chain_id_to_seq=map_from_chain_id_to_seq, atom_names=atom_names, residue_type=residue_type)
    return mesh_graph, protein_info


def surface_vertex_feature(prefix, protein_surface):
    """This function does some feature modification like dMasif

    Args:
        prefix: data name for loading pdb
        protein_surface: raw input feature

    Returns:
        a new PyG graph
    """
    parser = biopdb.PDBParser(QUIET=True)
    struct = parser.get_structure("name", f"{prefix}.pdb")
    struct = struct[0]

    atom_types = []
    atom_coords = []
    for chain in struct:
        for residue in chain:
            for atom in residue:
                atom_types.append(atom.get_id())
                atom_coords.append(atom.get_coord())

    atom_coords = np.stack(atom_coords)
    assert atom_coords.shape[1] == 3

    atom_feature = []
    for name in atom_types:
        if name[:2] in ele2num:
            atom_ = name[:2]
        else:
            atom_ = name[0]
        try:
            atom_feature.append(ele2num[atom_])
        except:
            atom_feature.append(7)  # atom feature will be in dim-8

    atom_feature = np.stack(atom_feature, 0)
    atom_feature = np.expand_dims(atom_feature, axis=-1)
    pro_vertex = protein_surface.mesh_vertex
    kdt = KDTree(atom_coords)
    atom_dist, nearest_atom_idx = kdt.query(pro_vertex, k=16)  # as dMaSIF find nearest 16 points

    atom_feat = atom_feature[nearest_atom_idx]
    protein_surface.atom_feat = atom_feat
    protein_surface.atom_dist = atom_dist
    return protein_surface
