# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:08:54 2020

@author: Osama
"""

from Bio import PDB
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation
import numpy as np

num_rbf = 16
num_rbf_bind = 16
num_rbf_rot = 3
        
number_pos_encoding = 16
nearest_neighbors = 30
        
min_dis = 0
max_dis = 20
        
min_dis_bind = 0
max_dis_bind = 100
        
min_dis_rot = 0
max_dis_rot = 6
        
node_features = 20 + 6 + 3 + num_rbf_rot
edge_features = 3 + 4 +num_rbf + number_pos_encoding


def valid_backbone(structure):
    
    
    atom_types = np.array(structure["atom_type"])
    
    atom_indices = np.logical_or(atom_types == "CA", atom_types == "N")
    
    atom_indices = np.logical_or(atom_types == "C", atom_indices)
 
    coordinates = np.array(structure["coordinates"])[atom_indices]
   
    if np.any(atom_types[atom_indices][0::3] != "N"):
     
        print(atom_types[atom_indices][0::3])
    
    if np.any(atom_types[atom_indices][1::3] != "CA"):
      
        print(atom_types[atom_indices][1::3])
    
    if np.any(atom_types[atom_indices][2::3] != "C"):
        
        print(atom_types[atom_indices][2::3])
        
   
    if len(coordinates) != 3*len(np.unique(structure["residue_number"])):
        
        return False
    
    return True


def calc_dihedral_angles(protein_atom_types, protein_coordinates, chain_ids, ordered_chain_ids, residue_index, map_from_chain_id_to_seq):
    
    #filter out non C CA and N atoms
    ret=[]
    
    for cid in ordered_chain_ids:
        pos=chain_ids==cid
        s=map_from_chain_id_to_seq[cid][1]
        atom_indices = np.logical_or(protein_atom_types == "CA", protein_atom_types == "N")
        atom_indices = np.logical_or(protein_atom_types == "C", atom_indices)
        atom_indices=atom_indices&pos
        ri=residue_index[atom_indices]
        # clean pos to make sure each residue has 3 pos
        ri_count=np.zeros(residue_index.max()+1, dtype=int)
        for v in ri:
            ri_count[v]+=1
        atom_indices=atom_indices&pos&(ri_count[residue_index]==3)
        ri=residue_index[atom_indices]
        
        coordinates = protein_coordinates[atom_indices]
            
        # compute vector normals of planes defined by bonds
        
        bond_vectors = coordinates[1:] - coordinates[:-1]
        
        bond_vectors = bond_vectors/np.linalg.norm(bond_vectors, 
                                                axis=1).reshape((len(bond_vectors),1))
        
        n_1 = np.cross(bond_vectors[1:-1], bond_vectors[2:])
        
        n_2 = np.cross(bond_vectors[:-2], bond_vectors[1:-1])
        
        n_1 = n_1/np.linalg.norm(n_1, axis=1).reshape(len(n_1),1)
        
        n_2 = n_2/np.linalg.norm(n_2, axis=1).reshape(len(n_2),1)
        
        # compute angle between the two planes
        
        dot = np.sum(n_1*n_2, axis=1)
        
        # clip for numerical stability
        dot = np.clip(dot, 1e-6, 1-1e-6)
        
        angles = np.arccos(dot)
        
        angles = np.pad(angles, (1, 2), mode="constant", constant_values=(0,0))
        
        angles = angles.reshape((len(angles)//3, 3))
        
        angles = np.concatenate((np.sin(angles), np.cos(angles)),1)
        
        save_angles=np.zeros([len(s), 6])
        unique_ri=sorted(np.unique(ri)) # sorted
        assert len(unique_ri)==len(angles), f'{len(unique_ri)} {len(angles)}'
        save_angles[unique_ri]=angles
        ret.append(save_angles)
        
    return np.concatenate(ret, 0)

def get_nearest_neighbor_distances(protein_atom_type, protein_coordinates,reindex_chain_id, chain_ids, ordered_chain_ids, residue_index, map_from_chain_id_to_seq):
    
    # find the k nearest neighbors for each residue and their distance
    atom_types = protein_atom_type
        
    ca_indices = atom_types == "CA"
        
    ca_coords = protein_coordinates[ca_indices]
    
    pairwise_distances = squareform(pdist(ca_coords))
    ca_ri=reindex_chain_id[ca_indices]
    
    # build map
    map_from_rid_to_saveid=np.zeros(reindex_chain_id.max()+1, dtype=int)-1
    bias=0
    for cid in ordered_chain_ids:
        rcd=reindex_chain_id[chain_ids==cid]
        ri=residue_index[chain_ids==cid]
        for a,b in zip(rcd, ri):
            if map_from_rid_to_saveid[a]==-1:
                map_from_rid_to_saveid[a]=b+bias
            else:
                assert map_from_rid_to_saveid[a]==b+bias
        bias+=len(map_from_chain_id_to_seq[cid][1])
    assert bias==sum([len(_[1]) for _ in map_from_chain_id_to_seq.values()])
    
    retdistance=np.zeros([bias, nearest_neighbors])
    retneighbor_indices=np.zeros([bias, nearest_neighbors])
    
    for index, row in enumerate(pairwise_distances):
        
        current_indices = np.argpartition(row, nearest_neighbors)
        
        retneighbor_indices[map_from_rid_to_saveid[ca_ri[index]]] =map_from_rid_to_saveid[ca_ri[current_indices[:nearest_neighbors]]]
        retdistance[map_from_rid_to_saveid[ca_ri[index]]] = row[current_indices[:nearest_neighbors]]
       
    return retdistance, retneighbor_indices

def get_rotamer_locations_and_distances(protein_atom_type, protein_coordinates, residue_number, reindex_residue_index, chain_ids, ordered_chain_ids, map_from_chain_id_to_seq):
     # build map
    map_from_rid_to_saveid=np.zeros(reindex_residue_index.max()+1, dtype=int)-1
    bias=0
    for cid in ordered_chain_ids:
        rcd=reindex_residue_index[chain_ids==cid]
        ri=residue_number[chain_ids==cid]
        for a,b in zip(rcd, ri):
            if map_from_rid_to_saveid[a]==-1:
                map_from_rid_to_saveid[a]=b+bias
            else:
                assert map_from_rid_to_saveid[a]==b+bias
        bias+=len(map_from_chain_id_to_seq[cid][1])
    assert bias==sum([len(_[1]) for _ in map_from_chain_id_to_seq.values()])
    
    # get the location and distance of the sidechain centroid from each residue
    
    atom_types =protein_atom_type
        
    ca_indices = atom_types == "CA"
    
    atom_indices = np.logical_and(atom_types != "CA", atom_types != "N")

    atom_indices = np.logical_and(atom_types != "C", atom_indices)
    
    atom_indices = np.logical_and(atom_types != "O", atom_indices)
    
    
    rotamer_ca_cord = np.zeros([bias, 3])
    rotamer_distances = np.zeros([bias])
    rotamer_locations=np.zeros([bias, 3])
    for number in np.unique(reindex_residue_index):
        mapped_index=map_from_rid_to_saveid[number]
        caid=ca_indices&(reindex_residue_index==number)
        if not caid.max(): # missing ca
            print("encountering mssing ca in get_rotamer_locations_and_distances")
            continue
        sideid=atom_indices&(reindex_residue_index==number)
        rotamer_ca_cord[mapped_index]=protein_coordinates[caid]
        if sideid.max():
            rotamer_distances[mapped_index]=np.linalg.norm(protein_coordinates[caid]-protein_coordinates[sideid].mean(0))
            rotamer_locations[mapped_index]=protein_coordinates[sideid].mean(0)
        else:
            rotamer_locations[mapped_index]=protein_coordinates[caid]
            
    return rotamer_locations, rotamer_distances, rotamer_ca_cord
  

def get_orientation_features(all_ca_coords, neighbor_indices,
                              rotamer_locations, ordered_chain_ids, map_from_chain_id_to_seq):
    
    # get orientation features for each node/edge corresponding to residues
    # in the protein
    start=0
    all_Os=[]
    for cid in ordered_chain_ids:
        l=len(map_from_chain_id_to_seq[cid][1])
        ca_coords=all_ca_coords[start:start+l]
        start+=l
        virtual_bonds = ca_coords[1:] - ca_coords[:-1]
        
        b1 = virtual_bonds[:-1]
        
        b0 = virtual_bonds[1:]

        # get norm of vectors before and after each residue
        n = np.cross(b1, b0) 
        
        n = n/np.clip(np.linalg.norm(n, axis=1),a_min=1e-6, a_max=float("inf")).reshape(len(n),1)
        
        # get negative bisector
        o = b1 - b0
        
        o = o/np.clip(np.linalg.norm(o, axis=1), 1e-6, float("inf")).reshape(len(o),1)
        
        O = np.concatenate((o, n, np.cross(o, n)), axis=1)
        
        # add 0s for the first and last residue
        O = np.pad(O, ((1, 1),(0,0)), mode="constant", constant_values=0)
        all_Os.append(O)
    O=np.concatenate(all_Os, 0)
    neighbor_directions = np.zeros(neighbor_indices.shape + (3,))
    
    neighbor_orientations = np.zeros(neighbor_indices.shape + (4,))
    
    rotamer_directions = np.zeros((len(rotamer_locations), 3))
    
    
    
    for residue_number, orientation in enumerate(O):
       
        
        # get neighbor indicies
        
        adjacent_indices = neighbor_indices[residue_number].astype(int)
        
        # calculate pairwise CA directios
        displacement = all_ca_coords[residue_number] - all_ca_coords[adjacent_indices]
        
        directions = np.matmul(orientation.reshape((3,3)), displacement.T).T
        
        norm = np.linalg.norm(directions, axis=1).reshape(len(directions),1)
        
        directions = np.divide(directions,norm,where=norm!=0)
        
        neighbor_directions[residue_number] = directions
        
        # calculate rotamer centroid direction
        displacement = all_ca_coords[residue_number] - rotamer_locations[residue_number]
        
        directions = np.matmul(orientation.reshape((3,3)), displacement.T).T
        
        norm = np.linalg.norm(directions*1.0)
        
        directions = np.divide(directions,norm,where=norm!=0)
        
        rotamer_directions[residue_number] = directions
        
        # calculate relative orientation of coordinate systems
        neighbor_matricies = O[adjacent_indices].reshape((-1, 3, 3))
        
        
        rotation_matricies = np.matmul(orientation.reshape((3,3)), 
                                    np.transpose(neighbor_matricies, (0,2,1)))
       
        rotation_matricies = Rotation.from_matrix(rotation_matricies)
        
        rotation_matricies = rotation_matricies.as_quat()
        
        norm = np.linalg.norm(rotation_matricies, axis=1).reshape(len(rotation_matricies),1)
        
        rotation_matricies = np.divide(rotation_matricies,norm,where=norm!=0)
        
        neighbor_orientations[residue_number] = rotation_matricies
  
       
    return neighbor_directions, neighbor_orientations, rotamer_directions


def rbf(distances, rotamer_distance=False, binding_site_distance=False):
    
    # lift the input distances to a radial basis
    
    if not rotamer_distance and not binding_site_distance:
        min_dist = min_dis
        max_dist = max_dis
        counts = num_rbf
    elif rotamer_distance:
        min_dist = min_dis_rot
        max_dist = max_dis_rot
        counts = num_rbf_rot
    elif binding_site_distance:
        min_dist = min_dis_bind
        max_dist = max_dis_bind
        counts = num_rbf_bind
        
    means = np.linspace(min_dist, max_dist, counts)
    
    std = (max_dist - min_dist)/counts
    
    distances = np.repeat(np.expand_dims(distances, axis=len(distances.shape)), 
                          counts, len(distances.shape))
    
    distances = np.exp(-((distances-means)/std)**2)
    
    return distances

def positional_embedding(indices):
    
    # performs a positional embedding for the edges of the input
    
    differences = indices - np.arange(len(indices)).reshape(len(indices),1)
    
    result = np.exp(np.arange(0, number_pos_encoding,2)*-1*(np.log(10000)/number_pos_encoding))
    
    differences = np.repeat(np.expand_dims(differences, axis=len(differences.shape)),
                            number_pos_encoding/2, len(differences.shape))
    
    result = differences*result
    
    result = np.concatenate((np.sin(result), np.cos(result)),2)
    
    return result



def get_features(parsed_structure):
    
    # store relevant pdb information in dictionary for easy access
    protein_residues = protein_chain.get_list()
    
    protein_information = {}
    protein_information["coordinates"] = []
    protein_information["residue_number"] = []
    protein_information["residue_type"] = []
    protein_information["atom_type"] = []
    protein_information["actual_number"] = []
    
    
    for index, res in enumerate(protein_residues):
        
        # skip non standard residues
        if not PDB.Polypeptide.is_aa(res.get_resname(), standard=True):
            
            continue
        
        resname = res.get_resname()
        resnumber = index
       
        all_atoms = [atom.get_name().strip() for atom in res.get_atoms()]
        
        
        # skip residues lacking a backbone antom
        if "C" not in all_atoms or "CA" not in all_atoms or "N" not in all_atoms:
            continue
        
  
        n_atom = []
        ca_atom = []
        c_atom = []
        for a_i, atom in enumerate(res.get_atoms()):
            if atom.get_name().strip() != "H":
                
                if atom.get_name().strip() == "N":
                    n_atom.append(atom.get_coord())
                    n_atom.append(resnumber)
                    n_atom.append(res.get_id()[1])
                    n_atom.append(resname)
                    n_atom.append(atom.get_name().strip())
                    continue
                elif atom.get_name().strip() == "CA":
                    ca_atom.append(atom.get_coord())
                    ca_atom.append(resnumber)
                    ca_atom.append(res.get_id()[1])
                    ca_atom.append(resname)
                    ca_atom.append(atom.get_name().strip())
                    continue
                elif atom.get_name().strip() == "C":
                    c_atom.append(atom.get_coord())
                    c_atom.append(resnumber)
                    c_atom.append(res.get_id()[1])
                    c_atom.append(resname)
                    c_atom.append(atom.get_name().strip())
                    continue
                protein_information["coordinates"].append(atom.get_coord())
                protein_information["residue_number"].append(resnumber)
                protein_information["actual_number"].append(res.get_id()[1])
                protein_information["residue_type"].append(resname)
                protein_information["atom_type"].append(atom.get_name().strip())
                
                
        protein_information["coordinates"].append(n_atom[0])
        protein_information["residue_number"].append(n_atom[1])
        protein_information["actual_number"].append(n_atom[2])
        protein_information["residue_type"].append(n_atom[3])
        protein_information["atom_type"].append(n_atom[4])
        
        protein_information["coordinates"].append(ca_atom[0])
        protein_information["residue_number"].append(ca_atom[1])
        protein_information["actual_number"].append(ca_atom[2])
        protein_information["residue_type"].append(ca_atom[3])
        protein_information["atom_type"].append(ca_atom[4])
        
        protein_information["coordinates"].append(c_atom[0])
        protein_information["residue_number"].append(c_atom[1])
        protein_information["actual_number"].append(c_atom[2])
        protein_information["residue_type"].append(c_atom[3])
        protein_information["atom_type"].append(c_atom[4])
                

    peptide_information = {}
    peptide_information["coordinates"] = []
    peptide_information["residue_number"] = []
    peptide_information["residue_type"] = []
    peptide_information["atom_type"] = []
    peptide_information["actual_number"] = []
    
    
                
    # check that the backbones are valid for both molecules
    
    if not valid_backbone(protein_information):
        raise ValueError("The backbone of the specified chain is not well formed")
        return
    
   
    
    
    
   
    
    
    
    # define array for node and edge features
    nodes = np.zeros((len(np.unique(protein_information["residue_number"])),
                      node_features))
    edges = np.zeros((len(np.unique(protein_information["residue_number"])), 
                      nearest_neighbors,
                      edge_features))
    
 
    
    # encode sequence of amino acids as node feature
    
    amino_acid_index = np.array(protein_information["atom_type"]) == "CA"
    
    amino_acid_sequence = np.array(protein_information["residue_type"])[amino_acid_index]
  
    
    for index, amino_acid in enumerate(amino_acid_sequence):
        temp_target = np.zeros(20)
        temp_target[PDB.Polypeptide.d3_to_index[amino_acid]] = 1
        nodes[index, 0:20] = temp_target
        
    # encode dihedral angles for amino acids
        
    angles = calc_dihedral_angles(protein_information) 
    
    nodes[:, 20:26] = angles 
    
    
    # get distances and coordinates for orientation features
    neighbor_distances, neighbor_indices= get_nearest_neighbor_distances(
        protein_information)
   
   
    adjusted_neighbor_distances = rbf(neighbor_distances)
    
    
    edges[:, :, 0:num_rbf] = adjusted_neighbor_distances
    
    
    # get rotamer distances and coordinates for orientation features
    rotamer_locations, rotamer_distances = get_rotamer_locations_and_distances(protein_information)
    
    adjusted_rotamer_distances = rbf(np.array(rotamer_distances),
                                           True)
    
    nodes[:, 26:26+num_rbf_rot] = adjusted_rotamer_distances
    

    # get orientation features
    neighbor_directions, neighbor_orientations, rotamer_directions = get_orientation_features(
            protein_information, neighbor_indices, rotamer_locations)
    
    edges[:, :, num_rbf:num_rbf+3] = neighbor_directions
    edges[:, :, num_rbf+3:num_rbf+7] = neighbor_orientations
    
    
    nodes[:, 26+num_rbf_rot:26+num_rbf_rot+3] = rotamer_directions
    
    edge_embeddings = positional_embedding(neighbor_indices)
    

    
    edges[:, :, num_rbf+7: num_rbf+7+number_pos_encoding] = edge_embeddings
    
    
    return nodes, edges, neighbor_indices
    
        