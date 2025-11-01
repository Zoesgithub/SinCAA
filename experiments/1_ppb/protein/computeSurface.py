# -*- coding: utf-8 -*-
"""
@File   :  computeSurface.py
@Time   :  2024/02/05 16:00
@Author :  Yufan Liu
@Desc   :  Essential parts for compute surface, refer to MaSIF preprocessing
"""

import os
from subprocess import PIPE, Popen

import Bio.PDB as biopdb
import numpy as np
from Bio.SeqUtils import IUPACData
from pymol import cmd

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]


def protonate(args, infilename, outfilename, suffix="pdb"):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file.
    # Remove protons first, in case the structure is already protonated

    assert os.path.exists(infilename) == True, "PDB do not exist"
    reduce_arg = [args.REDUCE_BIN, "-Trim", infilename]
    p2 = Popen(reduce_arg, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(outfilename, "w")
    outfile.write(stdout.decode("utf-8").rstrip())
    outfile.close()
    # Now add them again.
    reduce_arg = [args.REDUCE_BIN, "-HIS", outfilename]
    p2 = Popen(reduce_arg, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(outfilename, "w")
    outfile.write(stdout.decode("utf-8"))
    outfile.close()


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, "r"):
        if line[:6] == "SEQRES":
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


# Exclude disordered atoms.
class NotDisordered(biopdb.Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


def extractPDB(infilename, outfilename, chain_ids=None):
    chain_ids = chain_ids.split("-")
    # extract the chain_ids from infilename and save in outfilename.
    parser = biopdb.PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = biopdb.Selection.unfold_entities(struct, "M")[0]
    chains = biopdb.Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = biopdb.StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()
    modified_amino_acids = find_modified_amino_acids(infilename)

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm

    for chain in model:
        # print(f"chain id is {chain.get_id()}")
        if chain_ids == None or chain.get_id() in chain_ids:

            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0] == "W":  # exclude water
                    continue
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = biopdb.PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())


def generate_xyzr(infilename, outfilename, selection="all"):
    cmd.reinitialize()  # lyf This is important, load the previous strucutre otherwise.
    cmd.load(infilename)
    with open(outfilename, "w") as f:
        model = cmd.get_model(selection)
        for atom in model.atom:
            insertion = "x"  # insertion will be
            x, y, z = atom.coord
            radius = atom.vdw
            if any(c.isalpha() for c in atom.resi):
                insertion = atom.resi[-1]
                res_id = atom.resi[:-1]
            else:
                res_id = atom.resi
            f.write(f"{x} {y} {z} {radius} 1 {atom.chain}_{res_id}_{insertion}_{atom.resn}_{atom.name} \n")
            # the last field is important to denote the atom and residue for feature computing

    # print(f"File '{outfilename}' has been written with {len(model.atom)} atoms.")
    return outfilename


def read_msms(file_root):
    vertfile = open(file_root + ".vert", errors="ignore")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face", errors="ignore")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id


def generate_surface(args, infilename, outfilename, cache=False):
    # pymol to generate xyzr, and MSMS to generate vertex files
    generate_xyzr(infilename, outfilename)
    file_base = "".join(infilename.split(".")[:-1])
    msms_arg = [
        args.MSMS_BIN,
        "-density",
        "3.0",
        "-hdensity",
        "3.0",
        "-probe",
        "1.5",
        "-if",
        outfilename,
        "-of",
        file_base,
        "-af",
        file_base,
    ]
    # print msms_bin+" "+`args`
    # print(msms_arg)
    p2 = Popen(msms_arg, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base + ".area")
    next(ses_file)  # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    # Remove temporary files.
    if not cache:
        os.remove(file_base + ".area")
        os.remove(file_base + ".xyzrn")
        os.remove(file_base + ".vert")
        os.remove(file_base + ".face")
    return vertices, faces, normals, names, areas
