# -*- coding: utf-8 -*-
"""
@File   :  pdb.py
@Time   :  2024/02/01 15:17
@Author :  Yufan Liu
@Desc   :  Tools for tackling pdb files
"""

import Bio.PDB as biopdb
from Bio.SeqUtils import IUPACData

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]


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


def extractPDB(infilename, outfilename, chain_ids=None, format="pdb"):
    """
    Extract specific chain from PDB file
    """
    # extract the chain_ids from infilename and save in outfilename.
    assert format in ["pdb", "cif"], "Structure file format not supported."
    if format == "pdb":
        parser = biopdb.PDBParser(QUIET=True)
    elif format == "cif":  # tmp
        parser = biopdb.MMCIFParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = biopdb.Selection.unfold_entities(struct, "M")[0]
    chains = biopdb.Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = biopdb.StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

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
