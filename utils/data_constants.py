from openfold.np.residue_constants import restype_3to1
from rdkit import Chem
ResNameDict = {key: i for i, key in enumerate(restype_3to1.keys())}
ExplicitCoDict = {
    "ACE": ["C", None],
    "PHQ": ["C1", "CL1"],
    "MPR": ["C1", None],
    "MAZ": ["C16", None],
    "1BO": ["C4", None],

}
ExplicitNDict = {
    "0HQ": "C1",
    "AEA": "N1",
    "CFT": "C1",
    "T2X": "O28",
    "CF0": "F1"
}
MaxNumOfAtoms = 200
MaxNumOfEdges = 10
NumNodeEmb = 5
NumNodeFloatFeats = 4
NumEdgeEmb = 1
BondThreshold = 1.7
MaxNumOfTorsionAngles = 20
cache_data_path = "data/cache/"

ResNameDict["Nonstandard"] = len(ResNameDict)


PeriodicTable = Chem.GetPeriodicTable()
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
    "AABOND": 9,
    "None":10
}

CIPCodeDict = {
    "N": 1,
    "S": 2,
    "R": 3
}
WildCardAA = "GLY"
DefaultCord = 0
DefaultAngle = 0
SmartsOfRotableBond = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'  # not cycle and not H
SymetricTorsionAngleRmsdThreshold = 0.05
AABackbondPattern = "[N!H0]-[C:1]-[C:2](=[O:3])-[OH]"
CooHPattern = "[C:2](=[O:3])-[OH]"
NHPattern = "[N!H0]"
BackbonePattern = [AABackbondPattern]


DNARNAList = ['A', 'C', 'G', 'T', 'U', 'DA', 'DT',
              'DC', 'DG', "TPN", "APN", "CPN", "GPN"]

WaterList = ["HOH"]

IgnoreResidues = set(["UNL", "UNX", "BVA"])
""" "ASX", "GLX",
                  "EMC",  "BAL", "PRQ", "ACA", "BIL", "HMR",
                  "XCP", "XPC", "AEA", "B3T", "CF0", "B3D", "3FB", "B3E",
                  "PSA", "B3L", "BSE", "KFB"]"""  # ignore these residues as they are not clearly defined

Ions = ["CA", "Ca", "MG", "Mg", "Zn",
        "ZN", "Na", "NA", "K", "FE2",
        "SM", "NI", "CD", "FE", "CL",
        "MN", "RB", "CU", "AG", "CO", "LI",
        "CU1", "CU2", "H", "YB", "CS",
        "I", "IOD", "HG", "BA", "SR",
        "GD", "GA",  "PR", "PB", "YT3",
        "TB", "O", "RE", "PT", "BR", "U1",
        "LA", "LU", "YB2", "ARS"]
