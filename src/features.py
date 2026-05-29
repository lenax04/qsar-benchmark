"""
features.py — Molecular descriptor calculation for QSAR modeling.

Supported descriptor sets:
    - 'morgan'    : Morgan circular fingerprints (ECFP4, radius=2, 2048 bits)
    - 'rdkit_fp'  : RDKit topological fingerprints (2048 bits)
    - 'maccs'     : MACCS keys (167 bits)
    - 'physchem'  : 12 physicochemical descriptors (MW, LogP, HBD, HBA, TPSA, ...)
    - 'combined'  : Morgan + physicochemical (default for best performance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

DescriptorType = Literal["morgan", "rdkit_fp", "maccs", "physchem", "combined"]


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit Mol object, return None if invalid."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_morgan_fp(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan circular fingerprint (ECFP4 equivalent)."""
    from rdkit.Chem import AllChem
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def compute_rdkit_fp(mol, n_bits: int = 2048) -> np.ndarray:
    """Compute RDKit topological fingerprint."""
    from rdkit.Chem import RDKFingerprint
    fp = RDKFingerprint(mol, fpSize=n_bits)
    return np.array(fp)


def compute_maccs_keys(mol) -> np.ndarray:
    """Compute MACCS keys (167 bits)."""
    from rdkit.Chem import MACCSkeys
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def compute_physchem(mol) -> np.ndarray:
    """
    Compute 12 physicochemical descriptors:
    MW, LogP, HBD, HBA, TPSA, RotBonds, AromaticRings,
    RingCount, FractionCSP3, MolarRefractivity, QED, NumHeavyAtoms
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.QED import qed

    try:
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.MolMR(mol),
            qed(mol),
            mol.GetNumHeavyAtoms(),
        ]
        return np.array(desc, dtype=float)
    except Exception:
        return np.full(12, np.nan)


PHYSCHEM_NAMES = [
    "MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds",
    "AromaticRings", "RingCount", "FractionCSP3",
    "MolarRefractivity", "QED", "NumHeavyAtoms"
]


def compute_descriptors(
    smiles_list: list[str],
    descriptor_type: DescriptorType = "combined",
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
    rdkit_bits: int = 2048,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute molecular descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        Input SMILES strings.
    descriptor_type : str
        Type of descriptors to compute. One of:
        'morgan', 'rdkit_fp', 'maccs', 'physchem', 'combined'.
    morgan_radius : int
        Radius for Morgan fingerprints (default 2 = ECFP4).
    morgan_bits : int
        Number of bits for Morgan fingerprints (default 2048).
    rdkit_bits : int
        Number of bits for RDKit fingerprints (default 2048).
    verbose : bool
        Print progress every 500 molecules.

    Returns
    -------
    X : np.ndarray of shape (n_valid, n_features)
    valid_indices : list of int
        Indices of valid SMILES (molecules that could be parsed).
    feature_names : list of str
    """
    X_list = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        if verbose and i % 500 == 0:
            print(f"  Processing molecule {i}/{len(smiles_list)}...")
        mol = smiles_to_mol(smi)
        if mol is None:
            continue

        if descriptor_type == "morgan":
            vec = compute_morgan_fp(mol, morgan_radius, morgan_bits)
        elif descriptor_type == "rdkit_fp":
            vec = compute_rdkit_fp(mol, rdkit_bits)
        elif descriptor_type == "maccs":
            vec = compute_maccs_keys(mol)
        elif descriptor_type == "physchem":
            vec = compute_physchem(mol)
        elif descriptor_type == "combined":
            morgan = compute_morgan_fp(mol, morgan_radius, morgan_bits)
            phys   = compute_physchem(mol)
            vec    = np.concatenate([morgan, phys])
        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")

        if not np.any(np.isnan(vec)):
            X_list.append(vec)
            valid_indices.append(i)

    X = np.array(X_list, dtype=np.float32)

    # Feature names
    if descriptor_type == "morgan":
        feature_names = [f"Morgan_{i}" for i in range(morgan_bits)]
    elif descriptor_type == "rdkit_fp":
        feature_names = [f"RDKit_{i}" for i in range(rdkit_bits)]
    elif descriptor_type == "maccs":
        feature_names = [f"MACCS_{i}" for i in range(167)]
    elif descriptor_type == "physchem":
        feature_names = PHYSCHEM_NAMES
    elif descriptor_type == "combined":
        feature_names = (
            [f"Morgan_{i}" for i in range(morgan_bits)] + PHYSCHEM_NAMES
        )
    else:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    if verbose:
        print(f"  Valid molecules: {len(valid_indices)}/{len(smiles_list)}")
        print(f"  Feature matrix shape: {X.shape}")

    return X, valid_indices, feature_names


def compute_all_descriptor_sets(
    smiles_list: list[str],
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, list[int], list[str]]]:
    """
    Compute all descriptor sets for benchmark comparison.

    Returns a dict mapping descriptor name -> (X, valid_indices, feature_names).
    """
    results = {}
    for dtype in ["morgan", "rdkit_fp", "maccs", "physchem", "combined"]:
        if verbose:
            print(f"\nComputing {dtype} descriptors...")
        X, valid_idx, feat_names = compute_descriptors(
            smiles_list, descriptor_type=dtype, verbose=verbose
        )
        results[dtype] = (X, valid_idx, feat_names)
    return results
