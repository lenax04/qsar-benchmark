"""Unit tests for the features module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    smiles_to_mol,
    compute_morgan_fp,
    compute_rdkit_fp,
    compute_maccs_keys,
    compute_physchem,
    compute_descriptors,
    PHYSCHEM_NAMES,
)

# Test SMILES: aspirin, erlotinib (EGFR inhibitor), invalid
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
INVALID = "not_a_smiles"


class TestSmilesToMol:
    def test_valid_smiles(self):
        mol = smiles_to_mol(ASPIRIN)
        assert mol is not None

    def test_invalid_smiles(self):
        mol = smiles_to_mol(INVALID)
        assert mol is None

    def test_erlotinib(self):
        mol = smiles_to_mol(ERLOTINIB)
        assert mol is not None


class TestMorganFingerprint:
    def test_shape(self):
        mol = smiles_to_mol(ASPIRIN)
        fp = compute_morgan_fp(mol, radius=2, n_bits=2048)
        assert fp.shape == (2048,)

    def test_binary(self):
        mol = smiles_to_mol(ASPIRIN)
        fp = compute_morgan_fp(mol)
        assert set(fp).issubset({0, 1})

    def test_different_molecules(self):
        mol1 = smiles_to_mol(ASPIRIN)
        mol2 = smiles_to_mol(ERLOTINIB)
        fp1 = compute_morgan_fp(mol1)
        fp2 = compute_morgan_fp(mol2)
        assert not np.array_equal(fp1, fp2)


class TestRDKitFingerprint:
    def test_shape(self):
        mol = smiles_to_mol(ASPIRIN)
        fp = compute_rdkit_fp(mol, n_bits=2048)
        assert fp.shape == (2048,)

    def test_binary(self):
        mol = smiles_to_mol(ASPIRIN)
        fp = compute_rdkit_fp(mol)
        assert set(fp).issubset({0, 1})


class TestMACCSKeys:
    def test_shape(self):
        mol = smiles_to_mol(ASPIRIN)
        fp = compute_maccs_keys(mol)
        assert fp.shape == (167,)

    def test_binary(self):
        mol = smiles_to_mol(ASPIRIN)
        fp = compute_maccs_keys(mol)
        assert set(fp).issubset({0, 1})


class TestPhyschemDescriptors:
    def test_shape(self):
        mol = smiles_to_mol(ASPIRIN)
        desc = compute_physchem(mol)
        assert desc.shape == (12,)
        assert len(PHYSCHEM_NAMES) == 12

    def test_aspirin_mw(self):
        mol = smiles_to_mol(ASPIRIN)
        desc = compute_physchem(mol)
        mw = desc[0]  # MW is first
        assert 175 < mw < 185  # Aspirin MW ~180.16

    def test_no_nan(self):
        mol = smiles_to_mol(ERLOTINIB)
        desc = compute_physchem(mol)
        assert not np.any(np.isnan(desc))


class TestComputeDescriptors:
    def test_morgan_batch(self):
        smiles = [ASPIRIN, ERLOTINIB]
        X, valid_idx, feat_names = compute_descriptors(
            smiles, descriptor_type="morgan", verbose=False
        )
        assert X.shape == (2, 2048)
        assert len(valid_idx) == 2
        assert len(feat_names) == 2048

    def test_invalid_filtered(self):
        smiles = [ASPIRIN, INVALID, ERLOTINIB]
        X, valid_idx, feat_names = compute_descriptors(
            smiles, descriptor_type="morgan", verbose=False
        )
        assert X.shape[0] == 2  # INVALID filtered out
        assert valid_idx == [0, 2]

    def test_combined_shape(self):
        smiles = [ASPIRIN, ERLOTINIB]
        X, valid_idx, feat_names = compute_descriptors(
            smiles, descriptor_type="combined", verbose=False
        )
        assert X.shape == (2, 2048 + 12)
        assert len(feat_names) == 2048 + 12

    def test_physchem_shape(self):
        smiles = [ASPIRIN, ERLOTINIB]
        X, valid_idx, feat_names = compute_descriptors(
            smiles, descriptor_type="physchem", verbose=False
        )
        assert X.shape == (2, 12)
