from typing import Callable
import re

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Mol, rdChemReactions

TOKENIZER_PATTERN = "(\%\([0-9]{3}\)|\[[^\]]+]|Se?|Si?|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
TOKENIZER_REGEX = re.compile(TOKENIZER_PATTERN)


def pad_to_len(token_array: list[int], max_length: int) -> list[int]:
    """
    Pad a token array to a maximum length.

    Args:
        token_array: Token array to pad.
        max_length: Maximum length of the token array.

    Returns:
        Padded token array.
    """
    return token_array + [0] * (max_length - len(token_array))


def smiles_to_alphabet(smiles: str) -> list[str]:
    """
    Get the alphabet of a SMILES string.

    Args:
        smiles: SMILES string to get the alphabet of.

    Returns:
        Alphabet of the SMILES string.
    """
    return [a for a in TOKENIZER_REGEX.findall(smiles)]


def tokenize_smiles(smiles_series: pd.Series, all_smiles: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """
    Tokenize a SMILES string.

    Args:
        smiles_series: series of SMILES strings to tokenize.

    Returns:
        List of tokens.
    """
    # Get SMILES alphabet that's used in the dataset
    all_smiles_chars: list[set[str]] = [set(smiles_to_alphabet(smiles)) for smiles in all_smiles]
    alphabet: list[str] = sorted(set().union(*all_smiles_chars))
    alphabet.insert(0, "PAD")

    # Get token dictionary
    token_dict: dict[str, int] = {token: i for i, token in enumerate(alphabet)}

    # Padding length
    max_length: int = max(len(abc) for abc in all_smiles_chars)

    # Tokenize each SMILES string
    tokenized_smiles: pd.Series = smiles_series.apply(
        lambda s: pad_to_len([token_dict[char] for char in smiles_to_alphabet(s)], max_length))

    return tokenized_smiles, token_dict


def tokenize_selfies(selfies_series: pd.Series, all_selfies: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """
    Tokenize SELFIES strings in a Series.

    Args:
        selfies_series: Series of SELFIES strings to tokenize.
        all_selfies: List of all SELFIES strings.

    Returns:
        Series of token lists.
    """
    # Get SELFIES alphabet that's used in the dataset
    alphabet: list[str] = sorted(sf.get_alphabet_from_selfies(all_selfies))
    alphabet.insert(0, "[nop]")

    # Create token dictionary
    selfie_tokens: dict[str, int] = {s: i for i, s in enumerate(alphabet)}

    # Padding length
    max_selfie_length = max(sf.len_selfies(s) for s in selfies_series)

    # Tokenize each SELFIES string
    tokenized_selfies = selfies_series.apply(
        lambda s: sf.selfies_to_encoding(s, selfie_tokens, pad_to_len=max_selfie_length, enc_type="label"))

    return tokenized_selfies, selfie_tokens


def tokenize_brics(brics_series: pd.Series, all_brics: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """
    Tokenize a BRICS string.

    Args:
        brics_series: series of BRICS fragments to tokenize.

    Returns:
        List of tokens.
    """
    # Get SMILES alphabet that's used in the dataset
    all_brics_frags: list[set[str]] = [set(brics) for brics in all_brics]
    unique_brics_frags: list[str] = sorted(set().union(*all_brics_frags))
    unique_brics_frags.insert(0, "PAD")

    # Get token dictionary
    token_dict: dict[str, int] = {token: i for i, token in enumerate(unique_brics_frags)}

    # Padding length
    max_length: int = max(len(frags) for frags in all_brics_frags)

    # Tokenize each SMILES string
    tokenized_brics: pd.Series = brics_series.apply(
        lambda b: pad_to_len([token_dict[frag] for frag in b], max_length))

    return tokenized_brics, token_dict


tokenizer_factory: dict[str, Callable] = {"SMILES":  tokenize_smiles,
                                          "SELFIES": tokenize_selfies,
                                          "BRICS":   tokenize_brics}


def sanitize_dummies(smiles: str) -> str:
    """
    Sanitize a SMILES string by removing dummy atoms.

    Args:
        smiles: SMILES string to sanitize.

    Returns:
        Sanitized SMILES string.
    """
    pattern: str = r'\[\d+\*]'
    sanitized_dummy_smiles: str = re.sub(pattern, '*', smiles)
    return sanitized_dummy_smiles


def generate_brics(mol: Mol) -> list[str]:
    """
    Get the BRICS fragments of a RDKit Mol object.

    Args:
        mol: RDKit Mol object

    Returns:
        List of BRICS fragments as SMILES strings.
    """
    raw_frags: set[str] = BRICS.BRICSDecompose(mol)
    brics_frags: list[str] = [sanitize_dummies(rf) for rf in raw_frags]
    return brics_frags


def generate_fingerprint(mol: Mol, radius: int = 3, nbits: int = 1024) -> np.array:
    """
    Generate ECFP fingerprint.

    Args:
        mol: RDKit Mol object
        radius: Fingerprint radius
        nbits: Number of bits in fingerprint

    Returns:
        ECFP fingerprint as numpy array
    """
    fingerprint: np.array = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits))
    return fingerprint


def convert_to_bigsmiles(smiles: str) -> str:
    # BUG: Doesn't quite work. Sometimes only converts one cCH3 to c[Li] instead of both.
    """Convert SMILES string to BigSMILES."""
    dummy_atom: str = "Li"
    # Create an RDKit molecule from the SMILES representation
    mol = Chem.MolFromSmiles(smiles)
    new_mol = mol

    # Create a reaction template to match the cCH3 motif and replace with c[Si]
    reaction_smarts = f"[c:1][CH3:2]>>[c:1][{dummy_atom}:2]"
    reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    rxn_products = ((0,),)

    while len(rxn_products) != 0:
        # Apply the reaction to the molecule to convert cCH3 motifs to c[Si]
        rxn_products = reaction.RunReactants((new_mol,))
        if len(rxn_products) == 0:
            break
        new_mol = rxn_products[0][0]
        # return converted_mol

    # Generate the BigSMILES representation from the converted molecule
    bigsmiles = Chem.MolToSmiles(new_mol)
    replacement_count: int = bigsmiles.count(f"[{dummy_atom}]")
    assert replacement_count == 2, f"Found {replacement_count} {dummy_atom} atoms for {bigsmiles}."
    bigsmiles: str = "{" + bigsmiles.replace(dummy_atom, "$") + "}"
    return bigsmiles


def canonicalize_column(smiles_column: pd.Series) -> list[str]:
    """Canonicalize SMILES strings."""
    return smiles_column.apply(lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
