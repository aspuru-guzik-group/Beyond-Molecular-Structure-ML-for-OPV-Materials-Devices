import copy
import re
from collections import Counter

import numpy as np
import selfies as sf

TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
TOKENIZER_REGEX = re.compile(TOKENIZER_PATTERN)


class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, input):
        tokens = [t for t in TOKENIZER_REGEX.findall(input)]
        return " ".join(tokens)

    def pad_input(self, tokenized_array, seq_len):
        """Function that pads the reactions with 0 (_PAD) to a fixed length
        PRE-PADDING (features[ii, -len(review) :] = np.array(review)[:seq_len])
        POST-PADDING (features[ii, : len(review)] = np.array(review)[:seq_len])
        """
        features = np.zeros((len(tokenized_array), seq_len), dtype=int)
        for ii, review in enumerate(tokenized_array):
            if len(review) != 0:
                features[ii, -len(review) :] = np.array(review)[:seq_len]
        return features

    def tokenize_data(self, input_series):
        """Function that arranges series (pandas) of SMILES/BigSMILES/SELFIES and tokenizes each SMILES/BigSMILES/SELFIES.

        Parameter
        ----------
        input_series: a series of SMILES/BigSMILES/SELFIES

        Return
        -------
        tokenized_array: array of arrays of tokenized SMILES/BigSMILES/SELFIES for each SMILES/BigSMILES/SELFIES input
        """

        # tokenizing SMILES/BigSMILES/SELFIES
        input_dict = (
            Counter()
        )  # Dictionary that will map an atom to the number of times it appeared in all the training set
        tokenized_array = copy.copy(input_series)
        for i, reaction in enumerate(input_series):
            # The SMILES/BigSMILES/SELFIES will be stored as a list of tokens
            tokenized_array[i] = []
            for token in self.tokenize(reaction):  # tokenizing SMILES/BigSMILES/SELFIES
                input_dict.update([token])
                tokenized_array[i].append(token)
            if i % 50 == 0:
                print(str((i * 100) / len(tokenized_array)) + "% done")

        # organizing SMILES/BigSMILES/SELFIES dictionary
        # Sorting the atoms according to the number of appearances, with the most common atom being first
        input_dict = sorted(input_dict, key=input_dict.get, reverse=True)
        # Adding padding and unknown to our vocabulary so that they will be assigned an index
        input_dict = ["_PAD", "_UNK"] + input_dict
        print(input_dict)
        # Dictionaries to store the token to index mappings and vice versa
        token2idx = {j: i for i, j in enumerate(input_dict)}
        print("token2idx: ", token2idx)

        for i, reaction in enumerate(input_series):
            # Looking up the mapping dictionary and assigning the index to the respective reactions
            tokenized_array[i] = [
                token2idx[token] if token in token2idx else 0 for token in reaction
            ]

        max_length = 0
        for input_array in tokenized_array:
            if len(input_array) > max_length:
                max_length = len(input_array)
        print("Max sequence length: ", max_length)
        tokenized_array = self.pad_input(tokenized_array, max_length)
        vocab_length = len(input_dict)

        return tokenized_array, max_length, vocab_length

    def tokenize_selfies(self, da_pair):
        """
        Function that returns max selfie length and unique selfie dictionary
        """
        selfie_dict = {"[nop]": 0, ".": 1}
        max_selfie_length = 1
        for pair in da_pair:
            selfie_alphabet = sorted(list(sf.get_alphabet_from_selfies([pair])))
            len_selfie = sf.len_selfies(pair)
            if len_selfie > max_selfie_length:
                max_selfie_length = len_selfie
            for char in selfie_alphabet:
                if char not in selfie_dict.keys():
                    selfie_dict[char] = len(selfie_dict)

        return selfie_dict, max_selfie_length

    def tokenize_given_dictionary(self, input, input_dict):
        """
        Function that returns the tokenized "one-hot" encoded version of the input given a dictionary
        """
        tokenized_input = []
        for char in input:
            one_hot_idx = input_dict[char]
            tokenized_input.append(one_hot_idx)

        return tokenized_input


# for presentation (can be deleted)
tk = Tokenizer()
input_dict = {
    "[nop]": 0,
    ".": 1,
    "[#C]": 2,
    "[#N]": 3,
    "[/C]": 4,
    "[=C]": 5,
    "[=N]": 6,
    "[=O]": 7,
    "[Branch1_1]": 8,
    "[Branch1_2]": 9,
    "[Branch1_3]": 10,
    "[Branch2_1]": 11,
    "[Branch2_2]": 12,
    "[Branch2_3]": 13,
    "[C]": 14,
    "[Expl=Ring1]": 15,
    "[Expl=Ring2]": 16,
    "[N]": 17,
    "[O]": 18,
    "[P]": 19,
    "[Ring1]": 20,
    "[Ring2]": 21,
    "[S]": 22,
    "[\\C]": 23,
    "[F]": 24,
    "[Siexpl]": 25,
    "[/S]": 26,
    "[Cl]": 27,
    "[=S]": 28,
    "[\\S]": 29,
    "[B]": 30,
    "[I]": 31,
    "[Seexpl]": 32,
}
# tokenized_smile = tk.tokenize_given_dictionary(
#     "[C][C][C][C][C][C][C][Branch1_1][Branch1_1][C][C][C][C][C][S][O][C][Branch1_2][C][=O][C][C][=C][Branch2_1][Ring2][N][C][=C][C][Branch2_1][Ring1][Branch1_1][C][Branch1_2][C][=O][O][S][C][C][Branch1_1][Branch1_1][C][C][C][C][C][C][C][C][C][C][=C][Branch1_1][O][C][=C][C][=C][Branch1_1][C][C][S][Ring1][Branch1_2][S][Ring2][Ring1][O][S][C][Expl=Ring2][Ring1][S][C][=C][C][=C][Branch1_1][C][C][S][Ring1][Branch1_2]",
#     input_dict,
# )
tokenized_selfie = sf.selfies_to_encoding(
    "[C][C][C][C][C][C][C][Branch1_1][Branch1_1][C][C][C][C][C][S][O][C][Branch1_2][C][=O][C][C][=C][Branch2_1][Ring2][N][C][=C][C][Branch2_1][Ring1][Branch1_1][C][Branch1_2][C][=O][O][S][C][C][Branch1_1][Branch1_1][C][C][C][C][C][C][C][C][C][C][=C][Branch1_1][O][C][=C][C][=C][Branch1_1][C][C][S][Ring1][Branch1_2][S][Ring2][Ring1][O][S][C][Expl=Ring2][Ring1][S][C][=C][C][=C][Branch1_1][C][C][S][Ring1][Branch1_2]",
    input_dict,
    pad_to_len=-1,
    enc_type="label",
)
print(tokenized_selfie)
