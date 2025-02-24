from nltk.corpus import cmudict
import pyphen
import re

cmu_dict = cmudict.dict()
dic = pyphen.Pyphen(lang="en")


def get_stress_index(word: str):
    if word.lower() not in cmu_dict:
        return f"'{word}' không có trong từ điển CMU."

    phonemes = cmu_dict[word.lower()]
    stresses = [
        [int(char[-1]) for char in pron if char[-1].isdigit()] for pron in phonemes
    ]

    syllables = dic.inserted(word).split("-")

    stress_indices = []
    for stress_pattern in stresses:
        syllable_index = 0
        indices = []
        for stress in stress_pattern:
            if stress > 0:
                indices.append(syllable_index)
            syllable_index += 1
        stress_indices.append(indices)

    return {
        "Word": word,
        "Syllables": syllables,
        "Total syllables": len(syllables),
        "Stress Indices": stress_indices,
    }


# word = "intelligence"
# result = get_stress_index(word)
# print(result)
