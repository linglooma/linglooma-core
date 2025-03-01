from nltk.corpus import cmudict
import pyphen

cmu_dict = cmudict.dict()
dic = pyphen.Pyphen(lang="en_US")

VALID_ONSETS = {
    "b",
    "bl",
    "br",
    "c",
    "ch",
    "cl",
    "cr",
    "d",
    "dr",
    "f",
    "fl",
    "fr",
    "g",
    "gl",
    "gr",
    "h",
    "j",
    "k",
    "kl",
    "kr",
    "l",
    "m",
    "n",
    "p",
    "pl",
    "pr",
    "qu",
    "r",
    "s",
    "sc",
    "sh",
    "sk",
    "sl",
    "sm",
    "sn",
    "sp",
    "st",
    "str",
    "sw",
    "t",
    "th",
    "tr",
    "v",
    "w",
    "x",
    "y",
    "z",
}


def advanced_syllabify(word: str, expected_count: int):
    vowels = "aeiouy"
    word_lower = word.lower()
    original_vowel_positions = [i for i, ch in enumerate(word_lower) if ch in vowels]
    if not original_vowel_positions:
        return [word]
    threshold = 2
    effective_vowel_positions = []
    for pos in original_vowel_positions:
        if (
            not effective_vowel_positions
            or pos - effective_vowel_positions[-1] > threshold
        ):
            effective_vowel_positions.append(pos)
    if len(effective_vowel_positions) != expected_count:
        n = expected_count
        length = len(word)
        indices = [round(i * length / n) for i in range(n)] + [length]
        return [word[indices[i] : indices[i + 1]] for i in range(n)]

    boundaries = []
    for i in range(len(effective_vowel_positions) - 1):
        v_i = effective_vowel_positions[i]
        v_j = effective_vowel_positions[i + 1]
        cluster = word_lower[v_i + 1 : v_j]
        valid_split = 0
        for j in range(len(cluster) + 1):
            onset_candidate = cluster[j:]
            if onset_candidate == "" or onset_candidate in VALID_ONSETS:
                valid_split = j
                break
        boundary = v_i + 1 + valid_split
        boundaries.append(boundary)

    syllables = []
    start = 0
    for b in boundaries:
        syllables.append(word[start:b])
        start = b
    syllables.append(word[start:])
    return syllables


def get_stress_index(word: str):
    word_lower = word.lower()
    if word_lower not in cmu_dict:
        return None

    syllables = dic.inserted(word).split("-")
    first_pron = cmu_dict[word_lower][0]
    expected_count = sum(1 for ph in first_pron if ph[-1].isdigit())

    if len(syllables) != expected_count:
        syllables = advanced_syllabify(word, expected_count)

    all_stress = []
    for pron in cmu_dict[word_lower]:
        vowels_in_pron = [ph for ph in pron if ph[-1].isdigit()]
        stresses = [i for i, ph in enumerate(vowels_in_pron) if int(ph[-1]) > 0]
        if len(vowels_in_pron) == len(syllables):
            all_stress.append(stresses)

    return (
        {
            "Word": word,
            "Syllables": syllables,
            "Total syllables": len(syllables),
            "Stress Indices": max(max(indices, default=0) for indices in all_stress),
        }
        if all_stress
        else None
    )


WORD_DATABASE = {}
for word in cmu_dict.keys():
    details = get_stress_index(word)
    if details:
        WORD_DATABASE[word] = details

print(WORD_DATABASE.get("rarely"))
