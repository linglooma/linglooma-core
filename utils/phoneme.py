import re


def find_all_word_indices(text, word_list):
    word_indices = {word: [] for word in word_list}

    for word in set(word_list):
        matches = list(re.finditer(r"\b" + re.escape(word) + r"\b", text))
        for match in matches:
            start, end = match.start(), match.end() - 1
            word_indices[word].append((start, end))

    return word_indices


def update_transcription_error_indices(transcription, errors):
    words_to_find = [entry.transcribedWord for entry in errors]
    word_positions = find_all_word_indices(transcription, words_to_find)

    word_occurrences = {
        word: iter(word_positions.get(word, [])) for word in words_to_find
    }

    for entry in errors:
        word = entry.transcribedWord
        if word in word_occurrences:
            indices = next(word_occurrences[word], (-1, -1))
            (
                entry.errorStartIndexTranscription,
                entry.errorEndIndexTranscription,
            ) = indices

    return errors
