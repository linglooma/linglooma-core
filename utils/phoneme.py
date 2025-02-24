import re
import json


def find_all_word_indices(text, word_list):
    word_indices = {word: [] for word in word_list}

    for word in set(word_list):
        matches = list(re.finditer(r"\b" + re.escape(word) + r"\b", text))
        for match in matches:
            start, end = match.start(), match.end() - 1
            word_indices[word].append((start, end))

    return word_indices


def update_transcription_error_indices(transcription, errors):
    words_to_find = [entry["transcribedWord"] for entry in errors]
    word_positions = find_all_word_indices(transcription, words_to_find)

    word_occurrences = {
        word: iter(word_positions.get(word, [])) for word in words_to_find
    }

    for entry in errors:
        word = entry["transcribedWord"]
        if word in word_occurrences:
            indices = next(word_occurrences[word], (-1, -1))
            (
                entry["errorStartIndexTranscription"],
                entry["errorEndIndexTranscription"],
            ) = indices

    return errors


transcription_text = "I rarely like reading English. I rarely find it youthful."
error_entries = [
    {
        "transcribedWord": "rarely",
        "expectedWord": "really",
        "expectedPronunciation": "ˈrɪli",
        "actualPronunciation": "ˈrɛrli",
        "errorType": "substitution",
        "errorStartIndexWord": 2,
        "errorEndIndexWord": 3,
        "substituted": "ɛr",
        "errorDescription": "/ˈrɛrli/ was pronounced instead of /ˈrɪli/",
        "improvementAdvice": "Replace /ɛr/ with /ɪ/ in 'really'",
        "errorStartIndexTranscription": -1,
        "errorEndIndexTranscription": -1,
    },
    {
        "transcribedWord": "rarely",
        "expectedWord": "really",
        "expectedPronunciation": "ˈrɪli",
        "actualPronunciation": "ˈrɛrli",
        "errorType": "substitution",
        "errorStartIndexWord": 2,
        "errorEndIndexWord": 3,
        "substituted": "ɛr",
        "errorDescription": "/ˈrɛrli/ was pronounced instead of /ˈrɪli/",
        "improvementAdvice": "Replace /ɛr/ with /ɪ/ in 'really'",
        "errorStartIndexTranscription": -1,
        "errorEndIndexTranscription": -1,
    },
    {
        "transcribedWord": "youthful",
        "expectedWord": "useful",
        "expectedPronunciation": "ˈjusfəl",
        "actualPronunciation": "ˈjuθfəl",
        "errorType": "substitution",
        "errorStartIndexWord": 2,
        "errorEndIndexWord": 3,
        "substituted": "juθ",
        "errorDescription": "/ˈjuθfəl/ was pronounced instead of /ˈjusfəl/",
        "improvementAdvice": "Replace /θ/ with /s/ in 'useful'",
        "errorStartIndexTranscription": -1,
        "errorEndIndexTranscription": -1,
    },
]

updated_errors = update_transcription_error_indices(transcription_text, error_entries)

print(json.dumps(updated_errors, indent=2, ensure_ascii=True))
