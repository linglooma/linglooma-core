import json
import time
import logging
from typing import List, Literal, Dict, Any
import numpy as np
import librosa
import librosa.display
from pydantic import BaseModel
from services.transcribe import AudioProcessor, AudioTranscription
from utils.logging import log_execution_time
from utils.stress import WORD_DATABASE

logging.basicConfig(level=logging.INFO)


class StressError(BaseModel):
    word: str
    syllableBreakdown: List[str]
    errorType: Literal[
        "Stress Misplacement",
        "Vowel Reduction",
        "Consonant Substitution",
        "Insertion",
        "Omission",
        "None",
    ]
    actualStressedSyllableIndex: int
    expectedStressedSyllableIndex: int
    errorDescription: str
    improvementAdvice: str
    errorStartIndex: int
    errorEndIndex: int


class StressErrorList(BaseModel):
    errors: List[StressError]


def extract_pitch(audio_path):
    """ Extract pitch from the audio file. """
    start_time = time.time()

    y, sr = librosa.load(audio_path, sr=None)
    times = np.linspace(0, len(y) / sr, len(y))
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = np.array(
        [
            np.max(pitches[:, i]) if np.any(pitches[:, i]) else 0
            for i in range(pitches.shape[1])
        ]
    )
    pitch_times = librosa.frames_to_time(np.arange(len(pitch_values)), sr=sr)

    end_time = time.time()
    logging.info(f"⏳ Pitch extraction took {end_time - start_time:.4f} seconds")

    return pitch_times, pitch_values


def analyze_stress(transcription_data, pitch_times, pitch_values):
    """ Analyze word stress using pitch analysis. """
    start_time = time.time()
    words = transcription_data["word_timestamps"]["words"]
    results = []

    for word_entry in words:
        word = word_entry["word"].lower()
        start_time_word, end_time_word = word_entry["start"], word_entry["end"]

        word_mask = (pitch_times >= start_time_word) & (pitch_times <= end_time_word)
        word_pitches = pitch_values[word_mask]

        if len(word_pitches) == 0:
            results.append((word, None, []))  # ✅ Ensure it always returns 3 values
            continue

        word_info = WORD_DATABASE.get(
            word, {"Syllables": [], "Total syllables": 1, "Stress Indices": 0}
        )
        syllables = word_info["Syllables"]
        num_syllables = word_info["Total syllables"]
        stress_index = word_info["Stress Indices"]

        syllable_duration = (end_time_word - start_time_word) / num_syllables
        syllable_pitches = []

        for i in range(num_syllables):
            period_start = start_time_word + (i * syllable_duration)
            period_end = period_start + syllable_duration
            period_indices = np.where(
                (pitch_times >= period_start) & (pitch_times < period_end)
            )[0]

            if len(period_indices) > 0:
                period_pitches = word_pitches[
                    period_indices[0]: period_indices[-1] + 1
                ]
            else:
                logging.warning(f"No data for syllable {i + 1} of '{word}'")
                period_pitches = np.array([])

            syllable_pitches.append(
                np.max(period_pitches) if len(period_pitches) > 0 else 0
            )

        predicted_stress = int(np.argmax(syllable_pitches)) if num_syllables > 1 else 0
        results.append((word, predicted_stress, syllables))

    end_time = time.time()
    logging.info(f"⏳ Stress analysis took {end_time - start_time:.4f} seconds")
    logging.info(f"Stress analysis output: {results}")

    return results


class WordstressEvaluationService:
    @staticmethod
    @log_execution_time
    async def evaluate_stress(
        audio_transcription: AudioTranscription, audio_path: str
    ) -> List[StressError]:
        """ Evaluate word stress in speech. """
        start = time.time()
        pitch_times, pitch_values = extract_pitch(audio_path)
        transcription_data = audio_transcription.model_dump()
        stress_analysis = analyze_stress(transcription_data, pitch_times, pitch_values)
        errors = []

        for word, predicted_stress, syllables in stress_analysis:
            if syllables:
                expected_stress = WORD_DATABASE.get(word, {"Stress Indices": 0})[
                    "Stress Indices"
                ]
                if predicted_stress != expected_stress:
                    error = StressError(
                        word=word,
                        syllableBreakdown=syllables,
                        errorType="Stress Misplacement",
                        actualStressedSyllableIndex=predicted_stress,
                        expectedStressedSyllableIndex=expected_stress,
                        errorDescription="Chưa code em ơi, Gọi LLM lâu lắm",
                        improvementAdvice="Chưa code em ơi, Gọi LLM lâu lắm",
                        errorStartIndex=0,
                        errorEndIndex=len(word),
                    )
                    errors.append(error.model_dump())

        end = time.time()
        logging.info(f"Wordstress evaluation took {end - start:.4f} seconds")
        return errors


async def main():
    audio_path = "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/recorded_audio.mp3"
    start_time = time.time()

    actual_text = await AudioProcessor.transcribe(audio_path)
    end_time = time.time()
    logging.info(f"Transcription took {end_time - start_time:.4f} seconds")

    res = await WordstressEvaluationService.evaluate_stress(actual_text, audio_path)
    logging.info(f"Final result: {res}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
