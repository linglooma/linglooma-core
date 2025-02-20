import asyncio
import librosa
import numpy as np
import json
import re
from typing import Dict, Tuple
from dataclasses import dataclass
from pydantic import Json
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from config.client import openai_client


@dataclass
class IntonationAnalysis:
    expected_type: str
    confidence: float
    rule_matched: str
    error_start: int = 0
    error_end: int = 0
    pitch_statistics: Dict[str, float] = None
    phonetic_features: Dict[str, float] = None
    acoustic_features: Dict[str, float] = None


@dataclass
class EnhancedIntonationRules:
    QUESTION_PATTERNS = [
        r"^(what|where|when|who|why|how|which|whose|whom)(?:\s+\w+){0,5}\b(?:\s+(?:do|does|did|is|are|was|were|have|has|had))?\b",
        r"^(?:(?:do|does|did|is|are|am|was|were|have|has|had|can|could|will|would|shall|should|may|might|must)|(?:isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|don't|doesn't|didn't))\b.*\??",
        r".*,\s*(?:isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|don't|doesn't|didn't)\s+(?:he|she|it|you|they|we|I)\b\??",
        r"\b(?:could|would|can|will)\s+you\s+(?:tell|explain|show|let\s+me\s+know)\s+(?:what|where|when|who|why|how)\b",
        r"^(?:would|do|does|did)\s+you\s+(?:prefer|want|like|need)(?:\s+to\s+\w+)?\s+(?:or)\s+",
    ]

    STATEMENT_PATTERNS = [
        r"^[A-Z][^.!?]*(?:because|although|though|since|when|if|unless|while|whereas)[^.!?]*(?:that|which|who)[^.!?]*\.",
        r"^[A-Z][^.!?]*(?:and|but|or|yet|so)[^.!?]*\.",
        r"\b(?:said|mentioned|explained|stated|suggested|believed|thought)\s+that\b.*\.",
        r"\b(?:is|are|am|was|were|have been|has been|had been)\s+(?:\w+ing|\w+ed)\b.*\.",
        r"^[A-Z][^.!?]*\b(?:the|a|an)\s+\w+\s+(?:is|are|was|were)\b.*\.",
    ]

    LIST_PATTERNS = [
        r".*(?::\s*(?:1\.|a\.|•|\*)\s*[^,;]+(?:;\s*(?:2\.|b\.|•|\*)\s*[^,;]+)+)",
        r"\b(?:both|either|neither)\b.*\b(?:and|or|nor)\b.*",
        r"\b(?:first(?:ly)?|initial(?:ly)?)[^,]*,\s*(?:second(?:ly)?|next|then)[^,]*,\s*(?:final(?:ly)?|lastly|ultimately)",
        r".*:\s*(?:[^,]+(?:\s+\([^)]+\))?(?:,\s*|$))+",
        r"\b(?:on\s+(?:the|one)\s+hand|in\s+contrast|similarly|likewise)\b.*\b(?:on\s+the\s+other\s+hand|however|whereas|while)\b",
    ]

    @staticmethod
    def preprocess_pitch(pitch_array: np.ndarray) -> np.ndarray:
        """Tiền xử lý dữ liệu pitch"""
        if len(pitch_array) < 3:
            return pitch_array

        pitch_cleaned = pitch_array[~np.isnan(pitch_array)]
        if len(pitch_cleaned) == 0:
            return np.zeros(1)

        scaler = StandardScaler()
        pitch_normalized = scaler.fit_transform(pitch_cleaned.reshape(-1, 1)).flatten()

        if len(pitch_normalized) > 5:
            return savgol_filter(pitch_normalized, 5, 2)
        return pitch_normalized

    @staticmethod
    def extract_acoustic_features(pitch_array: np.ndarray) -> Dict[str, float]:
        """Trích xuất đặc trưng âm học"""
        pitch_processed = EnhancedIntonationRules.preprocess_pitch(pitch_array)

        if len(pitch_processed) < 3:
            return {
                "mean_pitch": 0.0,
                "pitch_range": 0.0,
                "pitch_slope": 0.0,
                "pitch_variance": 0.0,
                "contour_complexity": 0.0,
            }

        mean_pitch = float(np.mean(pitch_processed))
        pitch_range = float(np.ptp(pitch_processed))
        pitch_slope = float(
            np.polyfit(np.arange(len(pitch_processed)), pitch_processed, 1)[0]
        )
        pitch_variance = float(np.var(pitch_processed))

        diff_sequence = np.diff(pitch_processed)
        zero_crossings = np.where(np.diff(np.signbit(diff_sequence)))[0]
        contour_complexity = float(len(zero_crossings)) / len(pitch_processed)

        return {
            "mean_pitch": mean_pitch,
            "pitch_range": pitch_range,
            "pitch_slope": pitch_slope,
            "pitch_variance": pitch_variance,
            "contour_complexity": contour_complexity,
        }

    @staticmethod
    def analyze_pitch_pattern(
        pitch_array: np.ndarray, threshold: float = 0.15
    ) -> Tuple[str, float]:
        features = EnhancedIntonationRules.extract_acoustic_features(pitch_array)

        confidence = min(
            1.0,
            abs(features["pitch_slope"]) * 2
            + features["pitch_variance"]
            + features["contour_complexity"],
        )

        if features["pitch_slope"] > threshold and features["pitch_range"] > threshold:
            return "Rising", confidence
        elif (
            features["pitch_slope"] < -threshold and features["pitch_range"] > threshold
        ):
            return "Falling", confidence
        elif (
            abs(features["pitch_slope"]) < threshold / 2
            and features["pitch_variance"] < threshold
        ):
            return "Flat", confidence
        elif features["contour_complexity"] > 0.5:
            if features["pitch_slope"] > 0:
                return "Rising-Falling", confidence
            else:
                return "Falling-Rising", confidence

        return "Unclear", confidence * 0.5


class InnotationEvaluationService:
    rules = EnhancedIntonationRules()

    @staticmethod
    def analyze_intonation(text: str, pitch_data: np.ndarray) -> IntonationAnalysis:
        acoustic_type, acoustic_confidence = (
            InnotationEvaluationService.rules.analyze_pitch_pattern(pitch_data)
        )
        acoustic_features = InnotationEvaluationService.rules.extract_acoustic_features(
            pitch_data
        )

        scores = {
            "Question": {"score": 0.0, "weight": 1.0},
            "Statement": {"score": 0.0, "weight": 1.0},
            "List": {"score": 0.0, "weight": 1.0},
        }

        if acoustic_confidence > 0.7:
            if acoustic_type == "Rising":
                scores["Question"]["weight"] = 1.5
            elif acoustic_type == "Falling":
                scores["Statement"]["weight"] = 1.5
            elif acoustic_type in ["Rising-Falling", "Falling-Rising"]:
                scores["List"]["weight"] = 1.5

        for pattern in InnotationEvaluationService.rules.QUESTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                scores["Question"]["score"] += 0.3 * scores["Question"]["weight"]

        for pattern in InnotationEvaluationService.rules.STATEMENT_PATTERNS:
            if re.search(pattern, text):
                scores["Statement"]["score"] += 0.3 * scores["Statement"]["weight"]

        for pattern in InnotationEvaluationService.rules.LIST_PATTERNS:
            if re.search(pattern, text):
                scores["List"]["score"] += 0.3 * scores["List"]["weight"]

        final_scores = {k: v["score"] for k, v in scores.items()}
        max_score_type = max(final_scores.items(), key=lambda x: x[1])

        error_start, error_end = 0, len(text)
        if max_score_type[1] > 0.5:
            for pattern in getattr(
                InnotationEvaluationService.rules,
                f"{max_score_type[0].upper()}_PATTERNS",
            ):
                match = re.search(pattern, text)
                if match:
                    error_start, error_end = match.span()
                    break

        expected_type = {
            "Question": "Rising",
            "Statement": "Falling",
            "List": "Rising-Falling",
        }.get(max_score_type[0], "Unclear")

        return IntonationAnalysis(
            expected_type=expected_type,
            confidence=max_score_type[1],
            rule_matched=max_score_type[0],
            error_start=error_start,
            error_end=error_end,
            pitch_statistics=acoustic_features,
            phonetic_features={
                "pattern_type": acoustic_type,
                "pattern_confidence": acoustic_confidence,
            },
            acoustic_features=acoustic_features,
        )

    @staticmethod
    def get_gpt_analysis(
        text: str, actual_intonation: str, rule_analysis: IntonationAnalysis
    ) -> Dict:
        prompt = f"""
        You are an expert phonetics analyst specializing in speech intonation patterns. Your task is to analyze the provided speech segment and provide detailed feedback.
            INPUT CONTEXT:
            Text: "{text}"
            Detected Intonation Pattern: "{actual_intonation}"
            Rule-based Analysis Results:
            - Expected Intonation Type: {rule_analysis.expected_type}
            - Analysis Confidence Score: {rule_analysis.confidence:.2f}
            - Matching Rule Pattern: {rule_analysis.rule_matched}
            - Detected Acoustic Features: {rule_analysis.acoustic_features}

            ANALYSIS REQUIREMENTS:
            1. Compare the detected intonation against expected patterns
            2. Identify specific points where intonation deviates from expected patterns
            3. Provide actionable feedback for improvement
            4. Include precise position markers for error locations
            5. Consider linguistic context and semantic meaning
            6. The errorDescription should be concise and informative and MAXIMUM 2 SENTENCES
            7. The improvementAdvice should be specific and actionable and ONLY ONE SENTENCE

            RESPONSE FORMAT:
            Return a JSON object with the following structure (no additional text or markdown or ```json):
            {{
                "expectedIntonationType": string,     // The correct intonation pattern that should be used
                "errorDescription": string,           // Detailed analysis of what went wrong. MAX 2 SENTENCES.
                "improvementAdvice": string,          // Specific, actionable advice for improvement. ONLY ONE SENTENCE.
                "errorStartIndex": number,            // Starting character position of the error
                "errorEndIndex": number              // Ending character position of the error
            }}

            INTONATION TYPE GUIDELINES:
            - Rising: Used for questions, uncertainty, or continuation
            - Falling: Used for statements, commands, or completion
            - Rising-Falling: Used for emphasis or contrast
            - Falling-Rising: Used for uncertainty or reservation
            - Level: Used for listing or incomplete thoughts
            - Complex: Multiple intonation patterns within the same utterance

            ERROR DESCRIPTION GUIDELINES:
            - Be specific about the nature of the error
            - Reference the acoustic features detected
            - Explain why the current intonation is inappropriate
            - Consider the semantic context

            IMPROVEMENT ADVICE GUIDELINES:
            - Provide practical exercises or techniques
            - Give examples of correct intonation patterns
            - Suggest specific words or syllables to focus on
            - Include rhythm and stress considerations

            Example Outputs:
            {{
                "expectedIntonationType": "Rising",
                "errorDescription": "The sentence is a yes/no question but uses a falling intonation, making it sound like a statement.",
                "improvementAdvice": "Practice raising your pitch by 20-30Hz on the final syllable, try saying 'up' in your mind as you reach the end of the question.",
                "errorStartIndex": 15,
                "errorEndIndex": 25
            }}

            {{
                "expectedIntonationType": "Complex",
                "errorDescription": "The conditional statement requires a rising pattern on the if-clause followed by a falling pattern on the main clause.",
                "improvementAdvice": "Break the sentence into two parts, rise on 'if-clause' to show continuation, then fall on the main clause to show completion.",
                "errorStartIndex": 0,
                "errorEndIndex": 45
            }}
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            gpt_analysis = response.choices[0].message.content.strip()
            return json.loads(gpt_analysis)
        except Exception:
            return {
                "expectedIntonationType": rule_analysis.expected_type,
                "errorDescription": "An error occurred during analysis",
                "improvementAdvice": "Try again with a different input",
                "errorStartIndex": rule_analysis.error_start,
                "errorEndIndex": rule_analysis.error_end,
            }

    @staticmethod
    async def process_audio(actual_text: str, audio_path: str) -> Json:
        try:
            y, sr = librosa.load(audio_path, sr=None)
            f0, _, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )
            f0_cleaned = f0[~np.isnan(f0)] if np.any(~np.isnan(f0)) else np.zeros(1)
            if not actual_text:
                raise ValueError("Failed to transcribe audio")
            rule_analysis = InnotationEvaluationService.analyze_intonation(
                actual_text, f0_cleaned
            )
            actual_intonation, _ = (
                InnotationEvaluationService.rules.analyze_pitch_pattern(f0_cleaned)
            )
            gpt_analysis = InnotationEvaluationService.get_gpt_analysis(
                actual_text, actual_intonation, rule_analysis
            )
            return {
                "clauseText": actual_text,
                "actualIntonationType": actual_intonation,
                "expectedIntonationType": gpt_analysis.get(
                    "expectedIntonationType", "Unknown"
                ),
                "errorDescription": gpt_analysis.get("errorDescription", ""),
                "improvementAdvice": gpt_analysis.get("improvementAdvice", ""),
                "errorStartIndex": gpt_analysis.get("errorStartIndex", 0),
                "errorEndIndex": gpt_analysis.get("errorEndIndex", 0),
            }
        except Exception as e:
            return {"error": str(e), "errorType": type(e).__name__}


async def main():
    audio_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )
    actual_text = "I rarely like learning English. I find it youthful"
    result = await InnotationEvaluationService.process_audio(actual_text, audio_path)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
