import asyncio
import logging
import time
import instructor
import librosa
import numpy as np
import json
import time
import logging
from config.client import groq_client
import re
from typing import Dict, Literal, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from config.client import openai_client
from utils.logging import log_execution_time


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


class IntonationFeedback(BaseModel):
    expectedIntonationType: Literal[
        "Rising", "Falling", "Rising-Falling", "Falling-Rising", "Level", "Complex"
    ]
    errorDescription: str = Field(..., max_length=200)
    improvementAdvice: str = Field(..., max_length=150)
    errorStartIndex: int
    errorEndIndex: int


def extract_pitch_faster(y: np.ndarray, sr: int) -> np.ndarray:
    """Trích xuất pitch nhanh hơn bằng `librosa.yin`"""
    # Chia nhỏ tín hiệu âm thanh thành cửa sổ nhỏ hơn
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=512,
    )

    # Xử lý NaN thành 0
    return np.nan_to_num(f0)


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
    @log_execution_time
    async def get_gpt_analysis(
        text: str, actual_intonation: str, rule_analysis: IntonationAnalysis
    ) -> Dict:
        """Phân tích và tối ưu phản hồi từ AI"""
        if not isinstance(text, str):
            raise TypeError(f"Expected `text` to be a string but got {type(text)}")

        prompt = f"""
        You are an expert in phonetics analysis. Analyze the speech below and provide feedback.
        
        📌 **INPUT:**
        - **Text:** "{text}"
        - **Detected Intonation:** "{actual_intonation}"
        - **Expected Intonation:** "{rule_analysis.expected_type}"
        - **Analysis Confidence:** {rule_analysis.confidence:.2f}

        ⚠️ **TASK:**
        - Identify if intonation is incorrect.
        - If wrong, explain **briefly** (MAX **2 sentences**).
        - Give **1 short improvement tip**.

        📜 **RESPONSE FORMAT:**
        {{
            "expectedIntonationType": "Falling",
            "errorDescription": "The intonation is unclear. Try a lower pitch at the end.",
            "improvementAdvice": "Practice lowering your pitch on the final word.",
            "errorStartIndex": {rule_analysis.error_start},
            "errorEndIndex": {rule_analysis.error_end}
        }}

        ❌ **DO NOT return any markdown or extra text. Only JSON.**
        """

        try:
            client = instructor.from_groq(groq_client, mode = instructor.Mode.JSON)
            response = await client.chat.completions.create(
                model="llama-3.2-1b-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_model=IntonationFeedback
            )
            print(response.model_dump_json())
            return response.model_dump()
        except Exception as e:
            logging.error(e)
            return {
                "expectedIntonationType": rule_analysis.expected_type,
                "errorDescription": "Error processing analysis.",
                "improvementAdvice": "Try re-recording with clearer intonation.",
                "errorStartIndex": rule_analysis.error_start,
                "errorEndIndex": rule_analysis.error_end,
            }

    @staticmethod
    @log_execution_time
    async def process_audio(actual_text: str, audio_path: str) -> dict:
        """Xử lý âm thanh, phân tích trọng âm, và đo thời gian từng bước"""
        try:
            logging.info("🔄 Bắt đầu xử lý audio...")
            start_total = time.perf_counter()

            start = time.perf_counter()
            y, sr = librosa.load(audio_path, sr=None)
            end = time.perf_counter()
            logging.info(f"✅ Load audio hoàn thành (⏱️ {end - start:.4f}s)")

            start = time.perf_counter()
            f0_cleaned = extract_pitch_faster(y, sr)
            end = time.perf_counter()
            logging.info(f"✅ Trích xuất pitch (YIN) hoàn thành (⏱️ {end - start:.4f}s)")

            start = time.perf_counter()
            if not actual_text:
                raise ValueError(
                    "🚨 Lỗi: Không thể chuyển đổi giọng nói thành văn bản!"
                )
            end = time.perf_counter()
            logging.info(f"✅ Kiểm tra transcription hoàn thành (⏱️ {end - start:.4f}s)")

            start = time.perf_counter()
            rule_analysis = InnotationEvaluationService.analyze_intonation(
                actual_text, f0_cleaned
            )
            end = time.perf_counter()
            logging.info(
                f"✅ Phân tích trọng âm bằng quy tắc hoàn thành (⏱️ {end - start:.4f}s)"
            )

            start = time.perf_counter()
            actual_intonation, _ = (
                InnotationEvaluationService.rules.analyze_pitch_pattern(f0_cleaned)
            )
            end = time.perf_counter()
            logging.info(
                f"✅ Xác định intonation thực tế hoàn thành (⏱️ {end - start:.4f}s)"
            )

            start = time.perf_counter()
            gpt_analysis = await InnotationEvaluationService.get_gpt_analysis(
                actual_text, actual_intonation, rule_analysis
            )
            end = time.perf_counter()
            logging.info(f"✅ Gọi OpenAI API hoàn thành (⏱️ {end - start:.4f}s)")

            end_total = time.perf_counter()
            logging.info(
                f"✅ Hoàn thành toàn bộ xử lý (⏱️ {end_total - start_total:.4f}s)"
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
            logging.error(f"🚨 Lỗi trong process_audio: {e}")
            return {"error": str(e), "errorType": type(e).__name__}


async def main():
    audio_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/recorded_audio.mp3"
    )
    actual_text = "Commit to speaking English every day, even without a partner. Practice thinking in English to improve fluency and accuracy. The more you read, the more natural you become."
    result = await InnotationEvaluationService.process_audio(actual_text, audio_path)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
