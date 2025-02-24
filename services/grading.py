import json
import base64
import logging
from openai import AsyncOpenAI, OpenAI
from config.settings import OPENAI_API_KEY
from utils.logging import log_execution_time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class IELTSGradingService:
    IELTS_GRADING_PROMPT = """
    You are an official IELTS Speaking examiner with expertise in assessing spoken English. Your task is to evaluate the given audio response using the official IELTS Speaking Band Descriptors.

    ### **📝 Grading Criteria (Based on IELTS Band Descriptors)**
    Evaluate the response on a **0 to 9** scale according to the following four official criteria:
    REMEMBER TO STRICLY FOLLOW THE BAND DESCRIPTORS
    ---
    
    ### **1️⃣ Fluency & Coherence (FC)**
    - **Band 9**: Fluent, with only very occasional repetition or self-correction. Hesitation is only for idea preparation, not word retrieval. Fully coherent and appropriately extended responses.
    - **Band 8**: Fluent with only very occasional repetition or self-correction.
    - **Band 7**: Able to keep going and readily produce long turns without noticeable effort.
    - **Band 6**: Can keep going but with noticeable hesitation, repetition, and self-correction.
    - **Band 5**: Relies on repetition and self-correction, overuses discourse markers, and hesitates frequently.
    - **Band 4**: Frequent pauses, difficulty in forming connected speech, and significant breakdowns in coherence.
    - **Band 3**: Frequent, sometimes long, pauses occur while candidate searches for words.
    
    ---
    
    ### **2️⃣ Lexical Resource (LR)**
    - **Band 9**: Total flexibility, precise word use, and sustained idiomatic usage.
    - **Band 8**: Wide range of vocabulary, including less common words, with minor inaccuracies.
    - **Band 7**: Can discuss various topics using a range of vocabulary.
    - **Band 6**: Sufficient vocabulary but occasional inappropriate word choices.
    - **Band 5**: Limited vocabulary with occasional misuse.
    - **Band 4 & below**: Very basic vocabulary, frequent errors, and difficulty conveying meaning.
    
    ---
    
    ### **3️⃣ Grammatical Range & Accuracy (GRA)**
    - **Band 9**: Structures are precise and accurate, with only native-like minor slips.
    - **Band 8**: A wide range of structures is used flexibly; mostly error-free.
    - **Band 7**: A variety of structures are used, but some errors persist.
    - **Band 6**: Mix of simple and complex sentences with frequent errors.
    - **Band 5**: Basic structures are controlled, but complex sentences contain many errors.
    - **Band 4 & below**: Very limited grammatical control, frequent errors impede communication.
    
    ---
    
    ### **4️⃣ Pronunciation (PRON)**
    - **Band 9**: Uses full phonological range. Stress, intonation, and connected speech are flawless.
    - **Band 8**: Wide range of phonological features; minor lapses in stress and intonation.
    - **Band 7**: Pronunciation errors are minimal; slight accent influence but does not hinder understanding.
    - **Band 6**: Pronunciation is mostly clear but with inconsistent control over intonation and stress.
    - **Band 5**: Some mispronunciations, affecting intelligibility occasionally.
    - **Band 4 & below**: Frequent mispronunciations and phoneme-level errors, making comprehension difficult.
    
    ---
    RETURN ONLY THE JSON, DO NOT INCLUDE ```json, DO NOT PUT ANYTHING ELSE HERE
    ### **🔹 Output Format (JSON)**
    The response should follow this structured format:
    ```
    {
        "overall": <float>,  // Rounded to the nearest 0.5
        "fluencyCoherence": <float>,     
        "lexicalResource": <float>,       
        "grammaticalRangeAccuracy": <float>,       
        "pronunciation": <float>       
    }
    ```
    """

    @staticmethod
    def encode_audio(audio_path: str) -> str:
        """
        Encodes the given audio file to base64 format.

        :param audio_path: Path to the audio file.
        :return: Base64 encoded audio string.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                return base64.b64encode(audio_file.read()).decode("utf-8")
        except FileNotFoundError:
            logging.error(f"Audio file not found: {audio_path}")
            return None
        except Exception as e:
            logging.error(f"Error encoding audio: {e}")
            return None

    @staticmethod
    @log_execution_time
    async def grade_ielts_response(audio_path: str) -> dict:
        """
        Grades the IELTS Speaking response using OpenAI GPT-4o audio model.

        :param audio_path: Path to the audio file.
        :return: Dictionary containing IELTS scores.
        """
        logging.info("Starting IELTS grading process.")

        audio_content = IELTSGradingService.encode_audio(audio_path)
        if not audio_content:
            return {"error": "Failed to encode audio"}
        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            logging.info("Sending request to OpenAI GPT-4o audio model.")

            response = await client.chat.completions.create(
                model="gpt-4o-mini-audio-preview-2024-12-17",
                modalities=["text"],
                messages=[
                    {
                        "role": "system",
                        "content": IELTSGradingService.IELTS_GRADING_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                        Evaluate the IELTS Speaking response step-by-step as instructed above.     RETURN ONLY THE JSON, DO NOT INCLUDE ```json, DO NOT PUT ANYTHING ELSE HERE
                                        """,
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_content, "format": "mp3"},
                            },
                        ],
                    },
                ],
                temperature=0.1,
            )

            logging.info("Received response from OpenAI.")

            # Extract and parse JSON response
            raw_response = response.choices[0].message.content
            print(raw_response)
            try:
                parsed_response = json.loads(raw_response)
                logging.info(f"Successfully parsed response: {parsed_response}")
                return parsed_response
            except json.JSONDecodeError:
                logging.error("Invalid JSON response from OpenAI.")
                return {"error": "Invalid JSON response from OpenAI"}

        except Exception as e:
            logging.error(f"Error during IELTS grading: {e}")
            return {"error": "Failed to grade response"}


if __name__ == "__main__":
    import asyncio

    audio_file_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )
    parsed_response = asyncio.run(
        IELTSGradingService.grade_ielts_response(audio_file_path)
    )

    print("Final Grading Response:")
    print(parsed_response)
