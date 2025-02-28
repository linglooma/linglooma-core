import io
import base64
from pydub import AudioSegment
from pydantic import BaseModel, ValidationError
from typing import Literal


class AudioOutput(BaseModel):
    output_format: Literal["binary", "base64"]
    data: bytes | str

    def get_bytes(self) -> bytes:
        """Returns raw bytes regardless of output format."""
        if isinstance(self.data, str):  # If base64, decode it
            return base64.b64decode(self.data)
        return self.data


class AudioReader:
    @staticmethod
    def encode_audio_to_base64(audio_path: str) -> str:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    @staticmethod
    async def read_audio(
        file_path: str, output_format: Literal["binary", "base64"]
    ) -> AudioOutput:
        try:
            audio = AudioSegment.from_file(file_path)
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            binary_data = buffer.getvalue()

            if output_format == "binary":
                return AudioOutput(output_format=output_format, data=binary_data)
            elif output_format == "base64":
                base64_data = base64.b64encode(binary_data).decode("utf-8")
                return AudioOutput(output_format=output_format, data=base64_data)
        except Exception as e:
            raise RuntimeError(f"Error processing audio file: {e}")


if __name__ == "__main__":
    file_path = (
        "/home/xuananle/Documents/Linglooma/Linglooma-core/resources/audio/part2-2.mp3"
    )

    try:
        audio_binary = AudioReader.read_audio(file_path, "binary")
        audio_base64 = AudioReader.read_audio(file_path, "base64")

        print(f"Binary output (first 100 bytes): {audio_binary.data[:100]}")
        print(f"Base64 output (first 100 characters): {audio_base64.data[:100]}")

    except ValidationError as e:
        print(f"Validation Error: {e}")
