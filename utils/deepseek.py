import re
import json
from pydantic import Json


class DeepSeekResponseParser:
    @staticmethod
    def extract_reasoning(response_content: str) -> str:
        reasoning_match = re.search(
            r"<think>(.*?)</think>", response_content, re.DOTALL
        )
        return (
            reasoning_match.group(1).strip()
            if reasoning_match
            else "No reasoning provided."
        )

    @staticmethod
    def extract_json(response_content: str) -> dict:
        json_match = re.search(r"</think>\s*(\{.*\})", response_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON format"})
        return {}

    @classmethod
    def parse_response(cls, response_content: str) -> Json:
        return {
            "reasoning": cls.extract_reasoning(response_content),
            "json": json.dumps(cls.extract_json(response_content), ensure_ascii=True),
        }


if __name__ == "__main__":
    response_content = """
    <think>Here is the reasoning part of the response.</think>
    {
        "key1": "value1",
        "key2": "value2"
    }
    """

    parsed_response = DeepSeekResponseParser.parse_response(response_content)

    print(parsed_response["reasoning"])
    print(parsed_response["json"])
