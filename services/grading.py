from config.client import openai_client
import json
import base64

IELTS_GRADING_PROMPT = """
You are an official IELTS Speaking examiner with expertise in assessing spoken English. Your task is to evaluate the given audio response using the official IELTS Speaking Band Descriptors.

### **📝 Grading Criteria (Based on IELTS Band Descriptors)**
Evaluate the response on a **0 to 9** scale according to the following four official criteria:
REMEMBER TO STRICLY FOLLOW THE BAND DESCRIPTORS
---

### **1️⃣ Fluency & Coherence (FC)**
- **Band 9**: Fluent, with only very occasional repetition or self-correction. Hesitation is only for idea preparation, not word retrieval. Fully coherent and appropriately extended responses. Any hesitation that occurs is used only to
prepare the content of the next utterance and not to find words or grammar. Speech is situationally appropriate and cohesive features used effectively to hold the discourse together.
- **Band 8**: Fluent with only very occasional repetition or self-correction. Hesitation may occasionally be used to find words or grammar, but most will be content related. Topic development is coherent, appropriate and relevant
- **Band 7**: Able to keep going and readily produce long turns without noticeable effort. Some hesitation, repetition and/or self- correction may occur, often mid-sentence and indicate problems with accessing appropriate language. However, these will not affect
coherence.
Flexible use of spoken discourse markers,
connectives and cohesive features.
- **Band 6**: Can keep going but with noticeable hesitation, repetition, and self-correction. Some loss of coherence.
- **Band 5**: Relies on repetition and self-correction, overuses discourse markers, and hesitates frequently.
- **Band 4**: Frequent pauses, difficulty in forming connected speech, and significant breakdowns in coherence.
- **Band 3**:  Frequent, sometimes long, pauses occur while
candidate searches for words.
Limited ability to link simple sentences and go
beyond simple responses to questions.
Frequently unable to convey basic message.
---

### **2️⃣ Lexical Resource (LR)**
- **Band 9**: Total flexibility, precise word use, and sustained idiomatic usage.
- **Band 8**: Wide range of vocabulary, including less common words, with minor inaccuracies.
- **Band 7**: Can discuss various topics using a range of vocabulary, including some idiomatic expressions.
- **Band 6**: Sufficient vocabulary but occasional inappropriate word choices.
- **Band 5**: Limited vocabulary with occasional misuse; struggles with paraphrasing.
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
- **Band 9**: Uses full phonological range. Stress, intonation, and connected speech are flawless. Accent does not affect intelligibility.
- **Band 8**: Wide range of phonological features; minor lapses in stress and intonation.
- **Band 7**: Pronunciation errors are minimal; slight accent influence but does not hinder understanding.
- **Band 6**: Pronunciation is mostly clear but with inconsistent control over intonation and stress.
- **Band 5**: Some mispronunciations, affecting intelligibility occasionally.
- **Band 4 & below**: Frequent mispronunciations and phoneme-level errors, making comprehension difficult.

---

### **🔹 Output Format (JSON)**
The response should follow this structured format:
```
{
    "overall": <float>,  // Rounded to the nearest 0.5
    "fluencyCoherence": <float>,     
    "lexicalResource": <float>,       
    "grammaticalRangeAccuracy": <float>,       
    "pronunciation": <float>,       
}
"""


async def grade_ielts_response(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_content = base64.b64encode(audio_file.read()).decode("utf-8")
    print("I am grading here part 1")
    response = openai_client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {"role": "system", "content": IELTS_GRADING_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Evaluate the IELTS Speaking response step-by-step as instructed above.",
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
    print("I am grading here part 2")
    print(response.choices[0].message.content)

    return json.loads(response.choices[0].message.content)
