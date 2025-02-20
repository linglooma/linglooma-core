from pydantic import BaseModel
from typing import List, Optional


class PhonemeErrorDetail(BaseModel):
    word: str
    errorType: str
    actualPhoneme: str
    expectedPhoneme: str
    errorDescription: str
    improvementAdvice: str
    errorStartIndex: int
    errorEndIndex: int


class WordStressErrorDetail(BaseModel):
    word: str
    syllableBreakdown: List[str]
    errorType: str
    actualStressedSyllableIndex: int
    expectedStressedSyllableIndex: int
    errorDescription: str
    improvementAdvice: str
    errorStartIndex: int
    errorEndIndex: int


class IntonationErrorDetail(BaseModel):
    clauseText: str
    actualIntonationType: str
    expectedIntonationType: str
    errorDescription: str
    improvementAdvice: str
    errorStartIndex: int
    errorEndIndex: int


class PronunciationAssessment(BaseModel):
    actualPhoneticTranscription: str
    expectedPhoneticTranscription: str
    phonemeErrorDetails: Optional[List[PhonemeErrorDetail]] = []
    wordStressErrorDetails: Optional[List[WordStressErrorDetail]] = []
    intonationErrorDetails: Optional[IntonationErrorDetail] = None
