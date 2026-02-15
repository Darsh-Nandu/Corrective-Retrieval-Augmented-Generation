from typing import List, TypedDict
from langchain_core.documents import Document
from pydantic import BaseModel

class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    web_docs: List[Document]

    answer: str

class DocEvalScore(BaseModel):
    score: float
    reason: str

class KeepOrDrop(BaseModel):
    keep: bool

class WebQuery(BaseModel):
    query: str