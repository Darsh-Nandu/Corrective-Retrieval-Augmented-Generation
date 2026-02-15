from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from rag_state import State, DocEvalScore, KeepOrDrop, WebQuery
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import List
import re

load_dotenv()


# Data Loading and Storing Vectorized Docs in Vector Databases
docs = (
    PyPDFLoader("./documents/test.pdf").load()
    #PyPDFLoader("./documents/book1.pdf").load()
    #+ PyPDFLoader("./documents/book2.pdf").load()
    #+ PyPDFLoader("./documents/book3.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
for d in chunks:
    d.page_content =  d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Model
llm = ChatOllama(model="llama3.1")

# Retrieve Node
def retrieve_node(state: State) -> dict:
    q = state['question']
    return {'docs': retriever.invoke(q)} 


# Score-based doc evaluator

UPPER_TH = 0.7
LOWER_TH = 0.3

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

def eval_each_doc_node(state: State) -> dict:

    q= state['question']
    scores: List[float] = []
    good: List[Document] = []

    for d in state['docs']:
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)

        # Keep any doc above LOWER_TH as "weakly relevant"
        if out.score >  UPPER_TH:
            good.append(d)
    
    # CORRECT: at least one doc > UPPER_TH
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    # INCORRECT: all docs < LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}.",
        }

    # AMBIGUOUS: otherwise
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }


# Sentance Level Decomposer
def decompose_to_sentances(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentances = re.split(r"(?<=[.1?])\s+", text)
    return [s.strip() for s in sentances if len(s.strip()) > 20]

# Filter (LLM Judge)

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop) 


#Knowledge refinement
#(CORRECT => internal only)
#(INCORRECT => web only)
#(AMBIGOUS => internal + Web)

def refine_node(state: State) -> dict:
    q = state['question']

    if state.get("verdict") == "CORRECT":
        docs_to_use = state['good_docs']
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    strips = decompose_to_sentances(context)

    kept: List[str] = []
    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s)

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }

# Query Rewrite for Web Search

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6-14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}")
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


# Web Search Node
tavily = TavilySearchResults(max_results=5)

def web_search_node(state: State) -> dict:
    q = state.get('web_query') or state['question']
    results = tavily.invoke({"query":q})

    web_docs: List[Document] = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {'web_docs': web_docs}

# Generate 
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def generate_node(state: State) -> dict:
    out = (answer_prompt | llm).invoke({"question": state["question"], "context": state["refined_context"]})
    print(out.content)
    return {'answer': out.content}

# Routing 
def route_after_eval(state: State) -> str:
    if state['verdict'] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"
    

def build_graph():
    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)

    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)

    g.add_node("refine", refine_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")

    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query",
        }
    )

    # non-correct path
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")

    # correct path already goes to refine
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    app = g.compile()

    return app


def run(input: dict ):
    rag_app = build_graph()
    res = rag_app.invoke(input)
    return res