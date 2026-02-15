🚀 C-RAG: Corrective Retrieval-Augmented Generation

C-RAG is a modular implementation of a Corrective Retrieval-Augmented Generation (RAG) pipeline designed to improve answer reliability by validating, filtering, and refining retrieved context before final generation.

This project demonstrates:

📄 Document retrieval

🧠 Context grading

🧹 Strip extraction and filtering

🔁 Context refinement

✅ Verified final answer generation

🧠 Motivation

Standard RAG systems often:

Retrieve partially relevant documents

Use weak context directly in generation

Hallucinate when retrieval quality is low

C-RAG introduces a corrective layer that:

Grades retrieved documents

Extracts relevant text strips

Filters weak or noisy context

Refines context before final generation

The result is a more reliable and interpretable RAG pipeline.

🏗 Project Structure
C-RAG/
│
├── main.py              # Entry point
├── nodes.py             # Pipeline logic (orchestrator + workers)
├── rag_state.py         # Shared state schema
├── documents/           # Local document store
├── requirements.txt
└── .env                 # Environment variables (not committed)

⚙️ Pipeline Flow
User Query
    ↓
Retriever
    ↓
Document Grading
    ↓
Strip Extraction
    ↓
Strip Filtering
    ↓
Context Refinement
    ↓
Final Answer Generation


Each stage updates a shared state object, making the system modular and easy to debug.

🚀 Installation

Clone the repository:

git clone https://github.com/your-username/C-RAG.git
cd C-RAG


Create a virtual environment:

python -m venv crag_venv


Activate environment:

Windows

crag_venv\Scripts\activate


Linux / Mac

source crag_venv/bin/activate


Install dependencies:

pip install -r requirements.txt

🔐 Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_api_key_here


Make sure .env is added to .gitignore.

▶️ Usage

Modify the query inside main.py:

run({
    "question": "Batch normalization vs layer normalization",
    "docs": [],
    "good_docs": [],
    "verdict": "",
    "reason": "",
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
})


Run the pipeline:

python main.py

🧩 Core Components
rag_state.py

Defines the shared structured state passed across pipeline stages.

nodes.py

Contains:

Orchestrator logic

Worker nodes

Reducer logic

Implements the corrective retrieval strategy.

documents/

Local knowledge base used for retrieval.

🎯 Goals

Improve factual reliability in RAG systems

Reduce hallucinations

Make RAG pipelines modular and inspectable

Provide a clean educational implementation

🔮 Future Improvements

Hybrid retrieval (BM25 + embeddings)

Cross-encoder re-ranking

Confidence scoring

Streaming responses

Web interface

Deployment-ready architecture
