# C-RAG: Corrective Retrieval-Augmented Generation

C-RAG is a modular implementation of a Corrective Retrieval-Augmented Generation (RAG) pipeline. It improves answer reliability by systematically validating, filtering, and refining retrieved context prior to final generation — addressing core failure modes common in standard RAG architectures.

---

## Overview

Standard RAG systems frequently suffer from three interconnected problems: retrieval of partially relevant documents, direct use of weak context during generation, and hallucination when retrieval quality is low. C-RAG introduces a corrective layer between retrieval and generation that grades documents, extracts relevant text strips, filters noisy context, and refines it before producing a final answer. The result is a more reliable, interpretable, and auditable pipeline.

---

## Features

- **Document Retrieval** — Fetches candidate documents based on the user query
- **Context Grading** — Evaluates retrieved documents for relevance and quality
- **Strip Extraction** — Isolates the most pertinent text segments from each document
- **Strip Filtering** — Removes weak or noisy strips that could degrade generation
- **Context Refinement** — Consolidates and polishes the surviving context
- **Verified Answer Generation** — Produces a final answer grounded in validated context

---

## Motivation

Retrieval-Augmented Generation works best when the context fed to the language model is accurate and relevant. In practice, retrievers often return documents that are only partially on-topic, and using such context without correction leads to hallucinated or low-confidence outputs.

C-RAG addresses this gap by inserting a multi-stage corrective process between retrieval and generation. Each stage is independently auditable, making the pipeline transparent and easy to debug or extend.

---

## Pipeline

```
User Query
    │
    ▼
Retriever
    │
    ▼
Document Grading
    │
    ▼
Strip Extraction
    │
    ▼
Strip Filtering
    │
    ▼
Context Refinement
    │
    ▼
Final Answer Generation
```

---

## Project Structure

```
C-RAG/
├── main.py               # Entry point; configure and run the pipeline
├── nodes.py              # Individual pipeline stage implementations
├── rag_state.py          # Shared state schema across pipeline nodes
├── documents/            # Source documents for retrieval
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (API keys, config)
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Darsh-Nandu/Corrective-Retrieval-Augmented-Generation
cd Corrective-Retrieval-Augmented-Generation
```

**2. Create and activate a virtual environment**

```bash
python -m venv crag_venv
```

- **Windows:** `crag_venv\Scripts\activate`
- **macOS / Linux:** `source crag_venv/bin/activate`

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

Open `main.py` and set your query in the `run()` call:

```python
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
```

Then execute the pipeline:

```bash
python main.py
```

---

## Goals

- Improve factual reliability in RAG systems
- Reduce hallucinations through structured context validation
- Provide a modular, inspectable pipeline architecture
- Serve as a clean reference implementation for corrective RAG research

---

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

This project is open source.

---

## Acknowledgements

If you find this project useful in your work or research, consider giving it a star on GitHub — it helps others discover the project.
