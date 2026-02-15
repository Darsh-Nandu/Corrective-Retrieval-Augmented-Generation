# 🚀 C-RAG: Corrective Retrieval-Augmented Generation

C-RAG is a modular implementation of a Corrective Retrieval-Augmented Generation (RAG) pipeline designed to improve answer reliability by validating, filtering, and refining retrieved context before final generation.

---

## 📌 Features

- Document retrieval  
- Context grading  
- Strip extraction and filtering  
- Context refinement  
- Verified final answer generation  

---

## 🧠 Motivation

Standard RAG systems often:

- Retrieve partially relevant documents  
- Use weak context directly in generation  
- Hallucinate when retrieval quality is low  

C-RAG introduces a corrective layer that:

- Grades retrieved documents  
- Extracts relevant text strips  
- Filters weak or noisy context  
- Refines context before final generation  

This results in a more reliable and interpretable RAG pipeline.

---

## 🏗 Project Structure

```
C-RAG/
│
├── main.py
├── nodes.py
├── rag_state.py
├── documents/
├── requirements.txt
└── .env
```

---

## ⚙️ Pipeline Flow

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

---

## 🚀 Installation

Clone the repository:

```
git clone https://github.com/Darsh-Nandu/Corrective-Retrieval-Augmented-Generation
```

Create virtual environment:

```
python -m venv crag_venv
```

Activate:

Windows:
```
crag_venv\Scripts\activate
```

Mac/Linux:
```
source crag_venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

Make sure `.env` is added to `.gitignore`.

---

## ▶️ Usage

Modify the query inside `main.py`:

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

Run:

```
python main.py
```

---

## 🎯 Goals

- Improve factual reliability in RAG systems  
- Reduce hallucinations  
- Make RAG pipelines modular and inspectable  
- Provide a clean educational implementation  

---

## ⭐ Support

If you find this project useful, consider giving it a star.
