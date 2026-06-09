"""
C-RAG: Corrective Retrieval-Augmented Generation
Usage:
    python main.py "Your question here"
    python main.py  # prompts interactively
"""
import argparse
import sys
from nodes import run

EMPTY_STATE = {
    "question": "",
    "docs": [],
    "good_docs": [],
    "verdict": "",
    "reason": "",
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
    "web_query": "",
    "web_docs": [],
    "answer": "",
}


def main():
    parser = argparse.ArgumentParser(
        description="C-RAG: Corrective Retrieval-Augmented Generation pipeline"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask the pipeline (omit for interactive mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate pipeline state (verdict, kept strips, etc.)",
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Enter your question: ").strip()
    if not question:
        print("Error: question cannot be empty.", file=sys.stderr)
        sys.exit(1)

    print(f"\n🔍 Running C-RAG pipeline for: '{question}'\n")

    result = run({**EMPTY_STATE, "question": question})

    if args.verbose:
        print(f"Verdict     : {result.get('verdict')}")
        print(f"Reason      : {result.get('reason')}")
        print(f"Web query   : {result.get('web_query') or '(not used)'}")
        print(f"Strips kept : {len(result.get('kept_strips', []))}")
        print()

    print("Answer:\n")
    print(result.get("answer", "No answer generated."))


if __name__ == "__main__":
    main()