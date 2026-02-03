"""Build and test FAISS indexes for CareFlow."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from care_database import get_database
from vector_store_faiss import (
    FAISSIndex,
    load_guidelines_from_markdown,
    get_patient_index,
    get_guidelines_index,
)


def build_patient_index():
    """Build FAISS index from patient notes in database."""
    print("\n" + "=" * 60)
    print("BUILDING PATIENT NOTES INDEX")
    print("=" * 60)

    db = get_database()
    patients = db.get_all_patients()

    documents = []
    for patient in patients:
        notes = db.get_patient_notes(patient["patient_id"])
        for note in notes:
            documents.append({
                "id": f"{patient['patient_id']}_{note['id'][:8]}",
                "text": f"Patient: {patient['name']} ({patient['patient_id']})\nDate: {note['note_date']}\n\n{note['note_text']}",
                "metadata": {
                    "patient_id": patient["patient_id"],
                    "patient_name": patient["name"],
                    "note_date": note["note_date"],
                    "note_id": note["id"],
                }
            })

    if not documents:
        print("No patient notes found in database!")
        return None

    print(f"Found {len(documents)} patient notes")

    index = get_patient_index()
    index.build_index(documents)
    index.save()

    return index


def build_guidelines_index():
    """Build FAISS index from medical guideline markdown files."""
    print("\n" + "=" * 60)
    print("BUILDING MEDICAL GUIDELINES INDEX")
    print("=" * 60)

    documents = load_guidelines_from_markdown()

    if not documents:
        print("No guidelines found!")
        return None

    index = get_guidelines_index()
    index.build_index(documents)
    index.save()

    return index


def test_patient_queries(index: FAISSIndex):
    """Test queries against patient notes index."""
    print("\n" + "=" * 60)
    print("TESTING PATIENT NOTES QUERIES")
    print("=" * 60)

    queries = [
        "patient with high A1C not at goal",
        "hypertension not on ACE inhibitor",
        "diabetes well controlled",
    ]

    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        results = index.query(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result['score']:.3f}):")
            print(f"    ID: {result['id']}")
            print(f"    Patient: {result['metadata'].get('patient_name', 'N/A')}")
            # Show first 150 chars of text
            text_preview = result['text'][:150].replace('\n', ' ')
            print(f"    Preview: {text_preview}...")


def test_guideline_queries(index: FAISSIndex):
    """Test queries against guidelines index."""
    print("\n" + "=" * 60)
    print("TESTING MEDICAL GUIDELINES QUERIES")
    print("=" * 60)

    queries = [
        "A1C target for diabetic patients",
        "ACE inhibitor hypertension diabetes",
        "when to do foot exam",
        "statin therapy recommendations",
    ]

    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        results = index.query(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result['score']:.3f}):")
            print(f"    ID: {result['id']}")
            print(f"    Category: {result['metadata'].get('category', 'N/A')}")
            print(f"    Source: {result['metadata'].get('source', 'N/A')}")
            # Show first 100 chars of text
            text_preview = result['text'][:100].replace('\n', ' ')
            print(f"    Preview: {text_preview}...")


def main():
    """Build indexes and run test queries."""
    print("\n" + "#" * 60)
    print("# CareFlow FAISS Index Builder")
    print("#" * 60)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY not found in environment!")
        print("Set it in .env file or environment variable.")
        sys.exit(1)

    print(f"\nOpenAI API Key: ...{api_key[-8:]}")

    # Build indexes
    patient_index = build_patient_index()
    guidelines_index = build_guidelines_index()

    # Test queries
    if patient_index:
        test_patient_queries(patient_index)

    if guidelines_index:
        test_guideline_queries(guidelines_index)

    print("\n" + "=" * 60)
    print("INDEX BUILD AND TEST COMPLETE")
    print("=" * 60)
    print("\nIndexes saved to:")
    print("  - indexes/patients/")
    print("  - indexes/guidelines/")


if __name__ == "__main__":
    main()
