import os
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import chromadb  # Assuming Chroma database is used


def run_llm(query: str):
    # Use a BERT-based model for embeddings
    embeddings_model = SentenceTransformer('bert-base-nli-mean-tokens')  # Replace with any compatible model

    # Create the Chroma database with BERT embeddings
    db = chromadb.Client(os.environ['CHROMA_PATH'], embedding_function=embeddings_model.encode)

    # Perform similarity search on the database
    results = db.similarity_search_with_relevance_scores(query, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return "Unable to find matching results."

    # Concatenate context from the retrieved results
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Use BERT for Question-Answering
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad",
                           tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

    # Prepare question and context for BERT
    answer = qa_pipeline(question=query, context=context)["answer"]

    # Format the response
    formatted_response = f"{answer}\n\nThanks for asking!"
    return formatted_response
