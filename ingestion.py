import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer  # Import BERT embeddings model

load_dotenv()

if __name__ == "__main__":
    print("Ingestion...")

    # Load and process the document
    loader = TextLoader("./mbcet_website_data.txt")
    document = loader.load()
    print(f"loaded {len(document)} documents")
    print("splitting...")

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Delete any existing Chroma database at the specified path
    if os.path.exists(os.environ['CHROMA_PATH']):
        shutil.rmtree(os.environ['CHROMA_PATH'])

    # Initialize Sentence-BERT embeddings model
    embeddings_model = SentenceTransformer("bert-large-nli-mean-tokens")

    # Pass the embedding function directly
    db = Chroma.from_documents(
        texts, embedding_function=embeddings_model.encode, persist_directory=os.environ['CHROMA_PATH']
    )

    # Persist the database
    db.persist()
    print(f"Saved {len(texts)} chunks to {os.environ['CHROMA_PATH']}")

    print("finish")
