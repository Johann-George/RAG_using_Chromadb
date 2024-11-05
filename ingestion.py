import os
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingestion...")

    # jq_schema = ".[]"
    # loader = JSONLoader("./results.json", jq_schema=jq_schema,text_content=False)
    loader = TextLoader("./mbcet_website_data.txt")
    document = loader.load()
    print(f"loaded {len(document)} documents")
    print("splitting...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    if os.path.exists(os.environ['CHROMA_PATH']):
        shutil.rmtree(os.environ['CHROMA_PATH'])

    embeddings = OllamaEmbeddings(model="llama3")

    db = Chroma.from_documents(
        texts, embeddings, persist_directory=os.environ['CHROMA_PATH']
    )

    db.persist()
    print(f"Saved {len(texts)} chunks to {os.environ['CHROMA_PATH']}")

    print("finish")
