import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model='llama3')

    db = Chroma(persist_directory=os.environ['CHROMA_PATH'], embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(query, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    template = """Act as a receptionist for Mar Baselios College of engineering and technology and use the following pieces of context
    to answer the questions at the end. If you don't know the answer just say that you don't know,
    don't try to make up an answer. Always say "thanks for asking!" at the end of the answer.

            {chat_history}

            {context}

            Question: {question}

            Helpful answer:"""

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(chat_history=chat_history, context=context_text, question=query)

    llm = ChatOllama(model="llama3")
    response_text = llm.predict(prompt)

    # Append dictionary to chat history
    chat_history.append({"query": query, "response": response_text})
    print(chat_history)

    formatted_response = f"{response_text}"
    return formatted_response


if __name__ == "__main__":
    res = run_llm(query="Who is Dr Tessy Mathew?")
    print(res)
