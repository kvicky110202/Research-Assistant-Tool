from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ================== CONSTANTS ==================
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

# ================== INIT ==================
def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500,
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )

# ================== INGEST ==================
def process_urls(urls):
    yield "Initializing components"
    initialize_components()

    yield "Resetting vector store"
    vector_store.reset_collection()

    yield "Loading data"
    loader = WebBaseLoader(web_paths=urls)
    data = loader.load()

    yield "Splitting text"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        separators=["\n\n", "\n", ".", " "],
    )

    docs = splitter.split_documents(data)

    yield "Adding docs to vector DB"
    ids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=ids)

    print("Done")

# ================== LCEL RAG ==================
def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector store not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
You are a research assistant.
Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    # fetch sources explicitly
    docs = retriever.invoke(query)
    sources = list(
        {doc.metadata.get("source", "Unknown") for doc in docs}
    )

    return answer, sources

# ================== TEST ==================
if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Mahatma_Gandhi"
    ]

    process_urls(urls)
    answer, sources = generate_answer(
        "What is Mahatma Gandhi's date of birth and age?"
    )

    print("Answer:", answer)
    print("Sources:", sources)
