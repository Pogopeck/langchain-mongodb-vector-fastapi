# ingest.py
from vector_store import vector_store
from langchain_core.documents import Document

docs = [
    Document(page_content="MongoDB is a NoSQL document database.", metadata={"source": "doc1"}),
    Document(page_content="FastAPI is a modern Python web framework for building APIs.", metadata={"source": "doc2"}),
    Document(page_content="Gemini is a powerful AI model by Google.", metadata={"source": "doc3"}),
]

vector_store.add_documents(docs)
print("✅ Ingested 3 documents into MongoDB Atlas with embeddings.")