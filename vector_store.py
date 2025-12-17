# app/vector_store.py
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv("MONGODB_URI"))
collection = client[os.getenv("DATABASE_NAME")][os.getenv("COLLECTION_NAME")]

# Embedding model (local, no API needed)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=os.getenv("INDEX_NAME"),
)