# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import vector_store
from llm import llm

app = FastAPI(title="RAG with Gemini + MongoDB Atlas")

class QueryRequest(BaseModel):
    question: str
    k: int = 3

# Simple RAG chain
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the following context to answer the question."),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        answer = rag_chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Gemini RAG API with MongoDB Atlas Vector Search"}