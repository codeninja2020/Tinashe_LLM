from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import requests
import asyncio

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from get_embedding_function import get_embedding_function

# GROQ API key
GROQ_API_KEY = "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"

# Instantiate ChatGroq with the desired model
llm_groq = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name='mixtral-8x7b-32768'
)

# Create an instance of FastAPI
app = FastAPI()

# Path for Chroma DB
CHROMA_PATH = "chroma"

# Prompt template for RAG integration
PROMPT_TEMPLATE = """
Answer in first person based only on the following context. Do not include details about Moses Jambo; include Pauline Gutsa:

{context}

---

Answer the question in first person based on the above context: {question}
"""

# Function to perform the RAG query and get context from Chroma and generate a response from ChatGroq
async def query_rag(query_text: str):
    try:
        # Prepare the DB and embedding function
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB for relevant documents
        results = db.similarity_search_with_score(query_text, k=5)

        # Extract context from the retrieved documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Prepare the prompt for ChatGroq
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Get the response from ChatGroq
        response_text = await asyncio.to_thread(llm_groq.invoke, prompt)  # Use asyncio.to_thread for blocking calls

        # Extract and format the sources (document IDs)
        sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
        formatted_response = f"{response_text.content}\n"  # Add sources if needed

        return formatted_response
    except Exception as e:
        return f"Error: {str(e)}"

# Route for the root endpoint to query RAG
class QueryModel(BaseModel):
    question: str

@app.post("/query")
async def handle_query(query_model: QueryModel):
    try:
        rag_response = await query_rag(query_model.question)
        return {"response": rag_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/query")
async def handle_query(query_model: QueryModel):
    try:
        rag_response = await query_rag(query_model.question)
        return {"response": rag_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4100)

#gcloud compute scp --ssh-key-file ~/.ssh/google_compute_engine local_rag_api2.py tinashe@texciteai:/home/tinashe/Tinashe_LLM --zone us-central1-c --project abstract-lane-447113-t8
