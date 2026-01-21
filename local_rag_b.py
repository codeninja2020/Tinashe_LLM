import os
import chainlit as cl
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

# Path for Chroma DB
CHROMA_PATH = "chroma"

# Prompt template for RAG integration
PROMPT_TEMPLATE = """
Answer in first person based only on the following context dont include details about moses jambo, include pauline gutsa:

{context}

---

Answer the question in first person based on the above context: {question}
"""

# Function to perform the RAG query and get context from Chroma and generate response from ChatGroq
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
        print("Sending prompt to ChatGroq...")
        response_text = await asyncio.to_thread(llm_groq.invoke, prompt)  # Use asyncio.to_thread for blocking calls

        # Extract and format the sources (document IDs)
        sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
        #formatted_response = f"Response: {response_text}\nSources: {sources}"
        formatted_response = f"{response_text.content}\n"#Sources: {', '.join(sources)}"

        # Return the formatted response text
        return formatted_response
    except Exception as e:
        print(f"Error in query_rag: {str(e)}")
        return f"Error: {str(e)}"

@cl.on_chat_start
def start_chat():
    # Initialize the message history when chat starts
    system_message = {
        "role": "user",
        "content": ""
    }
    cl.user_session.set("message_history", system_message)

@cl.on_message
async def on_message(msg: cl.Message):
    # Send a placeholder message to indicate processing
    loading_message = await cl.Message(content="⏳ Processing your request...").send()
    print(f"Processing request: {msg.content}")

    try:
        # Perform the RAG query to retrieve documents and generate the response
        rag_response = await query_rag(msg.content)
        print(f"RAG response: {rag_response}")

        # Send the RAG response back to the user
        await cl.Message(content=rag_response).send()

        print("Response sent back to user.")
    except Exception as e:
        print(f"Error in on_message: {str(e)}")
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="About",
            message="Tell me about you.",
            icon="/public/lifelighter.svg",
        ),
        cl.Starter(
            label="Personality",
            message="Tell me about your Personality",
            icon="/public/ministers.svg",
        ),
        cl.Starter(
            label="Activities",
            message="Tell me about activities you like doing",
            icon="/public/prosperity.svg",
        )
    ]
