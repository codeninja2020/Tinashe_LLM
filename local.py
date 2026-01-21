import os
import chainlit as cl
import requests
import asyncio
import subprocess
import re

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

# Path for Chroma DB
CHROMA_PATH = "chroma"

# Prompt template for RAG integration
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# Function to perform the RAG query and get context from Chroma and generate a response from Ollama
async def query_rag(query_text: str):
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model="mistral")
        response_text = await asyncio.to_thread(model.invoke, prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        return formatted_response
    except Exception as e:
        return f"Error in query_rag: {str(e)}"


@cl.on_chat_start
def start_chat():
    system_message = {
        "role": "system",
        "content": "You are a religious leader using information in the database to accurately respond to user queries. You have deep knowledge of the text. If asked, your name is Patriach."
    }
    cl.user_session.set("message_history", system_message)


@cl.on_message
async def on_message(msg: cl.Message):
    loading_message = await cl.Message(content="⏳ Processing your request...").send()
    # Send a placeholder message to indicate processing
    loading_message = await cl.Message(content="⏳ Processing your request...").send()
    print(f"Processing request: {msg.content}")

    try:
        # Perform the RAG query to retrieve documents and generate the response
        rag_response = await query_rag(msg.content)
        print(f"RAG response: {rag_response}")

        # Send the RAG response back to the user
        await cl.Message(content=rag_response).send()

        # Example YouTube URL and its corresponding thumbnail
        youtube_url = "https://www.youtube.com"
        thumbnail_url = f"https://img.youtube.com/vi/your_video_id/maxresdefault.jpg"

        # Format sources as clickable links
        # Send the response, formatted sources, and the YouTube video link in Chainlit
        await cl.Message(
            content=f"🎥 Check out this YouTube video: {youtube_url}"
        ).send()
        await cl.Message(content=f"Here is a preview of the video:\n{thumbnail_url}").send()

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
            label="General Prompt 1",
            message="Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            icon="/public/lifelighter.svg",
        ),
        cl.Starter(
            label="General Prompt 2",
            message="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum.",
            icon="/public/ministers.svg",
        ),
        cl.Starter(
            label="General Prompt 3",
            message="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras mattis consectetur purus sit amet fermentum.",
            icon="/public/relationships.svg",
        ),
        cl.Starter(
            label="General Prompt 4",
            message="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur blandit tempus porttitor.",
            icon="/public/deliverance.svg",
        ),
        cl.Starter(
            label="General Prompt 5",
            message="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus sagittis lacus vel augue laoreet rutrum.",
            icon="/public/prosperity.svg",
        )
    ]
