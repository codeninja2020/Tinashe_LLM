import os
import chainlit as cl
import requests
import asyncio

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

# Function to perform the RAG query and get context from Chroma and generate response from Ollama
async def query_rag(query_text: str):
    try:
        # Prepare the DB and embedding function
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB for relevant documents
        results = db.similarity_search_with_score(query_text, k=5)

        # Extract context from the retrieved documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # Prepare the prompt for Ollama
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Instantiate the Ollama model (using Mistral model)
        model = Ollama(model="mistral")

        # Get the response from Ollama
        print("Sending prompt to Ollama...")
        response_text = await asyncio.to_thread(model.invoke, prompt)  # Use asyncio.to_thread for blocking calls

        # Extract and format the sources (document IDs)
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"

        # Return the formatted response text
        return formatted_response
    except Exception as e:
        print(f"Error in query_rag: {str(e)}")
        return f"Error: {str(e)}"



@cl.on_chat_start
def start_chat():
    system_message = {
        "role": "system",
        "content": "You are a religious leader using information in the database to accurately respond to user queries, you have deep knowledge of the text. If asked your name is Patriach"
    }
    cl.user_session.set("message_history", system_message)


# @cl.on_chat_resume
# async def on_chat_resume(thread: ThreadDict):
#  print("The user resumed a previous chat session!")

# @cl.on_chat_start
# async def on_chat_start():
# await cl.Message(content="Hello i am Patriach.AI").send()

@cl.on_message
async def on_message(msg: cl.Message):
    # await cl.Message(content="Processing: "+msg.content+"...").send()

    # Create a placeholder for the animated loading
    loading_message = await cl.Message(content="⏳ Processing your request...").send()

    # Simulate an animated loading indicator
    async def animate_loading(message):
        dots = ["⏳ Processing", "⏳ Processing.", "⏳ Processing..", "⏳ Processing..."]
        while True:
            for dot in dots:
                await message.update(content=dot)
                await asyncio.sleep(0.5)  # Adjust speed of animation as needed

    # Start the animation in a separate task
    animation_task = asyncio.create_task(animate_loading(loading_message))

    await cl.sleep(20)

    """content = f"Processing: {msg.content}..."

    sent_message = await cl.Message(content="").send()

    for i in range(len(content)):
        await sent_message.update(content=content[:i + 1])
        await asyncio.sleep(0.05)  # Adjust the delay for desired typing speed

    # Call the query function once the typing effect completes
    response = await query_data.query_rag(msg.content)
    await sent_message.update(content=response)

    await query_data.query_rag(msg.content)"""
    # print("The user sent: ", msg.content)

    """messages = cl.user_session.get("message_history")
    if len(msg.elements) > 0:
        for element in msg.elements:
            with open(element.path, "r") as uploaded_file:
                content = uploaded_file.read()
            messages.append({"role": "user", "content": content})
            confirm_message = cl.Message(content=f"Uploaded file: {element.name}")
            await confirm_message.send()

    messages.append({"role": "user", "content": ""})

    startQuery.query_rag(messages)

    await cl.Message(content=startQuery.query_rag(messages) + "...").send()"""

    # Query the RAG model using subprocess
    command = ['python3', 'query_data.py', msg.content]
    result = subprocess.run(command, capture_output=True, text=True)

    # Capture the standard output (stdout) from the command
    #response = result.stdout.strip()
    output = result.stdout

    # Extract the response message and sources using regex or parsing
    response_match = re.search(r"Response:\s*(.*)", output, re.DOTALL)
    sources_match = re.search(r"Sources:\s*\[(.*)\]", output)

    youtube_url = "https://www.youtube.com"
    thumbnail_url = f"https://img.youtube.com/vi/your_video_id/maxresdefault.jpg"

    # Extract and clean the response
    response = response_match.group(1).strip() if response_match else "No response found."
    print(f"Extracted Response:\n{response}")  # Debug: Check the response

    # Extract and split the sources into a list
    base_path = "/"
    sources_raw = sources_match.group(1).strip() if sources_match else ""
    print(f"Raw Sources:\n{sources_raw}")  # Debug: Check raw sources

    sources = [
        os.path.join(base_path, src.strip().strip("'"))
        for src in sources_raw.split(",")
    ] if sources_raw else []
    print(f"Formatted Sources:\n{sources}")  # Debug: Check formatted sources

    # Format sources as clickable links
    clickable_sources = "\n".join([f"[Source {i + 1}]({src})" for i, src in enumerate(sources)])

    # Send the response and formatted sources in Chainlit
    await cl.Message(
        content=f"{response}\n\nSources:\n{clickable_sources}\n" + f"🎥 Check out this YouTube video: {youtube_url}").send()

    # Send the response as part of the message content
    # await cl.Message(content=response+"\n\n"+f"🎥 Check out this YouTube video: {youtube_url}\n\nHere is a preview:\n{thumbnail_url}").send()

    # Send a message with the thumbnail image and the video URL
    # await cl.Message(content=f"🎥 Check out this YouTube video: {youtube_url}\n\nHere is a preview:\n{thumbnail_url}").send()



@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


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
