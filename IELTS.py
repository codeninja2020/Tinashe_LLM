import os
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
import pandas as pd

import requests
import nltk
nltk.download('punkt_tab')

import asyncio
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from get_embedding_function import get_embedding_function
from nltk import word_tokenize, pos_tag
from collections import Counter
from nltk.corpus import wordnet
from textblob import TextBlob
import pyphen


GROQ_API_KEY = "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"

llm_groq = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name='mixtral-8x7b-32768'
)

modal_100 = " "
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """


Answer in first person based only on the following context:
{context}
---
Answer the question in first person based on the above context: {question}
"""

# Read CSV file with pandas
data_frame = pd.read_csv("student_mistakes.csv", encoding="utf-8")

# Create user session to store data


async def query_rag(query_text: str):
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        response_text = await asyncio.to_thread(llm_groq.invoke, prompt)
        sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
        return f"{response_text.content}\n"
    except Exception as e:
        return f"Error: {str(e)}"

# IELTS Practice Tool Functions

@cl.on_chat_start
async def start_chat():
    global modal_100
    global data_frame
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Choose Mode",
                values=["Practice Mode", "Test Mode"],
                initial_index=0,
            )
        ]
    ).send()

    await cl.Message(content=f"{settings['Model']}, Type Start to proceed").send()


    modal_100 = settings["Model"]

    system_message = {
        "role": "user",
        "content": ""
    }
    cl.user_session.set("message_history", system_message)
    cl.user_session.set('data', data_frame)


@cl.on_settings_update
async def settingsupdate(settings):
    global modal_100  # Access the global variable
    curr_settings = settings["Model"]
    modal_100 = curr_settings
    print("Settings updated ", settings)
    print(settings["Model"])

    await cl.Message(content=settings["Model"]).send()
    await cl.Message(content="Type Start to proceed").send()
    cl.user_session.set("current_index", 0)



@cl.on_message
async def on_message(msg: cl.Message):

    if modal_100 == "Practice Mode":
        # Display loading message
        loading_message = await cl.Message(content="⏳ Processing your IELTS practice request...").send()

        # Access session data
        current_index = cl.user_session.get("current_index", 0)
        df = cl.user_session.get('data')

        # Provide feedback for the previous question (if any)
        if current_index > 0:  # Ensure there's a previous question to analyze
            previous_row = df.iloc[current_index - 1]

            try:
                if msg.content.strip() == previous_row['Corrected_Sentence']:
                    # Correct response - provide positive feedback
                    feedback_prompt = f"""
                            The user correctly corrected the sentence: 
                            "{previous_row['Original_Sentence']}" 
                            to "{msg.content}". 
                            Provide encouraging feedback and analyze fluency, lexical resource, and grammatical range.
                         """
                    response = await asyncio.to_thread(llm_groq.invoke, feedback_prompt)
                    fluency, lexical, grammar = analyse_text(msg.content)

                    await cl.Message(
                        content=(
                            f"✅ Correct!\n\n{response.content}\n\n"
                            f"### IELTS Scoring Simulation\n"
                            f"**Fluency & Coherence:** {fluency}\n"
                            f"**Lexical Resource:** {lexical}\n"
                            f"**Grammatical Range & Accuracy:** {grammar}"
                        )
                    ).send()
                else:
                    # Incorrect response - provide constructive feedback
                    feedback_prompt = f"""
                            The user attempted to correct the sentence: 
                            "{previous_row['Original_Sentence']}" 
                            but provided "{msg.content}". 
                            The correct sentence is: "{previous_row['Corrected_Sentence']}". 
                            Provide feedback on why the correction is incorrect and analyze fluency, lexical resource, and grammatical range.
                        """
                    response = await asyncio.to_thread(llm_groq.invoke, feedback_prompt)
                    fluency, lexical, grammar = analyse_text(msg.content)

                    await cl.Message(
                        content=(
                            f"❌ Incorrect! The correct sentence is: {previous_row['Corrected_Sentence']}\n\n"
                            f"{response.content}\n\n"
                            f"### IELTS Scoring Simulation\n"
                            f"**Fluency & Coherence:** {fluency}\n"
                            f"**Lexical Resource:** {lexical}\n"
                            f"**Grammatical Range & Accuracy:** {grammar}"
                        )
                    ).send()
            except Exception as e:
                await cl.Message(content=f"An error occurred during analysis: {str(e)}").send()

        # If there are more questions, send the next one
        if current_index < len(df):
            row = df.iloc[current_index]

            # Provide a hint for the next question
            hint_prompt = f"""
                    The following sentence contains a mistake: 
                    "{row['Original_Sentence']}" 
                    categorized as a {row['Mistake_Type']} mistake. 
                    Provide a brief hint to help the user identify and correct it. Do not
                    provide the correct answer. Tell the user to enter next for the next question.
                """
            hint = await asyncio.to_thread(llm_groq.invoke, hint_prompt)

            # Send the next question to the user
            await cl.Message(
                content=f"Correct the Original Sentence: {row['Original_Sentence']}\n(Mistake Type: {row['Mistake_Type']})\nHint: {hint.content}"
            ).send()

            # Update the session index
            cl.user_session.set("current_index", current_index + 1)
        else:
            # If all questions have been completed, end the session
            await cl.Message(content="🎉 Practice completed! Great job!").send()



    elif modal_100 == "Test Mode":

        # Test Mode Logic

        df = cl.user_session.get('data')

        current_index = cl.user_session.get("current_index", 0)

        # Initialize scoring variables if they don't exist

        correct_answers = cl.user_session.get("correct_answers", 0)

        total_questions = len(df)

        # Check if there is any data

        if df is None or len(df) == 0:
            await cl.Message(
                content="⚠️ No questions are available for the test. Please check the data and try again!").send()

            return

        # Handle the start of the test

        if current_index == 0:

            # Send the first question immediately

            row = df.iloc[current_index]

            await cl.Message(content="Welcome to the IELTS Test Mode! Let's begin. Respond with a corrected sentence").send()

            await cl.Message(

                content=f"Correct the Original Sentence: {row['Original_Sentence']}\n(Mistake Type: {row['Mistake_Type']})"

            ).send()

            # Update the session index

            cl.user_session.set("current_index", current_index + 1)

        else:

            # Compare the user's response to the previous sentence

            previous_row = df.iloc[current_index - 1]

            if msg.content.strip() == previous_row['Corrected_Sentence']:

                correct_answers += 1  # Increment correct answers

                cl.user_session.set("correct_answers", correct_answers)

                feedback_prompt = f"""

                        The user correctly corrected the sentence: 

                        "{previous_row['Original_Sentence']}" 

                        to "{msg.content}". 

                        Please provide encouraging feedback and explain why the correction is correct.

                    """

                response = await asyncio.to_thread(llm_groq.invoke, feedback_prompt)

                await cl.Message(content=f"✅ Correct!\n\n{response.content}").send()

            else:

                feedback_prompt = f"""

                        The user attempted to correct the sentence: 

                        "{previous_row['Original_Sentence']}" 

                        but provided "{msg.content}". 

                        The correct sentence is: "{previous_row['Corrected_Sentence']}". 

                        Please explain the mistake and provide detailed feedback on the correction.

                    """

                response = await asyncio.to_thread(llm_groq.invoke, feedback_prompt)

                await cl.Message(
                    content=f"❌ Incorrect! The correct sentence is: {previous_row['Corrected_Sentence']}").send()

            # Move to the next question or end the test

            if current_index < len(df):

                row = df.iloc[current_index]

                await cl.Message(

                    content=f"Correct the original Sentence: {row['Original_Sentence']}\n(Mistake Type: {row['Mistake_Type']})"

                ).send()

                cl.user_session.set("current_index", current_index + 1)

            else:

                # Test completed, calculate score

                score_percentage = (correct_answers / total_questions) * 100

                await cl.Message(

                    content=f"🎉 Test completed! Great job!\n\n**Your Score:** {correct_answers}/{total_questions} "

                            f"({score_percentage:.2f}%).\n\nThank you for taking the IELTS test!"

                ).send()

# Helper Functions for IELTS Scoring

def analyse_text(text):
    words = word_tokenize(text)
    word_count = len(words)

    # Fluency & Coherence (simplified timing analysis)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    fluency_score = "Good timing and logical flow." if sentence_count > 0 and word_count / sentence_count > 12 else "Consider improving sentence flow."

    # Lexical Resource
    unique_words = set(words)
    lexical_diversity = len(unique_words) / word_count
    lexical_score = "Great word choice." if lexical_diversity > 0.7 else "Use a wider variety of words."

    # Grammatical Range & Accuracy
    grammar_errors = TextBlob(text).correct()
    grammar_feedback = "Grammar looks good." if text == str(grammar_errors) else f"Consider: {str(grammar_errors)}"

    return fluency_score, lexical_score, grammar_feedback

def analyse_pronunciation(audio_file):
    model_path = "vosk-model-en-us-0.22"  # Replace with actual model path
    model = Model(model_path)
    audio = AudioSegment.from_wav(audio_file)
    recognizer = KaldiRecognizer(model, audio.frame_rate)

    recognizer.AcceptWaveform(audio.raw_data)
    result = recognizer.Result()

    return "Pronunciation feedback is under development. Stay tuned!"

@cl.on_stop
def on_stop():
    print("The user stopped the IELTS practice session.")
