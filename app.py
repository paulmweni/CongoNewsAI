"""Chainlit web chat interface for the News AI assistant.

Launches a browser-based chat UI backed by the :class:`~src.query.NewsQA`
retrieval-augmented pipeline. Each session keeps its own QA instance.

Run with:
    chainlit run app.py -w
"""

import chainlit as cl
from src.query import NewsQA

@cl.on_chat_start
async def start_chat():
    # Initialize your QA system here.
    # The CivilLawQA object is stored in the user session for later use.
    qa_system = NewsQA()
    cl.user_session.set("qa_system", qa_system)

    await cl.Message(
        content="Hello! I am an AI assistant specialized in Congolese News and Journalism. How can I help you today?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Retrieve the QA system from the user session.
    qa_system = cl.user_session.get("qa_system")

    # Get the answer from your existing logic.
    response = qa_system.answer_query(message.content)

    # Send the response back to the user.
    await cl.Message(content=response).send()