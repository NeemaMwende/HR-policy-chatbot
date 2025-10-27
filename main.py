# chat.py

import streamlit as st
from ragengine import build_chroma_vectorstore, create_rag_chain
import os

# === Streamlit UI setup ===
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="centered")

# Initialize database if it doesn't exist
if not os.path.exists("chroma_db"):
    with st.spinner("Building HR document knowledge base..."):
        build_chroma_vectorstore()

# Load the RAG chain
rag_chain = create_rag_chain()

# === Streamlit chat interface ===
st.markdown(
    """
    <style>
        .chat-container {
            background-color: #f4f4f9;
            border-radius: 16px;
            padding: 20px;
            max-width: 400px;
            margin: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .user-msg {
            background-color: #dcf8c6;
            border-radius: 16px;
            padding: 10px 14px;
            margin: 6px 0;
            text-align: right;
        }
        .bot-msg {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 10px 14px;
            margin: 6px 0;
            text-align: left;
            border: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 HR Assistant Chatbot")
st.write("Hi! 👋 I’m your HR assistant. Ask me anything about company HR policies.")

# Initialize chat session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# === Chat input ===
user_input = st.chat_input("Type your HR question here...")

if user_input:
    # Add user message
    st.session_state.history.append(("user", user_input))

    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input})
        answer = response.get("output", "Sorry, I couldn’t find that information.")

    # Add bot reply
    st.session_state.history.append(("bot", answer))

    st.rerun()

