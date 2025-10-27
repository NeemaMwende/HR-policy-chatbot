from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains.retrieval import create_retrieval_chain, create_history_aware_retriever
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def build_chat_history(chat_history_list):
    chat_history = []
    for message in chat_history_list:
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))
    return chat_history


def query(question, chat_history_list):
    chat_history = build_chat_history(chat_history_list)
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("Invoices/faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Prompt for reformulating user question based on chat history
    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. DO NOT answer the question, "
        "just reformulate it if needed and otherwise return it."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, new_db.as_retriever(), condense_question_prompt
    )

    # System prompt for the QA task
    system_prompt = (
        "You are an assistant for question-answering tasks on HR Policy. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences max and keep the answers concise.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain.invoke(
        {
            "chat_history": chat_history,
            "input": question
        }
    )


def show_ui():
    st.title("Yours Truly — Human Resources Chatbot")
    st.image("yk-Chatbot/c4x-cbt.png")
    st.subheader("Please enter your HR Query")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your HR Policy related Query:"):
        with st.spinner("Working on your query..."):
            response = query(question=prompt, chat_history_list=st.session_state.chat_history)

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.markdown(response["answer"])

        # Append messages to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        st.session_state.chat_history.append((prompt, response["answer"]))


if __name__ == "__main__":
    show_ui()
