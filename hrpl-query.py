from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrival_chain, create_history_aware_retriver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage 
from langchain.chains.combine_documents import create_stuff_documents_chain 

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def build_chat_history(chat_history_list):
    chat_history = [] 
    for message in chat_history_list:
        chat_history_list.append(HumanMessage(content=message[0]))
        chat_history_list.append(AIMessage(content=message[1]))
    return chat_history

    chat_history = build_chat_history(chat_history_list)
    embeddings = OpenAIEmbeddings() 
    new_db = FAISS.load_local("Invoices/faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    condense_question_system_template = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood," 
        "without the chat history, DO NOT answer the question,"
        "just reformulate it if needed and otherwise return it"
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [ 
            ("system", condense_question_system_template), 
            ("placeholder", "{chat_history}"), 
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        lm, new_db.as_retriever(), condense_question_prompt
    )
    system_prompt = ( 
        "You are an assistant for question-answering tasks on HR Policy."
        "Use th following pieces of retrieved context to answer the question" 
        "If you dont know the answer, say that you dont know."
        "Use three sentences max and keep the answers concise."
        ""
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

    convo_qa_chain = create_retrieval_chain(history_aware_retrival, qa_chain) 
    
    return convo_qa_chain.invoke( 
        {
            "chat_history": chat_history,
        }
    )
    
def show_ui(): 
    st.title("Yours Truly       Human Resources Chatbot")
    st.image("yk-Chatbot/c4x-cbt.png")
    st.subheader("Please enter your HR Query")
    
    if "messages" not in st.session_state: 
        st.session_state.messages = []
        st.session_state.chat_history = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if prompt := st.chat_input("Enter your HR Policy related Query: ")
        with st.spinner("Working on your query...")
            response = query(question=prompt, chat_history=st.session_state.chat_history) 
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assstant"):
                    st.markdown(response["answer"])
        
        #append user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        st.session_state.chat_history.extend([(prompt, response["answer"])])
        
if __name__ == "__main__":
    show_ui()       
        
        
        