# ragengine.py

# === Imports ===
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os


# === Function: Build Chroma Vectorstore from local documents ===
def build_chroma_vectorstore(folder_path="hr_docs"):
    """
    Loads HR documents, splits them into chunks, and stores embeddings in ChromaDB.
    """

    # Load all text/HTML documents from the folder
    loader = DirectoryLoader(folder_path, glob="**/*.html")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} HR documents")

    # Split documents into manageable chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"📄 Split into {len(split_docs)} smaller chunks")

    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings()

    # Store vectors in Chroma (local persistent DB)
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="chroma_db"  # folder where vectors are saved
    )

    vectorstore.persist()
    print("💾 Chroma vectorstore created and saved at ./chroma_db")
    return vectorstore


# === Function: Create a RAG chain ===
def create_rag_chain():
    """
    Loads Chroma vectorstore and builds a retrieval chain (RAG pipeline).
    """

    # Load embeddings and Chroma database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize ChatOpenAI model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # Define the system prompt
    system_prompt = (
        "You are an AI HR assistant that answers employee questions about HR policies. "
        "Use the following retrieved context to answer the question clearly and concisely. "
        "If unsure, say you don't know.\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Build retrieval chain (RAG)
    rag_chain = create_retrieval_chain(retriever, llm, prompt)

    return rag_chain


# === Optional: Run once to build index manually ===
if __name__ == "__main__":
    if not os.path.exists("chroma_db"):
        build_chroma_vectorstore()
    else:
        print("Chroma vectorstore already exists. Skipping rebuild.")
