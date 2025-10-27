# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings

# def build_faiss_index():
#     # Load all HTML documents
#     loader = DirectoryLoader(path="hr-policies", glob="**/*.html")
#     documents = loader.load()
#     print(f"{len(documents)} Pages Loaded")

#     # Split documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
#     )
#     split_documents = text_splitter.split_documents(documents)
#     print(f"Split into {len(split_documents)} Documents...")
#     print(split_documents[0].metadata)

#     # Create embeddings for all chunks
#     embeddings = OpenAIEmbeddings()

#     # Build FAISS index in memory (not saved locally)
#     db = FAISS.from_documents(split_documents, embeddings)
#     print("✅ FAISS index built and stored in memory.")
#     return db


# def faiss_query():
#     # Build in-memory FAISS index
#     db = build_faiss_index()

#     # Query directly without saving/loading from local storage
#     query = "Explain the Candidate Onboarding Process."
#     docs = db.similarity_search(query)

#     for doc in docs:
#         print("##---- Page ----##")
#         print(doc.metadata.get("source", "No source info"))
#         print("##---- Content ----##")
#         print(doc.page_content)


# if __name__ == "__main__":
#     faiss_query()
