from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def upload_html():
    loader = DirectoryLoader(path="hr-policies", glob="**/*.html")
    documents = loader.load()
    print(f"{len(documents)} Pages Loaded")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...") 
    
    print(split_documents[0].metadata)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("faiss_index")
     

def faiss_query():
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    query = "Explain the Candidate Onboarding Process."
    docs = new_db.similarity_search(query)
     
    for doc in docs:
        print("##---- Page ----##")
        print(doc.metadata["source"])
        print("##---- Content ----##")
        print(doc.page_content)
        

if __name__ == "__main__":
    #upload_html()
    faiss_query()
