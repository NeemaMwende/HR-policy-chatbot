from langchain.document_loaders import DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def upload_html():
    loader = DirectoryLoader(path="hr-policies")
    documents = loader.load() 
    print(f"{len(documents)} Pages Loaded")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["", "", " ", ""]
    )
    
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...") 
    
    print(split_documents[0].metadata)
    
    embeddings = OpenAIEmbeddings() 
    db = FAISS.from_documents(split_documents, embeddings) 
    db.save_local("faiss_index")
     
# def faiss_query():
#     db = FAISS.load_local("faiss_index", embeddings)
