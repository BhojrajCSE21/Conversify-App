from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
import chromadb
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
    texts = text_splitter.split_documents(documents)
    # Creating Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
    
    # Creating Vector Store
    client = chromadb.PersistentClient(path="db_metadata_v5")
    vector_db = Chroma.from_documents(
        client=client, 
        documents=texts, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    vector_db.persist()
    vector_db=None


if __name__ == "__main__":
    main()
    