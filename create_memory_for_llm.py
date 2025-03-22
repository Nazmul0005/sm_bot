from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. load raw pdf
DATA_PATH="data/"
def  load_pdf_files(data):
    loader=DirectoryLoader(data,
    glob="*.pdf", loader_cls=PyPDFLoader)

    docutments=loader.load()
    return docutments


documents=load_pdf_files(data=DATA_PATH)

#print("Length of PDF Pages:", len(documents))






# 2. create chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,

    )

    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Chunks:", len(text_chunks))


# 3. create vector embeddings


def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embedding_model

embedding_model=get_embedding_model()
# 4. store embeddings in Faiss

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)