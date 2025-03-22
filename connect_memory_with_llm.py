import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. setup llm mistral with HuggingFaceEmbeddings

HF_TOKEN = os.environ.get("HF_TOKEN")

huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=1,
            model_kwargs={"token": HF_TOKEN, "max_length":"1024"}

    )
    return llm

    



# 2. Connect llm with faiss and create chain

custom_prompt_template="""
Use the pieces of information provided in the context to answer User's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Start the answer directly. No smal talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )


# 3. load faiss and create chain

DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True)

#creae QA chain
qa_chain=RetrievalQA.from_chain_type(llm=load_llm(huggingface_repo_id=huggingface_repo_id),chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3}), return_source_documents=True,
chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)})

#Now invoke with a single query
user_query=input("Write Query Here:")
response=qa_chain({"query": user_query})
print("Result:", response["result"])
print("Sources:", response["source_documents"])