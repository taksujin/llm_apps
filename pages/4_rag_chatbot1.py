# RAG Q&A 앱 (RetrievalQA로 구현, 기억력 없음)
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
#추가
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, JSONLoader, UnstructuredMarkdownLoader, PyMuPDFLoader)
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
import chromadb, faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

st.title("RAG Q&A 앱")

embedding_model = OpenAIEmbeddings()
# FAISS vector store 관련
vs = FAISS(
    embedding_function=embedding_model,
    index=faiss.IndexFlatL2(1536),
    docstore=InMemoryDocstore(), index_to_docstore_id={})

def vs_add_file(file_path):
    if file_path.endswith('.txt'):
        text_loader = TextLoader(file_path)
        raw_doc = text_loader.load()
    elif file_path.endswith('.md'):
        markdown_loader = UnstructuredMarkdownLoader(file_path)
        raw_doc = markdown_loader.load()
    elif file_path.endswith('.pdf'):
        pdf_loader = PyMuPDFLoader(file_path)
        raw_doc = pdf_loader.load()
    elif file_path.endswith('.json'):
        json_loader = JSONLoader(file_path)
        raw_doc = json_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap = 200)
    docs = text_splitter.split_documents(raw_doc)

    vs.add_documents(docs)

def save_file(file):
    import os
    folder = 'tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

uploaded_files = st.file_uploader("Choose a data file", accept_multiple_files=True)
for file in uploaded_files:
    file_path = save_file(file)
    vs_add_file(file_path)

# retriever
def create_retriever():
    gpt4o = ChatOpenAI(model_name ='gpt-4o-mini', temperature=0.5)

    retriever = RetrievalQA.from_chain_type(
        llm=gpt4o,
        chain_type="stuff",
        retriever=vs.as_retriever(
            search_type='mmr',
            search_kwargs={'k':8, 'fetch_k':12}
        ),
        return_source_documents=True
    )
    return retriever

# 쿼리 및 응답 처리
query = st.chat_input("하고 싶은 말")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())

        retriever = create_retriever()
        response = retriever.invoke(
            {"query": query},
            {"callbacks": [callback]},
        )
        st.markdown(response["result"])

        # to show references
        for idx, doc in enumerate(response['source_documents'],1):
            filename = os.path.basename(doc.metadata['source'])
            ref_title = f":blue[Reference {idx}: *{filename}*]"
            with st.popover(ref_title):
                st.caption(doc.page_content)