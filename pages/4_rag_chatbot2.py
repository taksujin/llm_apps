# RAG 챗봇 (create_history_aware_retriever 구현)
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
st.set_page_config(page_title="RAG 챗봇", page_icon="📄", layout='wide')
st.header('RAG 챗봇')

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
        chunk_size = 500, chunk_overlap  = 100)
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

# chat history
if('app_name' not in st.session_state):
    st.session_state.app_name = 'rag_chatbot'
elif(st.session_state.app_name != 'rag_chatbot'):
    st.session_state.app_name = 'rag_chatbot'
    StreamlitChatMessageHistory().clear();

history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

# create_retriever_chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_retriever_chain(history):
    # LangChain의 create_history_aware_retriever를 사용해,
    # 과거의 대화 기록을 고려해 질문을 다시 표현하는 Chain을 생성
    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "위의 대화에서, 대화와 관련된 정보를 찾기 위한 검색 쿼리를 생성해 주세요."),
        ]
    )
    rephrase_llm = ChatOpenAI(model_name ='gpt-4o-mini', temperature=0.5)

    retriever=vs.as_retriever(
            search_type='mmr',
            search_kwargs={'k':8, 'fetch_k':12}
    )

    rephrase_chain = create_history_aware_retriever(
        rephrase_llm, retriever, rephrase_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "아래의 문맥만을 고려하여 질문에 답하세요.\n\n{context}"),
            (MessagesPlaceholder(variable_name="chat_history")),
            ("user", "{input}"),
        ]
    )

    callback = StreamlitCallbackHandler(st.container())

    qa_llm = ChatOpenAI(
        model_name ='gpt-4o',
        temperature=0.5,
        streaming=True,
        callbacks=[callback]
    )

    qa_chain = qa_prompt | qa_llm | StrOutputParser()

    # 두 Chain을 연결한 Chain을 생성
    conversational_retrieval_chain = (
        RunnablePassthrough.assign(context=rephrase_chain | format_docs) | qa_chain
        )

    return conversational_retrieval_chain

# 쿼리 및 응답
query = st.chat_input("하고 싶은 말")

if query:
    with st.chat_message("user"):
        history.add_user_message(query)
        st.markdown(query)

    with st.chat_message("assistant"):
        #callback = StreamlitCallbackHandler(st.container())

        retriever = create_retriever_chain(history)
        response = retriever.invoke(
            {"input": query, "chat_history": history.messages},
            #{"callbacks": [callback]},
        )
        history.add_ai_message(response)
        st.markdown(response)