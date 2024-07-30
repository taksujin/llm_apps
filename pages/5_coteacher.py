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
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

st.set_page_config(page_title="인공지능 보조교사", page_icon="⭐", layout='wide')
st.header('프로그래밍 인공지능 보조교사')

embedding_model = OpenAIEmbeddings()
# FAISS vector store 관련
faiss_dir = './faiss'
if(os.path.isdir(faiss_dir) == False):
    vs = FAISS(
            embedding_function=embedding_model, index=faiss.IndexFlatL2(1536),
            docstore=InMemoryDocstore(), index_to_docstore_id={})
else :
    vs = FAISS.load_local(faiss_dir, embedding_model,
            allow_dangerous_deserialization=True)

# chat history
if('app_name' not in st.session_state):
    st.session_state.app_name = 'coteacher'
elif(st.session_state.app_name != 'coteacher'):
    st.session_state.app_name = 'coteacher'
    StreamlitChatMessageHistory().clear();

history = StreamlitChatMessageHistory()

if len(history.messages) == 0:  # 대화내역이 전무하다면...
    hello = "안녕하세요? 무슨이야기를 해볼까요?"
    history.add_ai_message(hello)

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
            search_kwargs={'k':4, 'fetch_k':8}
    )

    rephrase_chain = create_history_aware_retriever(
        rephrase_llm, retriever, rephrase_prompt
    )

    coteacher_prompt = """
    You are an assistant teacher teaching programming desigend by 정진쌤.
    Please answer the student's questions appropriately.
    However, refuse requests to provide the correct answer code, to create a program, or to provide sample code.
    If requested, you can provide a psudo-code.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use the following context to answer the question at the end.
    Please answer in Korean unless otherwise requested.\n
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", coteacher_prompt),
            (MessagesPlaceholder(variable_name="chat_history")),
            ("user", "{input}"),
        ]
    )

    callback = StreamlitCallbackHandler(st.container())

    qa_llm = ChatOpenAI(
        model_name ='gpt-4o-mini',
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