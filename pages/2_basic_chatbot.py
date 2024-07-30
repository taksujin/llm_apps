# Chat Model 추가하여 AI응답 만들어냄
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# 추가
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

st.title("AI응답 챗봇")

# LLM 모델 생성
llm = ChatOpenAI(model_name ='gpt-4o-mini', temperature=0.5)

# chat history
history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

query = st.chat_input("하고 싶은 말")

if query:
    with st.chat_message("user"):
        history.add_user_message(query)
        st.markdown(query)

    with st.chat_message("assistant"):
        messages = [HumanMessage(content=query)] # 사용자 입력으로 대화내용을 만들고
        response = llm.invoke(messages) # 그것을 바탕으로 AI응답을 얻어냄.
        history.add_ai_message(response)
        st.markdown(response.content)   # response.content로 변경

# 문제점: '2024년 국민의힘 대표 선거 결과를 알려줘' -> '죄송하지만, 2023년 10월까지의 정보만 가지고 있으며 ...'