# 외부 정보 검색 기능 추가
# 원달러 환율을 알려줄 수 있음
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
# 추가
from langchain import hub
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools

# 외부 검색 가능한 도구를 추가한 AgentExcutor 생성
def create_agent_chain():
    llm = ChatOpenAI(model_name ='gpt-4o', temperature=0.5)

    tools = load_tools(["ddg-search", "wikipedia"])    # tools 정의
    prompt = hub.pull("hwchase17/openai-tools-agent")  # tools-agent 프롬프트 로드
    agent = create_openai_tools_agent(llm, tools, prompt) # agent 생성

    return AgentExecutor(agent=agent, tools=tools) # AgentExecutor 리턴

st.set_page_config(page_title="온라인 챗봇", page_icon="🌐", layout='wide')
st.header('온라인 챗봇')

history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

query = st.chat_input("하고 싶은 말")

if query:
    with st.chat_message("user"):
        history.add_user_message(query)
        st.markdown(query)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain()
        response = agent_chain.invoke(  # agent_chain이 응답을 반환할 때 [callback]이 호출되면서 AI의 응답이 자동으로 출력됨.
            {"input": query},
            {"callbacks": [callback]},
        )
        #messages = [HumanMessage(content=query)]  # 삭제
        #response = llm.invoke(messages)            # 삭제
        history.add_ai_message(response["output"])
        st.markdown(response["output"])  # agent_chain의 응답이므로 변경

# 문제점: 기억이 없음. 내 이름을 알려줘도 모름. 1 to 50 게임도 못함.
     