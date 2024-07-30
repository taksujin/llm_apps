# ConversationBufferMemory로 기억력 추가
# 1 to 50 게임 가능
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain import hub
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_openai_tools_agent
# 추가
from langchain.memory import ConversationBufferMemory

# 외부 검색 가능한 도구를 추가한 AgentExcutor 생성
def create_agent_chain(history): # history를 파라미터로 받음
    llm = ChatOpenAI(model_name ='gpt-4o', temperature=0.5)

    tools = load_tools(["ddg-search", "wikipedia"])
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)

    # 기억을 위해 ConversationBufferMemory 생성
    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True)

    return AgentExecutor(agent=agent, tools=tools, memory=memory)  # memory 추가

st.set_page_config(page_title="기억력 챗봇", page_icon="💽", layout='wide')
st.header('기억력 있는 온라인 챗봇')

history = StreamlitChatMessageHistory()

for message in history.messages:
    st.chat_message(message.type).write(message.content)

query = st.chat_input("하고 싶은 말")

if query:
    with st.chat_message("user"):
        #history.add_user_message(query)  # 삭제
        st.markdown(query)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain(history) # history를 파라미터로 패싱
        response = agent_chain.invoke(
            {"input": query},
            {"callbacks": [callback]},
        )
        #history.add_ai_message(response["output"]) # 삭제
        st.markdown(response["output"])  # agent_chain의 응답이므로 변경
