import streamlit as st

st.set_page_config(
    page_title="Langchain Streamlit App Examples",
    page_icon='💬',
    layout='wide'
)

st.header("Chatbot Implementations with Langchain + Streamlit")
st.write("""
[![view source code ](https://img.shields.io/badge/GitHub%20Repository-gray?logo=github)](https://github.com/akapo/llm-app)
""")
st.write("""
Langchain은 LLM(언어 모델)을 사용하여 애플리케이션 개발을 간소화하도록 설계된 강력한 프레임워크입니다. 다양한 구성 요소의 포괄적인 통합을 제공하여 강력한 응용 프로그램을 만들기 위해 조립 프로세스를 단순화합니다.

Langchain의 힘을 활용하면 챗봇 생성이 쉬워집니다. 다음은 다양한 사용 사례에 맞는 챗봇 구현의 몇 가지 예입니다.

- **💬translato**: 번역 서비스 앱(다양한 언어 지원).
- **online_chatbot**: 인터넷에 접속하여 정보를 검색하는 챗봇 구현.
- **💽memorye chatbot**: 컨텍스트를 유지하며 기억력을 가지는 챗봇 구현.
- **📄rag_chatbot**: 입력해준 문서를 기반으로 답변을 생성하는 챗봇 구현.
- **⭐coteacher**: 프로그래밍 인공지능 보조교사 구현.
- **🎓Knowlegebase**: 인공지능 보조교사에게 지식 주입.

각 챗봇의 샘플 사용법을 살펴보려면 해당 챗봇 섹션으로 이동하세요.""")
     