import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# 1. 환경 설정 및 DB 로드
load_dotenv()
st.set_page_config(page_title="사내 규정 AI 비서", page_icon="🤖")

DB_PATH = "./chroma_db"

@st.cache_resource
def load_rag_chain():
    # 저장된 DB 불러오기
    if not os.path.exists(DB_PATH):
        return None
        
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()
    
    # LLM (GPT-4o) 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # 온도를 0으로 해야 사실 기반 답변

    # 프롬프트 설계 (페르소나 부여)
    system_prompt = (
        "당신은 회사의 유능한 규정 담당 AI 비서입니다. "
        "아래 제공된 [Context]만을 근거로 답변하세요. "
        "만약 문서에 없는 내용이라면 '죄송합니다. 관련 규정을 찾을 수 없습니다.'라고 정직하게 말하세요. "
        "답변 끝에는 반드시 참고한 근거 문서를 명시하세요."
        "\n\n"
        "[Context]:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 체인 연결: 검색(Retriever) -> 답변생성(LLM)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# UI 구성
st.title("🤖 사내 규정/계약서 AI 검토 비서")
st.caption("궁금한 규정을 물어보세요. PDF 문서를 기반으로 답변해드립니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if question := st.chat_input("질문을 입력하세요 (예: 경조사 휴가는 며칠인가요?)"):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # AI 답변 생성
    with st.chat_message("assistant"):
        rag_chain = load_rag_chain()
        
        if rag_chain is None:
            st.error("⚠️ 학습된 데이터가 없습니다. 먼저 `ingest.py`를 실행해서 PDF를 학습시켜주세요.")
        else:
            with st.spinner("규정집을 검색하고 있습니다..."):
                response = rag_chain.invoke({"input": question})
                answer = response["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # (선택) 근거 문서 디버깅용 표시
                with st.expander("참고한 문서 조각 보기"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**[문서 {i+1}] {doc.metadata.get('source', 'Unknown')}**")
                        st.text(doc.page_content[:200] + "...")
