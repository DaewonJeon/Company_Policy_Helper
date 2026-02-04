import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. 환경 설정 및 DB 로드
load_dotenv()
st.set_page_config(page_title="사내 규정 AI 비서", page_icon="🤖")

DB_PATH = "./chroma_db"

# 모델 정보 정의
MODEL_OPTIONS = {
    "GPT-4o (OpenAI)": {
        "provider": "openai",
        "model": "gpt-4o",
        "description": "OpenAI 최신 멀티모달 모델, 빠르고 정확함"
    },
    "GPT-4o-mini (OpenAI)": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "description": "GPT-4o 경량 버전, 더 저렴하고 빠름"
    },
    "Gemini 2.5 Flash (Google)": {
        "provider": "google",
        "model": "models/gemini-2.5-flash",
        "description": "빠른 응답, 비용 효율적"
    },
    "Gemini 2.5 Pro (Google)": {
        "provider": "google",
        "model": "models/gemini-2.5-pro",
        "description": "복잡한 추론 및 코딩 작업에 적합 (무료 티어 가능)"
    },
    "Gemini 3 Flash Preview (Google)": {
        "provider": "google",
        "model": "models/gemini-3-flash-preview",
        "description": "최신 모델, 향상된 멀티모달 기능"
    },
}


def get_llm(model_name: str):
    """선택된 모델에 맞는 LLM 객체 반환"""
    model_info = MODEL_OPTIONS[model_name]
    
    if model_info["provider"] == "openai":
        return ChatOpenAI(model=model_info["model"], temperature=0)
    elif model_info["provider"] == "google":
        return ChatGoogleGenerativeAI(model=model_info["model"], temperature=0)


def load_vectorstore():
    """벡터 DB 로드"""
    if not os.path.exists(DB_PATH):
        return None
    
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )


def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(llm, retriever):
    """LCEL 방식으로 RAG 체인 생성"""
    # 프롬프트 설계
    system_prompt = """당신은 회사의 유능한 문서 검토 AI 비서입니다.
아래 제공된 [Context]를 꼼꼼히 읽고, 사용자의 질문에 정확하게 답변하세요.
Context에 관련 내용이 있다면 반드시 그 내용을 바탕으로 상세히 답변하세요.
Context에 관련 내용이 전혀 없는 경우에만 '죄송합니다. 관련 내용을 찾을 수 없습니다.'라고 말하세요.
답변 끝에는 참고한 문서 출처를 명시하세요.

[Context]:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # LCEL 체인 구성
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# ==================== UI 구성 ====================

st.title("🤖 사내 규정/계약서 AI 검토 비서")
st.caption("궁금한 규정을 물어보세요. PDF 문서를 기반으로 답변해드립니다.")

# 사이드바: 모델 선택
with st.sidebar:
    st.header("⚙️ 설정")
    
    selected_model = st.selectbox(
        "AI 모델 선택",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="답변을 생성할 AI 모델을 선택하세요"
    )
    
    # 선택된 모델 정보 표시
    model_info = MODEL_OPTIONS[selected_model]
    st.caption(f"📝 {model_info['description']}")
    
    # Google 모델 선택 시 API 키 확인 안내
    if model_info["provider"] == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            st.warning("⚠️ GOOGLE_API_KEY가 .env에 설정되지 않았습니다.")
    
    st.divider()
    
    # 채팅 기록 초기화 버튼
    if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 세션 상태 초기화
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
        vectorstore = load_vectorstore()
        
        if vectorstore is None:
            st.error("⚠️ 학습된 데이터가 없습니다. 먼저 `ingest.py`를 실행해서 PDF를 학습시켜주세요.")
        else:
            try:
                with st.spinner(f"📚 {selected_model}로 검색 중..."):
                    # 선택된 모델로 LLM 생성
                    llm = get_llm(selected_model)
                    retriever = vectorstore.as_retriever()
                    rag_chain = create_rag_chain(llm, retriever)
                    
                    # RAG 체인 실행
                    answer = rag_chain.invoke(question)
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # 근거 문서 표시
                    with st.expander("참고한 문서 조각 보기"):
                        docs = retriever.invoke(question)
                        for i, doc in enumerate(docs):
                            st.markdown(f"**[문서 {i+1}] {doc.metadata.get('source', 'Unknown')}**")
                            st.text(doc.page_content[:200] + "...")
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.info("💡 API 키가 올바르게 설정되었는지 확인하세요.")
