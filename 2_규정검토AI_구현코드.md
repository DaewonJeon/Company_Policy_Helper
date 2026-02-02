# [프로젝트 2] 사내 규정/계약서 검토 AI 비서: 구현 코드

> **목적**: Python, LangChain, OpenAI를 활용하여 내 컴퓨터에서 돌아가는 **사내 규정 Q&A 챗봇**을 만듭니다.
> **구조**: 데이터 주입기(`ingest.py`)와 채팅 인터페이스(`app.py`)로 나뉩니다.

---

## 1. 프로젝트 폴더 구조 (Directory Structure)

바탕화면에 `ai_secretary` 폴더를 만들고 아래와 같이 구성하세요.

```text
ai_secretary/
├── .env                  # API Key 저장소
├── requirements.txt      # 필수 라이브러리 목록
├── ingest.py             # 규정집 PDF를 읽어서 DB에 저장하는 코드 (학습용)
├── app.py                # 챗봇 화면 실행 코드 (서비스용)
└── data/                 # PDF 파일을 넣을 폴더
    └── (여기에 PDF 파일들을 넣으세요)
```

---

## 2. 필수 라이브러리 설치 (requirements.txt)

최신 AI 라이브러리들을 사용합니다.

```text
langchain
langchain-community
langchain-openai
langchain-chroma
streamlit
pypdf
python-dotenv
openai
chromadb
tiktoken
```

---

## 3. 환경 변수 설정 (.env)

`openai.com`에서 발급받은 키를 입력하세요. (카드 등록 필요, 사용량만큼 과금되나 테스트 시 몇백 원 수준)

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 4. 데이터 학습기 구현 (ingest.py)

이 코드는 `data` 폴더에 있는 PDF 파일들을 읽어서, AI가 검색할 수 있도록 **벡터 DB(Chroma)**에 저장합니다.
**최초 1회** 실행하거나, PDF가 바뀔 때마다 실행하면 됩니다.

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 환경 설정 로드
load_dotenv()

# 데이터 경로 설정
DATA_PATH = "./data"
DB_PATH = "./chroma_db"  # 벡터 DB가 저장될 폴더

def ingest_docs():
    print("🔄 문서 학습을 시작합니다...")
    
    # 2. PDF 파일 로드
    documents = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print("⚠️ data 폴더가 없어 생성했습니다. PDF 파일을 넣어주세요.")
        return

    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not files:
        print("⚠️ data 폴더에 PDF 파일이 없습니다.")
        return

    for file in files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())
        print(f"   - {file} 로드 완료 ({len(documents)} 페이지)")

    # 3. 텍스트 분할 (Chunking)
    # 문맥 유지를 위해 1000자 단위로 자르고, 200자는 겹치게(overlap) 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"📊 총 {len(splits)}개의 지식 조각으로 분해되었습니다.")

    # 4. 벡터 DB 저장 (Embeddings)
    # OpenAI의 임베딩 모델(text-embedding-3-small) 사용 - 저렴하고 성능 좋음
    print("💾 벡터 DB에 저장 중... (시간이 조금 걸릴 수 있습니다)")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=DB_PATH
    )
    
    print("✅ 학습 완료! 'chroma_db' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    ingest_docs()
```

---

## 5. 챗봇 UI 구현 (app.py)

실제 사용자가 질문을 던지는 화면입니다. `ingest.py`가 만들어둔 DB를 조회합니다.

```python
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
```
