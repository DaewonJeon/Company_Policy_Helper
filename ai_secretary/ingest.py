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
