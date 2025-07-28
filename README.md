# ModularRAG-with-HyperClovaX

- [Naver GreenDevelopers] Bogosa 후속 프로젝트(Modular 기반 HyperClovaX 활용 RAG 시스템 구축)
- [Bogosa 프로젝트](https://github.com/DS3th-AIFFEELTHON/Bogosa)

<br>
<br>


## DataSet
### KIS Weekly Report
- 🔗 [KIS자산평가/자료실/Weekly](https://www.bond.co.kr/post/10106/10254#none;)
- 발행 주관 : KIS 자산평가
- 발행 주기 : week
- 활용 범위 : 제1090호 - 제1120호 (2024년 6월 28일 - 2025년 1월 24일)
- 특징 : Chart, Table, Text가 혼합된 Multi Data 형태

<br>
<br>

## 개요
### 1. Advanced RAG 기반 시스템 -> Modular RAG 기반 시스템으로 변경
- 시스템 속도 향상
- 추후 모듈 활용 시 용이성 증대
#### ➡️ 1분 30초 대 실행 시간 -> 20초대 실행 시간으로 단축

<br>

### 2. HyperClova X 기반 LLM Agent 구축
-  한국어 인식 능력이 우수한 HyperClova X HCX-005 모델 기반 시스템 구축
-  한국어 인식 프롬프트 엔지니어링 수행
#### ➡️ 한국어 자연어 처리 능력 향상 확인

<br>
<br>


## 기술 스택
> ### Language
- Python `3.11`

<br>

> ### Format
- JSON
- JSONLines
- CSV

<br>

> ### API & Cloud
- HyperClova X

<br>

> ### RAG
- LangChain `0.3.19`
- LangGraph `0.2.74`

<br>

> ### VectorDB & Embedding
- Milvus
  - langchain_milvus
  - langchain_community.vectorstores
- Embeddings
  - HyperClova X Embedding `v2`

<br>

> ### Library
- LangChain
  - community `0.3.18`
  - openai `0.2.14`
  - core `0.3.40`
  - experimental `0.3.4`
  - milvus `0.1.8`

<br>
<br>


## 프로젝트 수행
### 1. Architecture
- 기존 Advanced-RAG 아키텍처를 LangGraph 기반 Modular RAG 아키텍처로 변경

<br>

### 2. Prompt Engineering
- HyperClova X 최적화 프롬프트 엔지니어링 수행
- 기존 시간 모듈 등 최적화 수행
<br>

### 3. 평가 방법
- Bogosa 프로젝트 테스트용 질문 활용한 정성평가 수행
<br>


## RAG Architecture
![Image](https://github.com/user-attachments/assets/e99f2929-f148-487e-98d9-2cefd858c419)

<br>
<br>

## 참고 문서 및 코드 참고

- 논문
  - [What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers](https://arxiv.org/abs/2109.04650)

