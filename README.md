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
- 참고
  - [<랭체인LangChain 노트> - LangChain 한국어 튜토리얼🇰🇷](https://wikidocs.net/233341)
  - [이토록 쉬운 RAG 시스템 구축을 위한 랭체인 실전 가이드](https://www.yes24.com/product/goods/136548871)
  - [AutoRAG 튜토리얼](https://www.youtube.com/playlist?list=PLIMb_GuNnFwdjfLUPrpUAzjQLfBJLQ7MC)
  - [LoRA의 개념](https://www.youtube.com/watch?v=0lf3CUlUQtA)
  - [BigQuery RAG 파이프라인(Document AI Layout Parser)](https://cloud.google.com/blog/ko/products/data-analytics/bigquery-and-document-ai-layout-parser-for-document-preprocessing)

- Milvus
  - [milvus 메타데이터 필터링](https://milvus.io/docs/ko/filtered-search.md)
  - [Efficiently Deploying Milvus on GCP Kubernetes: A Guide to Open Source Database Management](https://medium.com/@zilliz_learn/efficiently-deploying-milvus-on-gcp-kubernetes-a-guide-to-open-source-database-management-7e49d0b194d8)
  - [Command line tool (kubectl)](https://kubernetes.io/docs/reference/kubectl/)
  - [Kubernetes CLI 도구인 kubectl의 사용법 이해하기](https://velog.io/@pinion7/kubernetes-CLI-%EB%8F%84%EA%B5%AC%EC%9D%B8-kubectl%EC%9D%98-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)
  - [GKE에 Milvus 클러스터 배포하기](https://milvus.io/docs/ko/gcp.md)

<br>

- 논문
  - [AutoRAG를 이용한 금융 문서에 가장 최적화된 RAG 시스템 구현에 관한 연구](https://koreascience.or.kr/article/CFKO202433162114304.pdf)
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)
  - [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/html/2401.18059v1)

<br>

- 공식문서
  - [AutoRAG 평가 지표](https://docs.auto-rag.com/evaluate_metrics/retrieval.html)

<br>

- 사례
  - [신한투자증권 X 스켈터랩스 :: 증권사 RAG 활용 사례](https://www.skelterlabs.com/blog/rag-securities)
  - [기업용 금융 특화 LLM 모델 만들기 (1)- 필요성과 RAG](https://blog-ko.allganize.ai/alli-finance-llm-1/)

<br>

- Model
  - [mteb_ko_leaderboard(오픈소스 임베딩 모델)](https://github.com/su-park/mteb_ko_leaderboard)
