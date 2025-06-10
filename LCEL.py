import os
import re
import json
import jsonlines
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_naver.embeddings import ClovaXEmbeddings
from langchain_milvus.vectorstores import Milvus
from uuid import uuid4
from langchain_naver.chat_models import ChatClovaX
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
import pytz

from datasets import Dataset
from datetime import timedelta
from operator import itemgetter
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import (
  AttributeInfo,
  StructuredQueryOutputParser,
  get_query_constructor_prompt
)
from langchain_teddynote.evaluator import GroundednessChecker
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import warnings
from langchain_core.runnables import chain

warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv()

embeddings = ClovaXEmbeddings(
    model='bge-m3'
)

from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import instructor
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class TimeFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class SearchQuery(BaseModel):
    query: str
    time_filter: TimeFilter

class Label(BaseModel):
    chunk_id: int = Field(description="The unique identifier of the text chunk")
    chain_of_thought: str = Field(
        description="The reasoning process used to evaluate the relevance"
    )
    relevancy: int = Field(
        description="Relevancy score from 0 to 10, where 10 is most relevant",
        ge=0,
        le=10,
    )

class RerankedResults(BaseModel):
    labels: list[Label] = Field(description="List of labeled and ranked chunks")

    @field_validator("labels")
    @classmethod
    def model_validate(cls, v: list[Label]) -> list[Label]:
        return sorted(v, key=lambda x: x.relevancy, reverse=True)
    
def adjust_time_filter_to_week(time_filter):
    """
    특정 날짜(YYYY-MM-DD)가 주어진 경우, 해당 날짜를 포함하는 주(월~일)의
    첫 번째 날(월요일)과 마지막 날(일요일)로 변환하는 함수.

    :param time_filter: dict, {"start_date": datetime, "end_date": datetime}
    :return: dict, {"start_date": datetime, "end_date": datetime}
    """
    # Extract start_date and end_date from time_filter
    start_date = time_filter.start_date
    end_date = time_filter.end_date

    # Handle the case where start_date or end_date is None
    if start_date is None or end_date is None:
        if start_date is not None and end_date is None:
            start_of_week = start_date - timedelta(days=start_date.weekday())  # 월요일 찾기
            end_of_week = start_of_week + timedelta(days=6)  # 해당 주 일요일 찾기

            return {
                "start_date": start_of_week.replace(hour=0, minute=0, second=0),
                "end_date": end_of_week.replace(hour=23, minute=59, second=59)
            }
        elif end_date is not None and start_date is None:
            start_of_week = end_date - timedelta(days=end_date.weekday())  # 월요일 찾기
            end_of_week = start_of_week + timedelta(days=6)  # 해당 주 일요일 찾기

            return {
                "start_date": start_of_week.replace(hour=0, minute=0, second=0),
                "end_date": end_of_week.replace(hour=23, minute=59, second=59)
            }
        else:
            return None  # or return the time_filter as is if you prefer

    # 날짜가 동일한 경우, 주의 첫 번째 날(월요일)과 마지막 날(일요일)로 변경
    if start_date.year == end_date.year and start_date.month==end_date.month and start_date.day==end_date.day:
        start_of_week = start_date - timedelta(days=start_date.weekday())  # 월요일 찾기
        end_of_week = start_of_week + timedelta(days=6)  # 해당 주 일요일 찾기

        return {
            "start_date": start_of_week.replace(hour=0, minute=0, second=0),
            "end_date": end_of_week.replace(hour=23, minute=59, second=59)
        }

    # 날짜가 다르면 기존 time_filter 유지
    return {
        "start_date": start_date,
        "end_date": end_date
    }

def parse_search_query_response(response: str, question: str) -> SearchQuery:
    """
    ChatClovaX 응답을 SearchQuery 객체로 파싱
    """
    try:
        # 응답이 JSON 문자열이라고 가정
        data = json.loads(response.content)
        # time_filter가 null이면 빈 dict으로 변환
        if data.get("time_filter") is None:
            data["time_filter"] = {}
        # query 필드가 없으면 원본 question을 사용
        if "query" not in data:
            data["query"] = question
        return SearchQuery(**data)
    except Exception:
        # 파싱 실패 시, 기본값 반환
        return SearchQuery(query=question, time_filter=TimeFilter())

def get_query_date(question):
    today = datetime(2025, 1, 25)
    days_since_last_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_last_friday)
    issue_date = last_friday.strftime("%Y-%m-%d")

    # ChatClovaX 인스턴스 생성
    chat = ChatClovaX(
        model="HCX-005",
        temperature = 0
    )

    # 프롬프트: 반드시 SearchQuery 포맷(JSON)으로만 답변하게 유도
    system_prompt = f"""
    You are an AI assistant that extracts date ranges from financial queries.
    The current report date is {issue_date}.
    Your task is to extract the relevant date or date range from the user's query
    and format it in YYYY-MM-DD format.
    If no date is specified, answer with None value.
    Return your answer as a JSON object in this format:
    {{
        "query": "<원본 질문>",
        "time_filter": {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}} or {{"start_date": null, "end_date": null}}
    }}
    답변은 반드시 위 JSON 형태로만 해.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    response = chat.invoke(messages)
    # ChatClovaX 응답을 SearchQuery로 파싱
    search_query = parse_search_query_response(response, question)

    # adjust_time_filter_to_week는 기존 함수 그대로 사용
    parsed_dates = adjust_time_filter_to_week(search_query.time_filter)

    if parsed_dates:
        start = parsed_dates['start_date']
        end = parsed_dates['end_date']
    else:
        start = None
        end = None

    if start is None or end is None:
        expr = None
    else:
        expr = f"issue_date >= '{start.strftime('%Y%m%d')}' AND issue_date <= '{end.strftime('%Y%m%d')}'"
    return expr

def get_query_date(question):
    today = datetime(2025, 1, 25)
    days_since_last_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_last_friday)
    issue_date = last_friday.strftime("%Y-%m-%d")

    # ChatClovaX 인스턴스 생성
    chat = ChatClovaX(
        model="HCX-005",
        temperature = 0
    )

    # 프롬프트: 반드시 SearchQuery 포맷(JSON)으로만 답변하게 유도
    system_prompt = f"""
    You are an AI assistant that extracts date ranges from financial queries.
    The current report date is {issue_date}.
    Your task is to extract the relevant date or date range from the user's query
    and format it in YYYY-MM-DD format.
    If no date is specified, answer with None value.
    Return your answer as a JSON object in this format:
    {{
        "query": "<원본 질문>",
        "time_filter": {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}} or {{"start_date": null, "end_date": null}}
    }}
    답변은 반드시 위 JSON 형태로만 해.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    response = chat.invoke(messages)
    # ChatClovaX 응답을 SearchQuery로 파싱
    search_query = parse_search_query_response(response, question)

    # adjust_time_filter_to_week는 기존 함수 그대로 사용
    parsed_dates = adjust_time_filter_to_week(search_query.time_filter)

    if parsed_dates:
        start = parsed_dates['start_date']
        end = parsed_dates['end_date']
    else:
        start = None
        end = None

    if start is None or end is None:
        expr = None
    else:
        expr = f"issue_date >= '{start.strftime('%Y%m%d')}' AND issue_date <= '{end.strftime('%Y%m%d')}'"
    return expr

def convert_to_list(example):
    if isinstance(example["contexts"], list):
        contexts = example["contexts"]
    else:
        try:
            contexts = json.loads(example["contexts"])
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {example['contexts']} - {e}")
            contexts = []
    return {"contexts": contexts}

text_prompt = PromptTemplate.from_template(
'''
today is '2025-01-25'. You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
If question has date expressions, context already filtered with the date expression, so ignore about the date and answer without it.
Answer in Korean. Answer in detail.

#Question:
{question}
#Context:
{context}

#Answer:'''
)


# 평가 결과 모델
class GroundnessQuestionScore(BaseModel):
    score: str = Field(
        description="질문과 답변이 명확히 관련 있으면 'yes', 아니면 'no'"
    )

class GroundnessAnswerRetrievalScore(BaseModel):
    score: str = Field(
        description="검색된 문서와 답변이 명확히 관련 있으면 'yes', 아니면 'no'"
    )

class GroundnessQuestionRetrievalScore(BaseModel):
    score: str = Field(
        description="검색된 문서와 질문이 명확히 관련 있으면 'yes', 아니면 'no'"
    )

class GroundednessCheckerClovaX:
    """
    HyperCLOVA X 기반의 정확성 평가기 (OpenAI 버전과 동일한 구조)
    """

    def __init__(self, llm, target="retrieval-answer"):
        self.llm = llm
        self.target = target


    def create(self):
        # 프롬프트 선택
        if self.target == "retrieval-answer":
            template = """
                You are a grader assessing relevance of a retrieved document to a user question.
                Here is the retrieved document: {context}
                Here is the answer: {answer}
                If the document contains keyword(s) or semantic meaning related to the user answer, grade it as relevant.
                Return your answer as a JSON object: {{"score": "yes"}} or {{"score": "no"}}.
                답변은 반드시 위 JSON 형태로만 해.
            """
            input_vars = ["context", "answer"]
        elif self.target == "question-answer":
            template = """
                You are a grader assessing whether an answer appropriately addresses the given question.
                Here is the question: {question}
                Here is the answer: {answer}
                If the answer directly addresses the question and provides relevant information, grade it as relevant.
                Consider both semantic meaning and factual accuracy in your assessment.
                Return your answer as a JSON object: {{"score": "yes"}} or {{"score": "no"}}.
                답변은 반드시 위 JSON 형태로만 해.
            """
            input_vars = ["question", "answer"]
        elif self.target == "question-retrieval":
            template = """
                You are a grader assessing whether a retrieved document is relevant to the given question.
                Here is the question: {question}
                Here is the retrieved document: {context}
                If the document contains information that could help answer the question, grade it as relevant.
                Consider both semantic meaning and potential usefulness for answering the question.
                Return your answer as a JSON object: {{"score": "yes"}} or {{"score": "no"}}.
                답변은 반드시 위 JSON 형태로만 해.
            """
            input_vars = ["question", "context"]
        else:
            raise ValueError(f"Invalid target: {self.target}")

        def chain_func(**kwargs):
            prompt = template.format(**{k: kwargs[k] for k in input_vars})
            messages = [
                {"role": "system", "content": prompt}
            ]
            response = self.llm.invoke(messages)
            # 응답에서 yes/no만 추출
            answer = response.content.strip().replace('"', '').replace("'", "").lower()
            if answer not in ["yes", "no"]:
                import re
                match = re.search(r'\b(yes|no)\b', answer)
                if match:
                    answer = match.group(1)
                else:
                    raise ValueError(f"Invalid response: {response}")

            # Pydantic 모델 매핑
            if self.target == "retrieval-answer":
                return GroundnessAnswerRetrievalScore(score=answer)
            elif self.target == "question-answer":
                return GroundnessQuestionScore(score=answer)
            elif self.target == "question-retrieval":
                return GroundnessQuestionRetrievalScore(score=answer)

        return chain_func

question_answer_relevant = GroundednessCheckerClovaX(
  llm=ChatClovaX(model="HCX-005"), target='question-answer'
).create()

@chain
def kill_table(result):
    if question_answer_relevant(question=result['question'], answer=result['text']).score == 'no':
        result['context'] = table_chain.invoke({'question': result['question']})
    else:
        result['context'] = result['text']
    return result

URI = 'http://127.0.0.1:19530'

text_db = Milvus(
    embedding_function=embeddings,
    connection_args = {'uri': URI},
    index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
    collection_name='text_db'
)

image_db = Milvus(
    embedding_function=embeddings,
    connection_args = {'uri': URI},
    index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
    collection_name='image_db'
)

raptor_db = Milvus(
    embedding_function=embeddings,
    connection_args = {'uri': URI},
    index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
    collection_name='raptor_db'
)

table_db = Milvus(
    embedding_function=embeddings,
    connection_args = {'uri': URI},
    index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
    collection_name='table_db'
)

filepath = './chunked_jsonl/text_semantic_per_80.jsonl'

splitted_doc_text = []
with open(filepath, 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('\n('):
            continue
        data = json.loads(line)

        doc = Document(
            page_content=data['page_content'],
            metadata=data['metadata']
        )
        splitted_doc_text.append(doc)

filepath = './chunked_jsonl/table_v7.jsonl'

splitted_doc_table = []
with open(filepath, 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('\n('):
            continue
        data = json.loads(line)

        doc = Document(
            page_content=data['page_content'],
            metadata=data['metadata']
        )
        splitted_doc_table.append(doc)

bm25_retriever_table = KiwiBM25Retriever.from_documents(
    splitted_doc_table
)

bm25_retriever_table.k = 20

bm25_retriever_text = KiwiBM25Retriever.from_documents(
    splitted_doc_text
)
bm25_retriever_text.k = 50

bm25_2_retriever_text = KiwiBM25Retriever.from_documents(
    splitted_doc_text
)
bm25_2_retriever_text.k = 5

def format_docs(docs):
    # 각 문서의 issue_date와 page_content를 함께 출력하도록 포맷합니다.
    return "\n\n".join(
        f"Issue Date: {doc.metadata.get('issue_date', 'Unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )

table_prompt = PromptTemplate.from_template(
'''You are an assistant for question-answering tasks.
Use the following pieces of retrieved table to answer the question.
If you don't know the answer, just say that you don't know.
Answer in Korean. Answer in detail.

#Question:
{question}
#Context:
{context}

#Answer:'''
)

llm = ChatClovaX(model='HCX-005', temperature=0)

answer = []

text_chain = (
    RunnableParallel(
        question=itemgetter('question')
    ).assign(expr = lambda x: get_query_date(x['question'])
    ).assign(context_raw=lambda x: RunnableLambda(
            lambda _: text_db.as_retriever(
                search_kwargs={'expr': x['expr'], 'k': 25}
            ).invoke(x['question'])
        ).invoke({}),
    ).assign(
        formatted_context=lambda x: format_docs(x['context_raw'])
    )
    | RunnableLambda(
        lambda x: {
            "question": x['question'],
            "context": x['formatted_context'],  
        }
    )
    | text_prompt
    | llm
    | StrOutputParser()
)

table_chain = (
    RunnableParallel(
        question=itemgetter('question')
    ).assign(expr = lambda x: get_query_date(x['question'])
    ).assign(milvus=lambda x: RunnableLambda(
            lambda _: table_db.as_retriever(
                search_kwargs={'expr': x['expr'], 'k': 10}
            ).invoke(x['question'])
        ).invoke({}),
        bm25_raw=lambda x: bm25_retriever_table.invoke(x['question'])
    ).assign(
        bm25_filtered=lambda x: [
            doc for doc in x["bm25_raw"]
            if not x["expr"] or (
                x["expr"].split("'")[1] <= doc.metadata.get("issue_date", "") <= x["expr"].split("'")[3]
            )
        ],
    ).assign(
        context=lambda x: x['milvus'] + x['bm25_filtered']
    )
    | RunnableLambda(
        lambda x: {
            "question": x['question'],
            "context": x['context'],
        }
    )
    | table_prompt
    | llm
    | StrOutputParser()
)

general_prompt = PromptTemplate.from_template(
  '''You are question-answering AI chatbot about financial reports.
  주어진 두 개의 정보는 table과 text에서 가져온 정보들이야. 이 정보를 바탕으로 질문에 대해 자세히 설명해줘.
  
  If one of the table or text says it doesn't know or it can't answer, don't mention with that.
  And some questions may not be answered simply with context, but rather require inference. In those cases, answer by inference. 
  
  #Question:
  {question}

  #Text Answer:
  {text}

  #Table Answer:
  {table}
  '''
)

general_chain = (
    RunnableParallel(
        question=RunnablePassthrough(),
        text=text_chain,
        table=table_chain,
    )
    | general_prompt 
    | llm
)


predict_chain = (
    RunnableParallel(
        question=RunnablePassthrough(),
        text=text_chain,
        table=table_chain,
    )
    | general_prompt 
    | llm
)

metadata_field_info = [
  AttributeInfo(
    name='source',
    description='문서의 번호. 네 자리의 숫자와 "호"로 이루어져 있다. 현재 1090호부터 1120호까지 존재한다.',
    type='string',
  ),
]

prompt_query = get_query_constructor_prompt(
  'summary of weekly financial report about bonds',
  metadata_field_info
)

output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt_query | llm | output_parser


prompt_raptor = PromptTemplate.from_template(
'''You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer in Korean. Answer in detail.
If the context mentions an unrelated date, do not mention that part.
Summarize and organize your answers based on the various issues that apply to the period.

#Question:
{question}
#Context:
{context}

#Answer:'''
)

retriever_raptor = SelfQueryRetriever(
  query_constructor=query_constructor,
  vectorstore=raptor_db,
  structured_query_translator=MilvusTranslator(),
  search_kwargs={'k': 10}
)

raptor_chain = (
    RunnableParallel(
        question=itemgetter('question')
    ).assign(expr = lambda x: get_query_date(x['question'])
    ).assign(context=lambda x: retriever_raptor.invoke(x['question']))
    | RunnableLambda(
        lambda x: {
            "question": x['question'],
            "context": x['context'],
        }
    )
    | prompt_raptor
    | llm
)


raptor_date_chain = (
    RunnableParallel(
        question=itemgetter('question')
    ).assign(expr = lambda x: get_query_date(x['question'])
    ).assign(context=lambda x: RunnableLambda(
            lambda _: raptor_db.as_retriever(
                search_kwargs={'expr': x['expr'], 'k': 10}
            ).invoke(x['question'])
        ).invoke({})
    )
    | RunnableLambda(
        lambda x: {
            "question": x['question'],
            "context": x['context'],
        }
    )
    | prompt_raptor
    | llm
)

prompt_routing = PromptTemplate.from_template(
  '''주어진 사용자 질문을 `요약`, `예측`, 또는 `일반` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.
  
  <question>
  {question}
  </question>
  
  Classification:'''
)

chain_routing = (
  {'question': RunnablePassthrough()}
  | prompt_routing
  | llm
  | StrOutputParser()
)

prompt_routing_2 = PromptTemplate.from_template(
  '''주어진 사용자 질문을 `날짜`, `호수` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.
  
  <question>
  {question}
  </question>
  
  Classification:'''
)

chain_routing_2 = (
  {'question': RunnablePassthrough()}
  | prompt_routing_2
  | llm
  | StrOutputParser()
)

def route_2(info):
  if '날짜' in info['topic'].lower():
    return raptor_date_chain
  else:
    return raptor_chain
  
def route(info):
  if '요약' in info['topic'].lower():
    return route_2(info)
  elif '예측' in info['topic'].lower():
    return predict_chain
  else:
    return general_chain


full_chain = (
  {'topic': chain_routing, 'question': itemgetter('question')}
  | RunnableLambda(
    route
  )
  | StrOutputParser()
)

retriever_image = image_db.as_retriever(search_kwargs={'k': 3})

retrieval_answer_relevant = GroundednessCheckerClovaX(
  llm, target='retrieval-answer'
).create()

import matplotlib.pyplot as plt
from matplotlib import rc
from PIL import Image

def ask(question):
    expr = get_query_date(question)
    answer = full_chain.invoke({'question': question})
    print(answer)
    rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False        
    context = retriever_image.invoke(question, expr=expr)
    for i in context:
        rar = retrieval_answer_relevant(context=i, answer=answer)
        if rar.score=='yes':
            plt.title('참고 자료')
            image_path = i.metadata['image'].replace('raw_pdf_copy3', 'parsed_pdf')
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

ask('연준의 비공식 대변인은?')