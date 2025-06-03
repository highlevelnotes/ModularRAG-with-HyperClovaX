import os
import re
import json
import jsonlines
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_upstage import UpstageEmbeddings
from langchain_milvus.vectorstores import Milvus
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import pandas as pd
import pytz

from datasets import Dataset
import datetime
from datetime import timedelta
from operator import itemgetter
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.query_constructor.base import (
  StructuredQueryOutputParser,
  get_query_constructor_prompt
)
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import warnings

from datetime import datetime
from typing import Optional
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Literal


warnings.filterwarnings('ignore')

def save_docs_to_jsonl(documents, file_path):
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.dict())


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


def rerank_results(query: str, chunks: list[dict]) -> RerankedResults:
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=RerankedResults,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert search result ranker. Your task is to evaluate the relevance of each text chunk to the given query and assign a relevancy score.

                For each chunk:
                1. Analyze its content in relation to the query.
                2. Provide a chain of thought explaining your reasoning.
                3. Assign a relevancy score from 0 to 10, where 10 is most relevant.

                Be objective and consistent in your evaluations.
                """,
            },
            {
                "role": "user",
                "content": """
                <query>{{ query }}</query>

                <chunks_to_rank>
                {% for chunk in chunks %}
                <chunk id="{{ chunk.id }}">
                    {{ chunk.text }}
                </chunk>
                {% endfor %}
                </chunks_to_rank>

                Please provide a RerankedResults object with a Label for each chunk.
                """,
            },
        ],
        context={"query": query, "chunks": chunks},
    )


def get_query_date(question):
    issue_date = "2025-01-25"
    client = instructor.from_openai(OpenAI())
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=SearchQuery,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an AI assistant that extracts date ranges from financial queries.
                The current report date is {issue_date}.
                Your task is to extract the relevant date or date range from the user's query
                and format it in YYYY-MM-DD format.
                If no date is specified, answer with None value.
                """,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    parsed_dates = adjust_time_filter_to_week(response.time_filter)

    # parsed_dates를 순회하며 None인 경우도 처리
    if parsed_dates:
        start = parsed_dates['start_date']
        end=parsed_dates['end_date']
    else:
        start=None
        end = None

    if start is None or end is None:
        expr = None
    else:
        expr = f"issue_date >= '{start.strftime('%Y%m%d')}' AND issue_date <= '{end.strftime('%Y%m%d')}'"
        expr = expr
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

def generate_expr(question: str) -> dict:
    expr = get_query_date(question)
    return {"expr": expr}

def reranking(docs, question):
    chunks = [{"id": idx, "issue_date": doc.metadata['issue_date'],  "text": doc.page_content} for idx, doc in enumerate(docs)]
    documents_with_metadata = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
    reranked_results = rerank_results(query=question, chunks=chunks)

    chunk_dict = {chunk["id"]: chunk["text"] for chunk in chunks}
    top_k_results = [chunk_dict.get(label.chunk_id, "") for label in reranked_results.labels[:15] if label.chunk_id in chunk_dict]

    reranked_results_with_metadata = []
    for reranked_result in top_k_results:
        page_content = reranked_result

        matching_metadata = None
        for doc in documents_with_metadata:
            if doc["text"] == page_content:
                matching_metadata = doc["metadata"]
                break

        document = Document(
            metadata=matching_metadata,
            page_content=page_content
        )
        reranked_results_with_metadata.append(document)

    context_rerankedNbm25 = reranked_results_with_metadata
    return context_rerankedNbm25

text_prompt = PromptTemplate.from_template(
'''You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer specifically and provide detailed evidence from the retrieved context.
Answer in Korean. Answer in detail.

#Question:
{question}
#Context:
{context}

#Answer:'''
)