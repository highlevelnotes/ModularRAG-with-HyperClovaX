import json
from langchain.schema import Document
from langchain_upstage import UpstageEmbeddings
from langchain_milvus.vectorstores import Milvus
from langchain_community.vectorstores import Milvus as Mil2
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

from base import *
from langchain_teddynote.evaluator import GroundednessChecker
from langchain_core.messages.chat import ChatMessage
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rc
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

@st.cache_resource
def create_chain():
  embeddings = UpstageEmbeddings(
      model='solar-embedding-1-large-query',
  )


  filepath = './chunked_jsonl/250313_text_semantic_per_80.jsonl'

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

  URI = 'http://34.64.116.127:19530'

  vectorstore_text = Milvus(
      embedding_function=embeddings,
      connection_args={'uri':URI},
      index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
      collection_name='text_semantic_per_80_00'
  )

  vectorstore_table = Milvus(
    embedding_function=embeddings,
    connection_args={'uri':URI},
    index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
    collection_name='table_v7'
  )

  vectorstore_raptor = Mil2(
      embedding_function=embeddings,
      connection_args={'uri':URI},
      index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
      collection_name='raptor_v3'
  )

  bm25_retriever_table = KiwiBM25Retriever.from_documents(
      splitted_doc_table
  )

  bm25_retriever_table.k = 20

  llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

  bm25_retriever_text = KiwiBM25Retriever.from_documents(
      splitted_doc_text
  )
  bm25_retriever_text.k = 50

  bm25_2_retriever_text = KiwiBM25Retriever.from_documents(
      splitted_doc_text
  )
  bm25_2_retriever_text.k = 5


  text_chain = (
      RunnableParallel(
          question=itemgetter('question')
      ).assign(expr = lambda x: get_query_date(x['question'])
      ).assign(milvus=lambda x: RunnableLambda(
              lambda _: vectorstore_text.as_retriever(
                  search_kwargs={'expr': x['expr'], 'k': 20}
              ).invoke(x['question'])
          ).invoke({}),
          bm25_raw=lambda x: bm25_retriever_text.invoke(x['question'])
      ).assign(
          bm25_filtered=lambda x: [
              doc for doc in x["bm25_raw"]
              if not x["expr"] or (
                  x["expr"].split("'")[1] <= doc.metadata.get("issue_date", "") <= x["expr"].split("'")[3]
              )
          ],
      ).assign(
          context=lambda x: reranking(x['milvus'] + x['bm25_filtered'], x['question'])
      )
      | RunnableLambda(
          lambda x: {
              "question": x['question'],
              "context": x['context'],
          }
      )
      | text_prompt
      | llm
      | StrOutputParser()
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

  table_chain = (
      RunnableParallel(
          question=itemgetter('question')
      ).assign(expr = lambda x: get_query_date(x['question'])
      ).assign(milvus=lambda x: RunnableLambda(
              lambda _: vectorstore_table.as_retriever(
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


  llm_general = ChatOpenAI(model='gpt-4o-mini', temperature=0)

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
      | llm_general
  )

  predict_chain = (
      RunnableParallel(
          question=RunnablePassthrough(),
          text=text_chain,
          table=table_chain,
      )
      | general_prompt 
      | ChatOpenAI(model='o1', temperature=1)
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

  query_llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)
  query_constructor = prompt_query | query_llm | output_parser


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
    vectorstore=vectorstore_raptor,
    structured_query_translator=MilvusTranslator(),
    search_kwargs={'k': 10}
  )
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

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
              lambda _: vectorstore_raptor.as_retriever(
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
    | ChatOpenAI(model='gpt-4o-mini')
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
    | ChatOpenAI(model='gpt-4o-mini')
    | StrOutputParser()
  )

  def route_2(info):
    if '날짜' in info['topic'].lower():
      return raptor_date_chain
    else:
      return raptor_chain

  def route(info):
    if '요약' in info['topic'].lower():
      info['topic'] = chain_routing_2.invoke(info['question'])
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
  return full_chain


st.title("보고사")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

@st.cache_resource
def image_chain(user_input, response):
    URI = 'http://34.64.116.127:19530'
    embeddings = UpstageEmbeddings(
        model='solar-embedding-1-large-query',
    )
    vectorstore_image = Milvus(
      embedding_function=embeddings,
      connection_args={'uri':URI},
      index_params={'index_type': 'AUTOINDEX', 'metric_type': 'IP'},
      collection_name='image_v4'
    )
    retriever_image = vectorstore_image.as_retriever(search_kwargs={'k': 3})
    retrieval_answer_relevant = GroundednessChecker(
      llm=ChatOpenAI(model='gpt-4o-mini', temperature=0), target='retrieval-answer'
    ).create()
    expr = get_query_date(user_input)
    rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False 
    context = retriever_image.invoke(user_input, expr=expr)
    img = []
    for i in context:
      rar = retrieval_answer_relevant.invoke({'context': i, 'answer': response})
      if rar.score=='yes':
          image_path = i.metadata['image'].replace('raw_pdf_copy3', 'parsed_pdf')
          img.append(Image.open(image_path))
    return img


if clear_btn:
    st.session_state["messages"] = []
print_messages()

user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)

    chain = create_chain()
    response = chain.invoke({'question':user_input})
    images = image_chain(user_input, response)
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
        with st.expander('참고 자료'):
           for image in images:
              st.image(image)
              
    add_message("user", user_input)
    add_message("assistant", ai_answer)
