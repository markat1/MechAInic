import chainlit as cl
import tiktoken
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

RAG_PROMPT = """

CONTEXT:
{context}

QUERY:
{question}

You are a car specialist and can only provide your answers from the context. 

Don't tell in your response that you are getting it from the context.

"""

init_settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

car_manual = PyMuPDFLoader(os.environ.get('pdfurl'))
car_manual_data = car_manual.load()

text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 400,
chunk_overlap = 50,
length_function = tiktoken_len)
    
car_manual_chunks = text_splitter.split_documents(car_manual_data)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Pinecone.from_documents(car_manual_chunks, embedding_model, index_name=os.environ.get('index'))
retriever = vector_store.as_retriever()

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

model = ChatOpenAI(model="gpt-3.5-turbo")

@cl.on_chat_start
async def main():
    mecanic_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | rag_prompt | model | StrOutputParser()
    )

    cl.user_session.set("runnable", mecanic_qa_chain)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question":message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
