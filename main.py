import time
import dotenv
import os
from lxml import etree
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.llms import Tongyi, QianfanLLMEndpoint, BaichuanLLM

dotenv.load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_MSG_IN_HISTORY = int(os.getenv("MAX_MSG_IN_HISTORY"))
TOKEN = os.getenv("WECHAT_TOKEN")

chat_histories = {}

def create_chain():
    llm = QianfanLLMEndpoint()
    # llm = Tongyi(model_name="qwen-turbo")
    # llm = BaichuanLLM()
    vectorstore = FAISS.load_local('vectors', QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh"), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qa_system_prompt = """
        你是一个回答中国结算深圳分公司业务指南相关问题的助手。\
        请使用以下检索到的上下文来回答问题。\
        如果你不知道答案，或者遇到与上下文无关的问题，就直接说“根据已知内容无法回答，请修改问题。”。\
        如果问题范围模糊，难以回答，不要随便回答，要求用户澄清问题。\
        使用序号要点，并保持答案简洁，字数不要超过100字。\
        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def xml_format(user, me, content):
    return f'''
            <xml>
             <ToUserName><![CDATA[{user}]]></ToUserName>
             <FromUserName><![CDATA[{me}]]></FromUserName>
             <CreateTime>{int(time.time())}</CreateTime>
             <MsgType><![CDATA[text]]></MsgType>
             <Content><![CDATA[{content}]]></Content>
            </xml>
            '''

async def decode_message(request: Request):
    body = await request.body()
    body = body.decode()
    root = etree.fromstring(body)
    user_username = root.find('FromUserName').text
    my_username = root.find('ToUserName').text
    content = root.find('Content').text
    return user_username, my_username, content

@app.get('/test')
async def test():
    return "test"

conversational_rag_chain = create_chain()

@app.post('/wechat')
async def chat_with_knowledge_base(request: Request):
    user, me, message = await decode_message(request)
    if user in chat_histories:
        if len(chat_histories[user].messages) >= MAX_MSG_IN_HISTORY:
            chat_histories[user] = ChatMessageHistory()
    ai_reply = conversational_rag_chain.invoke({"input": message}, {'configurable': {'session_id': user}})
    return Response(content=xml_format(user, me, ai_reply['answer']), media_type="application/xml")

