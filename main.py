from fastapi import FastAPI, Request, Response, BackgroundTasks
from lxml import etree
import time
import hashlib
import dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

dotenv.load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_MSG_IN_HISTORY = os.getenv("MAX_MSG_IN_HISTORY")
TOKEN = os.getenv("WECHAT_TOKEN")

# chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=MAX_TOKENS)
chat = QianfanChatEndpoint(temperature=0.1)
chat_histories = {}




# llm = ChatOpenAI(model="gpt-3.5-turbo")
# all_pages = []
# for root, dirs, files in os.walk("rules"):
    # for name in files:
        # loader = PyPDFLoader(os.path.join(root, name))
        # pages = loader.load_and_split()
        # all_pages += pages
# print(len(all_pages))
# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(all_pages, embeddings)
# vectorstore.save_local(folder_path='vectors')

# vectorstore = FAISS.load_local('/home/ubuntu/fastapi/vectors', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
vectorstore = FAISS.load_local('vectors', QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh"), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
    chat, retriever, contextualize_q_prompt
)

qa_system_prompt = """
    你是一个回答中国结算深圳分公司业务指南相关问题的助手。\
    请使用以下检索到的上下文来回答问题。\
    如果你不知道答案，或者遇到与上下文无关的问题，就直接说“根据已知内容无法回答，请修改问题。”。\
    如果问题范围模糊，难以回答，不要随便回答，要求用户澄清问题。\
    使用序号要点，并保持答案简洁。 \
    {context}"""
    # 在答案的最后，将信息源按序号要点列出。 \

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# custom_rag_prompt = PromptTemplate.from_template(template)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | custom_rag_prompt
#     | chat
#     | StrOutputParser()
# )


# def get_access_token():
    # if time.time() > TOKEN_EXPIRY_TIME - 300:
    #     return ACCESS_TOKEN
    # res = requests.get(ACCESS_TOKEN_URL).json()
    # ACCESS_TOKEN = res['access_token']
    # TOKEN_EXPIRY_TIME = int(time.time()) + res['expires_in']
    # return ACCESS_TOKEN
    # res = requests.get(ACCESS_TOKEN_URL).json()
    # print(res)
    # return res['access_token']


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


# def check_signature(signature: str, timestamp: str, nonce: str) -> bool:
#     tmp_arr = [TOKEN, timestamp, nonce]
#     tmp_arr.sort()
#     tmp_str = ''.join(tmp_arr)
#     tmp_str = hashlib.sha1(tmp_str.encode('utf-8')).hexdigest()
#     return tmp_str == signature


# def get_reply(user, message):
#     if user not in chat_histories:
#         chat_histories[user] = []
#     user_chat_history = chat_histories[user]
#     ai_reply = rag_chain.invoke({"input": message, "chat_history": user_chat_history})
#     answer = ai_reply["answer"]
#     user_chat_history.extend([HumanMessage(content=message), answer])

@app.get('/test')
async def test():
    return "test"

@app.post('/wechat')
async def chat_with_knowledge_base(request: Request):
    user, me, message = await decode_message(request)
    if user in chat_histories:
        if len(chat_histories[user].messages) >= MAX_MSG_IN_HISTORY:
            chat_histories[user] = ChatMessageHistory()
    ai_reply = conversational_rag_chain.invoke({"input": message}, {'configurable': {'session_id': user}})
    return Response(content=xml_format(user, me, ai_reply['answer']), media_type="application/xml")

# def hand_shake(signature: str, timestamp: str, nonce: str, echostr: str):
    # if check_signature(signature, timestamp, nonce):
        # return int(echostr)
    # return False

