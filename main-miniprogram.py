from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import QianfanChatEndpoint
from pydantic import BaseModel
import dotenv

dotenv.load_dotenv()



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_MSG_IN_HISTORY = 3
TOKEN = os.getenv("WECHAT_TOKEN")
MAX_TOKENS = 4000
APPID = os.getenv("APP_ID")
APPSECRET = os.getenv("APP_SECRET")
PROGRAM_TOKEN_URL = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={APPID}&secret={APPSECRET}"

# chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=MAX_TOKENS)
chat = QianfanChatEndpoint(temperature=0.1)

chat_histories = {}
dialogue_count = {}
user_last_messages = {}
user_prev_answer = {}

vectorstore = FAISS.load_local('vectors', QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh"), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

contextualize_q_system_prompt = """鉴于聊天历史记录和最新的用户问题，这个问题可能引用了聊天历史记录中的上下文，\
请重新表述一个不需要聊天历史记录也能理解的独立问题。不要回答问题，只需在必要时重新表述问题，否则按原样返回问题。"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chat, retriever, contextualize_q_prompt
)

qa_system_prompt = """
    你是一个回答问题的助手。请使用以下检索到的上下文来回答问题。 \
    思考问题是否与上下文相关，如果无关，就不要回答。 \
    如果你根据上下文无法得到答案，就直接说你不知道。 \
    使用序号要点，并保持答案简洁。 \
    在答案的最后，将信息源的相关原文的节选按序号要点列出。 \
    --------\
    上下文：{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_reply(user, message):
    if user not in chat_histories:
        chat_histories[user] = []
    user_chat_history = chat_histories[user]
    ai_reply = rag_chain.invoke({"input": message, "chat_history": user_chat_history})
    answer = ai_reply["answer"]
    user_chat_history.extend([HumanMessage(content=message), answer])

class Message(BaseModel):
    message: str

@app.post('/wechat_program')
async def chat_with_knowledge_base_program(item: Message):
    return StreamingResponse(answer_generator(item.message), media_type="text/plain")

async def answer_generator(message: str):
    # if user not in chat_histories:
    #     chat_histories[user] = []
    # user_chat_history = chat_histories[user]
    user_chat_history = []
    # ai_reply = rag_chain.invoke({"input": message, "chat_history": user_chat_history})
    # answer = ai_reply["answer"]
    # user_chat_history.extend([HumanMessage(content=message), answer])
    for chunk in rag_chain.stream({"input": message, "chat_history": user_chat_history}):
        if "answer" in chunk:
            yield chunk["answer"]

@app.get('/')
async def test():
    return 'test'

    
# @app.post('/wechat')
# async def chat_with_knowledge_base(request: Request, background_tasks: BackgroundTasks):
#     user, me, message = awaiçt decode_message(request)

#     if message == "1":
#         if user not in chat_histories:
#             return Response(content=xml_format(user, me, '您还没有提问，请提问后再回复数字"1"'), media_type="application/xml")
#         user_chat_history = chat_histories[user]
#         if len(user_chat_history) == 0:
#             return Response(content=xml_format(user, me, '我还在飞速思考中，请等待5秒后回复数字“1”获取回答'), media_type="application/xml")
#         reply = chat_histories[user][-1]
#         if user in user_prev_answer and reply == user_prev_answer[user]:
#             return Response(content=xml_format(user, me, '我还在飞速思考中，请等待5秒后回复数字“1”获取回答'), media_type="application/xml")
#         else:
#             user_prev_answer[user] = reply
#             return Response(content=xml_format(user, me, reply), media_type="application/xml")
#     # # 检查消息是否重复
#     # if user in user_last_messages and message == user_last_messages[user]:
#     #     return xml_format(user, me, "已超时，请重试")
#     # user_last_messages[user] = message

#     # if user not in chat_histories:
#     #     chat_histories[user] = []

#     # user_chat_history = chat_histories[user]
#     # # reply = rag_chain.invoke(message)
#     # ai_reply = rag_chain.invoke({"input": message, "chat_history": user_chat_history})
#     # answer = ai_reply["answer"]
#     # user_chat_history.extend([HumanMessage(content=message), answer])
#     # return Response(content=xml_format(user, me, ai_reply["answer"]), media_type="application/xml")
#     background_tasks.add_task(get_reply, user, message)
#     return Response(content=xml_format(user, me, "飞速思考中，请等待5秒后回复数字“1”获取回答"), media_type="application/xml")


# async def chat_with_bot(request: Request):
#     user, me, message = await decode_message(request)

#     # 检查消息是否重复
#     if user in user_last_messages and message == user_last_messages[user]:
#         return xml_format(user, me, "已超时，请重试")
#     user_last_messages[user] = message

#     if user not in chat_histories:
#         chat_histories[user] = ChatMessageHistory()

#     user_chat_history = chat_histories[user]

#     if message == '清除对话历史':
#         user_chat_history.clear()
#         return xml_format(user, me, '已清除对话历史')

#     if user in dialogue_count:
#         if dialogue_count[user] >= MAX_MSG_IN_HISTORY:
#             user_chat_history.clear()
#             dialogue_count[user] = 0
#             return xml_format(user, me, f'对话次数达到{MAX_MSG_IN_HISTORY}次，已清除对话历史')
#         dialogue_count[user] += 1
#     else:
#         dialogue_count[user] = 1

#     user_chat_history.add_user_message(message)
#     ai_reply = chain.invoke(
#         {"messages": user_chat_history.messages}
#     ).content
#     user_chat_history.add_ai_message(ai_reply)

#     return xml_format(user, me, ai_reply)