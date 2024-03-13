from langchain.vectorstores import Chroma
import os
from model import miniCPM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from modelscope import snapshot_download
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import logging
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
# # 第一步：创建文件日志对象
# logger = logging.getLogger()
# # 第二步：创建文件日志处理器，默认logging会自己创建一个处理器
# file_fmt = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=logging.DEBUG, format=file_fmt, filename="./log.txt", filemode="a", encoding="utf-8")
# console_handler = logging.StreamHandler()
# # 第三步：添加控制台文本处理器
# console_handler.setLevel(level=logging.DEBUG)
# console_fmt = "%(asctime)s - %(levelname)s - %(message)s"
# fmt1 = logging.Formatter(fmt=console_fmt)
# console_handler.setFormatter(fmt=fmt1)
# # 第四步：将控制台日志器、文件日志器，添加进日志器对象中
# logger.addHandler(console_handler)

path = snapshot_download('OpenBMB/miniCPM-bf16')

def load_chain():
    
    # 向量数据库持久化路径
    persist_directory = '../db/data_base/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    # 加载自定义 LLM
    llm = miniCPM(model_path=path)
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )
    # 定义一个 Prompt Template
    template = """你是AISI科学智能研发的专家模型，请使用以下上下文来回答最后的问题。如果不知道答案，就说你不知道，不要编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {chat_history}
    {context}
    问题: {question}
    回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
    # 运行 chain
    chain = ConversationalRetrievalChain.from_llm(llm, memory=memory, retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})
    return chain

class myChatModel():
    """
    存储检索问答链的对象
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def get_reply(self, question: str):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return ""
        try:
            res = self.chain.invoke({"question": question})
            # print(res)
            return res
        except Exception as e:
            return e

myChat = myChatModel()

if __name__ == "__main__":
    chain = load_chain()
    res = chain.invoke({"question": "北京大学"})
    print(res)
