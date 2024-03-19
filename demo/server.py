from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage
from model import miniCPM
from langchain.prompts import PromptTemplate
from modelscope import snapshot_download
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
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
# 加载自定义 LLM
llm = miniCPM(model_path=path)


def load_RAchain():
    # 加载问答链
    # 定义 Embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 向量数据库持久化路径
    persist_directory = '../db/data_base/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )
    # 定义一个 Prompt Template
    template = """你是一名科学家，根据提问所用的语言来回答问题。总是在回答的最后说“谢谢提问！”。
    {chat_history}
    {context}
    问题: {question}
    回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)

    # 运行 chain
    chain = ConversationalRetrievalChain.from_llm(llm, memory=memory,
                                                  retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
                                                  return_source_documents=True,
                                                  combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})

    return chain


def load_chain():
    # SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。
    # EXAMPLE
    # Current summary:
    # Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。
    # New lines of conversation:
    # Human：为什么你认为人工智能是一种积极的力量？
    # AI：因为人工智能将帮助人类发挥他们的潜能。
    # New summary:
    # Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
    # END OF EXAMPLE
    # Current summary:
    # {summary}
    # New lines of conversation:
    # {new_lines}
    # New summary:"""
    #
    # SUMMARY_PROMPT = PromptTemplate(
    #     input_variables=["summary", "new_lines"],
    #     template=SUMMARIZER_TEMPLATE
    # )
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=512
    )
    # # 定义一个 Prompt Template
    # template = """你是一名科学家，根据提问所用的语言来回答问题。总是在回答的最后说“谢谢提问！”。
    # {chat_history}
    # 问题: {question}
    # 回答:"""
    #
    # CHAIN_PROMPT = PromptTemplate(input_variables=["chat_history", "question"], template=template)

    # 运行 chain
    chain = ConversationChain(llm=llm, memory=memory)

    return chain


class myChatModel():
    """
    存储检索问答链的对象
    """

    def __init__(self):
        # 构造函数，加载检索问答链
        self.RAchain = load_RAchain()
        self.chain = load_chain()

    def get_reply(self, question: str, status):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return ""
        try:
            if status == 2:
                res = self.RAchain.invoke({"question": question})
            else:
                res = self.chain.predict(input=question)
                print(res)
            return res
        except Exception as e:
            return e


myChat = myChatModel()
if __name__ == "__main__":
    chain = load_chain()
    res = chain.predict(input="北京大学")
    print(res)
