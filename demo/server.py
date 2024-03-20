from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage
from model import miniCPM
from langchain.prompts import PromptTemplate
from modelscope import snapshot_download
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

path = snapshot_download('OpenBMB/miniCPM-bf16')
# 加载自定义 LLM
llm = miniCPM(model_path=path)


def load_RAchain():
    
    # 向量数据库持久化路径
    persist_directory = '../db/data_base/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  
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
   
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=512
    )
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
