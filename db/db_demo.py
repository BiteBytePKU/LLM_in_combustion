from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from tqdm import tqdm
import os

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))

    return file_list


# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'pdf':
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs


def createDb():
    # 目标文件夹
    tar_dir = [
        # "../docs/InternLM",
        # "../docs/InternLM-XComposer",
        # "../docs/lagent",
        # "../docs/lmdeploy",
        # "../docs/opencompass",
        # "../docs/xtuner",
        "../docs/combustion"
    ]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    # 对文本进行分块

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # 加载开源词向量模型BAAI/bge-m3
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name = "BAAI/bge-m3"
    # model_kwargs = {"device": "gpu"}
    # encode_kwargs = {"normalize_embeddings": True}
    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    # )
    # 构建向量数据库
    # 定义持久化路径
    persist_directory = 'data_base/chroma'
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()


if __name__ == "__main__":
    createDb()
    persist_directory = '../db/data_base/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )
    query = "请你解释一下燃烧领域中的机器学习方法"
    docs = vectordb.similarity_search(query)  # 计算相似度，并把相似度高的chunk放在前面
    context = [doc.page_content for doc in docs]  # 提取chunk的文本内容
    print(context)

