"""
作业1：基于讲解的Langchain框架，开发对本地知识库进行问答的逻辑，只需要包括文档检索+llm问答流程
"""

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 千问大模型配置
API_KEY = "sk-9c6195bf91f7435d88ea4b819073c92c"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 初始化千问大模型
llm = ChatOpenAI(model="qwen-flash", base_url=BASE_URL, api_key=API_KEY)

# 使用本地 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="D:/AI/modelscope/bge-small-zh-v1.5")

# --- 1. 检索相关文档 ---
def retrieve_docs(vectorstore, question, k=3):
    """检索相关文档"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return docs


# --- 2. LLM回答 ---
def answer_question(question, docs):
    """基于检索文档回答问题"""
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""根据以下资料回答问题，如果资料中没有相关信息请说明。

资料：
{context}

问题：{question}
"""
    response = llm.invoke(prompt)
    return response.content


# --- 3. 测试 ---
if __name__ == "__main__":
    # 创建示例文档
    with open("./demo.txt", "w", encoding="utf-8") as f:
        f.write("""LangChain是一个用于开发大语言模型应用的框架。
核心组件包括：模型(Model)、提示词(Prompt)、索引(Index)、链(Chain)、代理(Agent)。
FAISS是Facebook开发的向量相似度搜索库，用于高效向量检索。
RAG是检索增强生成技术，先检索相关文档，再让模型基于检索结果生成回答。""")

    # 加载文档
    loader = TextLoader("./demo.txt", encoding="utf-8")
    documents = loader.load()

    # 文本切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # 每一段最大 200 个字
        chunk_overlap=50  # 段与段之间重叠 50 个字（防止语义被切断）
    )
    chunks = text_splitter.split_documents(documents)

    # 向量化并存入 FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 问答测试
    question = "LangChain有哪些核心组件？"
    print(f"❓问题：{question}")

    # 检索
    docs = retrieve_docs(vectorstore, question)
    print(f"🔍检索到 {len(docs)} 个相关文档")

    # 回答
    answer = answer_question(question, docs)
    print(f"🤖回答：{answer}")
