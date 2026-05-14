"""
基于LangGraph的本地知识库问答系统
包含文档检索 + LLM回答流程
"""

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json


# 初始化LLM模型
model = ChatOpenAI(
    model="qwen-flash",  # 模型的代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-d9a7f2143b1643c08ec968d650193995"
)

# 初始化嵌入模型（用于向量检索）
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


class RAGState(TypedDict):
    """RAG系统的状态定义"""
    question: str  # 用户问题
    retrieved_docs: List[Dict[str, Any]]  # 检索到的文档
    answer: str  # LLM生成的回答
    messages: Annotated[List[Any], operator.add]  # 对话历史


def load_knowledge_base(kb_path: str = "knowledge_base.json") -> List[Dict]:
    """
    加载本地知识库
    :param kb_path: 知识库文件路径
    :return: 知识库文档列表
    """
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"知识库文件 {kb_path} 不存在，请先创建知识库文件")


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    获取文本的向量表示
    :param texts: 文本列表
    :return: 向量矩阵
    """
    return embedding_model.encode(texts)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算余弦相似度
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 相似度分数
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def retrieve_documents(state: RAGState) -> RAGState:
    """
    文档检索节点：从本地知识库中检索与问题相关的文档
    """
    question = state["question"]
    
    # 加载知识库
    knowledge_base = load_knowledge_base()
    
    # 获取问题的向量表示
    question_embedding = get_embeddings([question])[0]
    
    # 计算问题与每个文档的相似度
    similarities = []
    for doc in knowledge_base:
        doc_embedding = get_embeddings([doc["content"]])[0]
        similarity = cosine_similarity(question_embedding, doc_embedding)
        similarities.append((doc, similarity))
    
    # 按相似度排序，取前3个最相关的文档
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, sim in similarities[:3]]
    
    # 更新状态
    state["retrieved_docs"] = top_docs
    
    return state


def generate_answer(state: RAGState) -> RAGState:
    """
    LLM回答节点：基于检索到的文档生成回答
    """
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]
    
    # 构建上下文
    context = "\n\n".join([f"文档标题: {doc['title']}\n内容: {doc['content']}" for doc in retrieved_docs])
    
    # 构建提示词
    prompt = f"""你是一个智能助手，请根据以下参考资料回答用户的问题。

参考资料:
{context}

用户问题: {question}

请基于上述参考资料提供准确、简洁的回答。如果参考资料中没有相关信息，请说明无法回答。"""

    # 调用LLM生成回答
    response = model.invoke([
        SystemMessage(content="你是一个专业的知识问答助手，擅长根据提供的资料回答问题。"),
        HumanMessage(content=prompt)
    ])
    
    # 更新状态
    state["answer"] = response.content
    state["messages"] = [HumanMessage(content=question), AIMessage(content=response.content)]
    
    return state


def build_rag_graph():
    """
    构建RAG图工作流
    """
    # 创建状态图
    rag_builder = StateGraph(RAGState)
    
    # 添加节点
    rag_builder.add_node("retrieve_documents", retrieve_documents)
    rag_builder.add_node("generate_answer", generate_answer)
    
    # 添加边
    rag_builder.add_edge(START, "retrieve_documents")
    rag_builder.add_edge("retrieve_documents", "generate_answer")
    rag_builder.add_edge("generate_answer", END)
    
    # 编译图
    rag_agent = rag_builder.compile()
    
    return rag_agent


def ask_question(question: str) -> Dict[str, Any]:
    """
    向RAG系统提问
    :param question: 用户问题
    :return: 包含答案和相关信息的字典
    """
    # 构建RAG图
    rag_agent = build_rag_graph()
    
    # 初始化状态
    initial_state = {
        "question": question,
        "retrieved_docs": [],
        "answer": "",
        "messages": []
    }
    
    # 执行RAG流程
    result = rag_agent.invoke(initial_state)
    
    return {
        "question": result["question"],
        "retrieved_docs": result["retrieved_docs"],
        "answer": result["answer"],
        "messages": result["messages"]
    }


if __name__ == "__main__":
    # 测试RAG系统
    test_questions = [
        "什么是langchain？",
        "langgraph架构介绍？"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"问题: {question}")
        print(f"{'='*50}")
        
        result = ask_question(question)
        
        print("\n检索到的相关文档:")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            print(f"{i}. {doc['title']}: {doc['content'][:100]}...")
        
        print(f"\n回答: {result['answer']}")