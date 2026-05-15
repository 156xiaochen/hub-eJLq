import json
import os
import re
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
KB_DIR = PROJECT_ROOT / "第6周：RAG工程化实现" / "04-government-advanced-rag" / "docs"
INDEX_PATH = BASE_DIR / "local_kb_index.json"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
TOP_K = 3

CLAUDE_MODEL = "claude-sonnet-4-5"

def load_text_documents(kb_dir):
  documents = []
  for file_path in kb_dir.glob("*"):
    if file_path.suffix.lower() not in [".md", ".txt"]:
      continue

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
      continue

    documents.append({
      "source": str(file_path),
      "text": text,
    })
  return documents


def split_text_with_overlap(text, chunk_size, chunk_overlap):
  chunks = []
  start = 0
  while start < len(text):
    end = start + chunk_size
    chunk = text[start:end].strip()
    if chunk:
      chunks.append(chunk)
    start = start + chunk_size - chunk_overlap
  return chunks


def build_local_index(documents):
  index = []
  for doc_id, document in enumerate(documents, start =1):
    chunks = split_text_with_overlap(text=document["text"], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for chunk_id, chunk_text in enumerate(chunks, start = 1):
      index.append({
        "doc_id": doc_id,
        "source":document["source"],
        "chunk_id": chunk_id,
        "text":chunk_text
      })
  return index


def save_index (index, index_path):
  index_path.write_text(
    json.dumps(index, ensure_ascii=False, indent=2),
    encoding = "utf-8"
  )

def load_index(index_path):
  return json.loads(index_path.read_text(encoding="utf-8"))


# simple implementation of tokenize and retrieve

def tokenize(text):
  text = text.lower()
  chars = re.findall(r"[\u4e00-\u9fff]", text)
  return set(chars)


def retrieve(question, index, top_k = TOP_K):
    question_tokens = tokenize(question)
    scored_chunks = []

    for chunk in index:
        chunk_tokens = tokenize(chunk["text"])
        score = len(question_tokens & chunk_tokens)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
    ]


def format_context(chunks):
    if not chunks:
        return "没有检索到相关资料。"

    context_list = []

    for idx, chunk in enumerate(chunks, start=1):
        context_list.append(
            f"资料{idx}：\n"
            f"来源：{chunk['source']}\n"
            f"内容：{chunk['text']}"
        )

    return "\n\n".join(context_list)


def build_question_chain():
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("请先设置环境变量 ANTHROPIC_API_KEY 后再调用 Claude。")

    prompt = ChatPromptTemplate.from_template(
        """
你是一个本地知识库问答助手。

请只根据下面的资料回答问题。
如果资料中没有答案，请回答：无法从本地知识库中找到答案。
请使用中文回答。

资料：
{context}

问题：
{question}

回答：
"""
    )

    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=api_key,
    )

    return prompt | llm | StrOutputParser()



def answer_question(question, index):
    related_chunks = retrieve(question, index)
    context = format_context(related_chunks)

    chain = build_question_chain()

    return chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )


def main():
    if not INDEX_PATH.exists():
        documents = load_text_documents(KB_DIR)
        index = build_local_index(documents)
        save_index(index, INDEX_PATH)
        print(f"本地索引创建完成，共保存 {len(index)} 个 chunk")
    else:
        index = load_index(INDEX_PATH)
        print(f"已加载本地索引，共 {len(index)} 个 chunk")

    question = "RAG 项目的知识库文档是如何存储的？"

    related_chunks = retrieve(question, index)
    print("\n检索到的资料：")
    print(format_context(related_chunks))

    try:
        answer = answer_question(question, index)
        print("\n最终回答：")
        print(answer)
    except ValueError as e:
        print("\n最终回答：")
        print(f"无法调用 Claude：{e}")


if __name__ == "__main__":
    main()
