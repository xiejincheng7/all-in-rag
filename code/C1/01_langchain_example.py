import os
from pprint import pprint
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

markdown_path = project_root / "data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = TextLoader(str(markdown_path), encoding="utf-8")
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 通过 SiliconFlow API 获取嵌入向量，无需下载本地模型
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
if not siliconflow_api_key:
    raise ValueError("请在 .env 中设置 SILICONFLOW_API_KEY")

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    api_key=siliconflow_api_key,
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template(
    """
    请根据下面提供的上下文信息来回答问题。
    请确保你的回答完全基于这些上下文。
    如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”
    
    上下文:{context}
    
    问题: {question}
    
    回答:
    """
)

# 配置大语言模型
# 使用同一个 SiliconFlow API Key 生成最终回答
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    temperature=0.7,
    max_tokens=4096,
    api_key=siliconflow_api_key,
    base_url="https://api.siliconflow.cn/v1",
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print("回答内容：")
print(answer.content)
# print("\nLangChain 消息对象及元数据：")
# pprint(answer.model_dump(), sort_dicts=False)
