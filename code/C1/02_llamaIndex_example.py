import os
import re
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

# 定位项目根目录，并从根目录的 .env 文件加载 API Key。
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

# 读取 SiliconFlow API Key；未配置时提前提示，避免请求失败后才发现。
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
if not siliconflow_api_key:
    raise ValueError("请在项目根目录的 .env 中设置 SILICONFLOW_API_KEY")

# 配置回答模型：查询时根据检索到的上下文生成最终答案。
Settings.llm = OpenAILike(
    model="Qwen/Qwen3-8B",
    api_key=siliconflow_api_key,
    api_base="https://api.siliconflow.cn/v1",
    is_chat_model=True,
)

# 配置嵌入模型：将文档和查询转换成向量，用于相似度检索。
Settings.embed_model = OpenAIEmbedding(
    # model 使用 LlamaIndex 支持的占位值，model_name 指定 SiliconFlow 上的实际模型名。
    model="text-embedding-3-small",
    model_name="Qwen/Qwen3-Embedding-8B",
    api_key=siliconflow_api_key,
    api_base="https://api.siliconflow.cn/v1",
)

# 加载本地 Markdown 文档，转换为 LlamaIndex 的文档对象。
markdown_path = project_root / "data/C1/markdown/easy-rl-chapter1.md"
docs = SimpleDirectoryReader(input_files=[str(markdown_path)]).load_data()

# 使用标点和换行切分句子，避免默认切分器依赖 NLTK 数据包。
def split_sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[。！？.!?])\s*|\n+", text) if part.strip()]


Settings.node_parser = SentenceSplitter(chunking_tokenizer_fn=split_sentences)

# 为文档切分文本、调用嵌入 API，并建立向量索引。
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎，负责检索相关片段并调用回答模型。
query_engine = index.as_query_engine()

# 打印默认提示词，方便查看 LlamaIndex 如何组织检索上下文和问题。
print(query_engine.get_prompts())

# 提出问题并打印基于文档内容生成的回答。
print(query_engine.query("文中举了哪些例子?"))
