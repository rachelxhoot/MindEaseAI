from RAG.utils import download_and_quantize_model, download_embedding_model, load_data, Config
from RAG.VectorBase import load_vector_database
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

config = Config()
# 清空之前的模型和数据
config.initialize()

# 准备模型
download_and_quantize_model(config.get("model_name"), config.get("cache_path"), config.get("model_path"))
download_embedding_model(config.get("embedding_model_name"), config.get("cache_path"))

# 加载和处理数据
embed_model = HuggingFaceEmbedding(model_name=config.get("embedding_model_path"))
data_path = config.get("data_path")
vector_store = load_vector_database(data_path, "load")

for type, dir in data_path:
    nodes = load_data(data_path=dir)
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    # 将 node 添加到向量存储
    vector_store[type].add(nodes)