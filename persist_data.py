from RAG.utils import load_data_json, Config
from RAG.VectorBase import load_vector_database
import os

config = Config()
# 清空之前的模型和数据
config.initialize()

from FlagEmbedding import FlagModel
        
# 这里写死 embedding model 为 BAAI/bge-small-zh-v1.5
embed_model = FlagModel('BAAI/bge-small-zh-v1.5', 
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

# embeddings_1 = model.encode(sentences_1)
# similarity = embeddings_1 @ embeddings_2.T

# 加载和处理数据
data_path = config.get("data_path")
vector_store = load_vector_database(config.get('persist_dir'), "create")

for type, dir in data_path.items():
    print(type)
    if os.path.isdir(dir):
        files = [os.path.join(dir, file) for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
        for file_path in files:
            nodes = load_data_json(data_path=file_path)
            for node in nodes:
                node_embedding = embed_model.encode(
                    node.get_content(metadata_mode="all")
                )
                node.embedding = node_embedding.tolist()
            # 将 nodes 添加到向量存储
            vector_store[type].add(nodes)
    else:
        print(f"提供的路径 {dir} 不是一个目录")