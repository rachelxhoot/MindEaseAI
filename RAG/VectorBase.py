import os
from typing import Dict, List, Optional, Tuple, Union
import json
from RAG.Embeddings import BaseEmbeddings, OpenAIEmbedding, AGICTOEmbedding
import numpy as np
from tqdm import tqdm
import chromadb

from typing import Any, List, Optional

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery

import chromadb
class ChromaVectorStore:
    def __init__(self, chroma_collection):
        self.chroma_collection = chroma_collection

    def count(self):
        return self.chroma_collection.count()

def load_vector_database(persist_dirs: dict, action: str) -> dict:
    """
    加载或创建多个向量数据库
    
    Args:
        persist_dirs (dict): 包含数据类型及其对应持久化目录路径的字典
        action (str): 操作类型 (load: 加载; create: 创建)
    
    Returns:
        dict: 包含所有加载的向量存储对象的字典
    """
    collections = {}
    
    for type, persist_dir in persist_dirs.items():
        if action == "load" and os.path.exists(persist_dir):
            print(f"正在加载现有的向量数据库: {persist_dir}")
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.get_collection(type)
        elif action == "create":
            print(f"创建新的向量数据库: {persist_dir}")
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            chroma_collection = chroma_client.create_collection(type)
        
        print(f"Vector store loaded with {chroma_collection.count()} documents for type {type}")
        collections[type] = ChromaVectorStore(chroma_collection=chroma_collection)
    
    return collections

# # 示例使用
# persist_dirs = {
#     "type1": "path/to/dir1",
#     "type2": "path/to/dir2",
#     "type3": "path/to/dir3"
# }
# collections = load_vector_database(persist_dirs, "load")


class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""

    #*****************
    #Hyperparameter  of similarity vectors/senetences/window/: similarity_top_k
    #************
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索相关文档
        
        Args:
            query_bundle (QueryBundle): 查询包
        
        Returns:
            List[NodeWithScore]: 检索到的文档节点及其相关性得分
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores
    
# # Reference:
# class VectorStore:
#     def __init__(self, document: List[str] = ['']) -> None:
#         self.document = document

#     # 获得文档的向量表示
#     def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
#         self.vectors = []
#         for doc in tqdm(self.document, desc="Calculating embeddings"):
#             self.vectors.append(EmbeddingModel.get_embedding(doc))
#         return self.vectors
    
#     # 数据库持久化，本地保存
#     def persist(self, path: str = 'storage'):
#         if not os.path.exists(path):
#             os.makedirs(path)
#         with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
#             json.dump(self.document, f, ensure_ascii=False)
#         if self.vectors:
#             with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
#                 json.dump(self.vectors, f)

#     # 从本地加载数据库
#     def load_vector(self, path: str = 'storage'):
#         with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
#             self.vectors = json.load(f)
#         with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
#             self.document = json.load(f)

#     def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
#         return BaseEmbeddings.cosine_similarity(vector1, vector2)
    
#     # 根据问题检索相关的文档片段
#     def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
#         query_vector = EmbeddingModel.get_embedding(query)
#         result = np.array([self.get_similarity(query_vector, vector)
#                           for vector in self.vectors])
#         return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()