import os
from copy import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# os.environ['CURL_CA_BUNDLE'] = ''
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path # 模型的路径
        self.is_api = is_api # 是否是 API 模型
    
    # 获取文本的向量表示
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    
    # 计算两个向量之间的余弦相似度
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
    

class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError

class AGICTOEmbedding(BaseEmbeddings):
    """
    class for AGICTO embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("AGICTO_API_KEY")
            self.client.base_url = os.getenv("AGICTO_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]: # 可以试试 text-embedding-3-large
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
