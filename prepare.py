from RAG.utils import download_and_quantize_model, download_embedding_model
from config import Config

config = Config()

# 准备模型
download_and_quantize_model(config.model_name, config.cache_path, config.model_path)
download_embedding_model(config.embedding_model_name, config.cache_path)