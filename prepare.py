from RAG.utils import download_and_quantize_model, Config, download_embedding_model, download_4bit_quantized_model

config = Config()
# 清空之前的模型和数据
config.initialize()

# 准备对话模型
download_and_quantize_model(config.get("model_name"), config.get("cache_path"), config.get("model_path"))
# 或者直接下载已经量化好的gptq模型或awq模型，模型选择请参照模型文档
# download_4bit_quantized_model(config.get("model_name"), config.get("cache_path"))
# 准备embedding模型
download_embedding_model(config.get("embedding_model_name"), config.get("embedding_model_path"))
