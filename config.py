class Config:
    # 使用的 llm 与 embedding 模型，可手动修改
    model_name="Qwen/Qwen2-1.5B-Instruct" # llm 模型
    embedding_model_name="AI-ModelScope/bge-small-zh-v1.5" # embedding 模型

    # 模型存储路径
    model_path = "./model_int4"
    cache_path = "./model_src"
    tokenizer_path = "./model_int4"
    embedding_model_path = ""

    # 数据库路径
    data_path = "./data/2106.01702v1.pdf"
    persist_dir = "./chroma_db"

    # 模型推理设置
    max_new_tokens = 128
    omp_num_threads="8"