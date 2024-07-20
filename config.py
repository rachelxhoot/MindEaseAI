class Config:
    model_name="Qwen/Qwen2-1.5B-Instruct"
    embedding_model_name="AI-ModelScope/bge-small-zh-v1.5"
    model_path = "./qwen2chat_int4"
    cache_path = "./qwen2chat_src"
    tokenizer_path = "./qwen2chat_int4"
    data_path = "./data/2106.01702v1.pdf"
    persist_dir = "./chroma_db"
    embedding_model_path = "./qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 128
    omp_num_threads="8"