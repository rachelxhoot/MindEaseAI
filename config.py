class Config:
    model_path = "./qwen2chat_int4"
    tokenizer_path = "./qwen2chat_int4"
    question = "How does Llama 2 perform compared to other open-source models?"
    data_path = "./data/llamatiny.pdf"
    persist_dir = "./chroma_db"
    embedding_model_path = "./qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 128