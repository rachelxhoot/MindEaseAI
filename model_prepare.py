import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
import argparse

def download_and_quantize_model(model_name, cache_dir, revision='master', quantize_dir='quantized_model'):
    """
    下载模型，进行量化并保存量化后的模型和分词器。

    参数:
    - model_name: 模型的名称。
    - cache_dir: 下载模型的缓存目录。
    - revision: 模型的版本号，默认为 'master'。
    - quantize_dir: 量化模型和分词器的保存目录。
    """
    # 下载模型
    model_dir = snapshot_download(model_name, cache_dir=os.path.join(cache_dir), revision=revision)
    print(f"模型下载到: {model_dir}")

    # 加载模型和分词器
    model_path = os.path.join(cache_dir, model_name.replace('.', '___'))
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"模型和分词器加载完成，路径: {model_path}")

    # 量化模型并保存
    model.save_low_bit(quantize_dir)
    tokenizer.save_pretrained(quantize_dir)
    print(f"量化后的模型和分词器已保存到: {quantize_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', action='store_true', help="Download only the specified embedding model if this flag is present")

    # 解析命令行参数
    args = parser.parse_args()

    # 定义模型名称、缓存目录和量化目录
    embedding_model = 'AI-ModelScope/bge-small-zh-v1.5'
    model_name = 'Qwen/Qwen2-1.5B-Instruct'
    cache_dir = 'model/qwen2chat_src'
    quantize_dir = 'model/qwen2chat_int4'

    # 确保目录存在
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(quantize_dir, exist_ok=True)

    if args.embedding:
        model_dir = snapshot_download(embedding_model, cache_dir=cache_dir, revision='master')
    else:
        download_and_quantize_model(model_name, cache_dir, revision='master', quantize_dir=quantize_dir)