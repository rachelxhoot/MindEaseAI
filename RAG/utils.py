import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re
import platform
if platform.machine() != "arm64":
    from modelscope import snapshot_download, AutoModel
    from ipex_llm.transformers import AutoModelForCausalLM
else:
    print("Running on Apple Silicon, skipping import ipex.")
from transformers import AutoTokenizer

from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

enc = tiktoken.get_encoding("cl100k_base")

import yaml
import shutil

class Config:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml')
        self.config_data = self.load_config()

    def load_config(self):
        """从文件中加载配置数据"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        return {}

    def save_config(self):
        """将配置数据保存到文件"""
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config_data, file, default_flow_style=False)

    def get(self, key, default=None):
        """获取配置项的值"""
        return self.config_data.get(key, default)

    def set(self, key, value):
        """设置配置项的值"""
        self.config_data[key] = value
        self.save_config()  # 更新配置后立即保存

    def __str__(self):
        """返回配置数据的字符串表示，方便打印和调试"""
        return yaml.dump(self.config_data, default_flow_style=False)
    
    def initialize(self):
        paths = []
        persist_dir = self.get('persist_dir')
        paths.append(persist_dir)
        cache_path = self.get('cache_path')
        paths.append(cache_path)
        model_path = self.get('model_path')
        paths.append(model_path)
    
        """删除指定目录下的所有文件和子目录"""
        for path in paths:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(path)
            else:
                print(f"The path {path} is not a directory or does not exist.")


def download_and_quantize_model(model_name, cache_dir, quantize_dir, revision='master', low_bit="sym_int4"):
    """
    Download, quantize, and save the model and tokenizer.

    Parameters:
    - model_name: Name of the model.
    - cache_dir: Directory to cache the downloaded model.
    - quantize_dir: Directory to save the quantized model and tokenizer.
    - revision: Model version, default is 'master'.
    """
    model_dir = snapshot_download(model_name, cache_dir=os.path.join(cache_dir), revision=revision)
    print(f"Model downloaded to: {model_dir}")

    model_path = os.path.join(cache_dir, model_name.replace('.', '___'))
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit=low_bit, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model and tokenizer loaded from: {model_path}")

    model.save_low_bit(quantize_dir)
    tokenizer.save_pretrained(quantize_dir)
    print(f"Quantized model and tokenizer saved to: {quantize_dir}")

def download_embedding_model(embedding_model, cache_dir):
    embedding_model_dir = snapshot_download(embedding_model, cache_dir=cache_dir, revision='master')
    # 更新配置数据
    config = Config()
    config.set('embedding_model_path', embedding_model_dir)
    print(f"Embedding model downloaded to: {embedding_model_dir}")

# use:
# download_and_quantize_model("Qwen/Qwen2-1.5B-Instruct", "qwen2chat_src", "qwen2chat_int4")
# download_embedding_model("AI-ModelScope/bge-small-zh-v1.5", "qwen2chat_src")

def load_data_json(data_path):
    # 读取 JSON 文件
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 构建文档字符串
    documents = []
    for item in data:
        question = item.get("question", "")
        description = item.get("description", "")
        answers = " ".join(answer.get("answer_text", "") for answer in item.get("answers", []))
        text_block = f"Question: {question}\nDescription: {description}\nAnswers: {answers}"
        documents.append(text_block)

    # 分割文档字符串为更小的文本块
    text_parser = SentenceSplitter(chunk_size=384)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    # 创建 TextNode 节点
    nodes = [TextNode(text=text_chunk) for text_chunk in text_chunks]

    return nodes

def load_data_pdf(data_path: str) -> List[TextNode]:
    """
    加载并处理PDF数据
    
    Args:
        data_path (str): PDF文件路径
    
    Returns:
        List[TextNode]: 处理后的文本节点列表
    """
    loader = PyMuPDFReader()
    documents = loader.load(file_path=data_path)

    
    #*****************
    #Hyperparameter  of the document splitter: chunk_size=384
    #************
    text_parser = SentenceSplitter(chunk_size=384)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes

def completion_to_prompt(completion: str) -> str:
    """
    将完成转换为提示格式
    
    Args:
        completion (str): 完成的文本
    
    Returns:
        str: 格式化后的提示
    """
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

def messages_to_prompt(messages: List[dict]) -> str:
    """
    将消息列表转换为提示格式
    
    Args:
        messages (List[dict]): 消息列表
    
    Returns:
        str: 格式化后的提示
    """
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    prompt = prompt + "<|assistant|>\n"

    return prompt
