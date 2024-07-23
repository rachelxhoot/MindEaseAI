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
from modelscope import snapshot_download, AutoModel
from ipex_llm.transformers import AutoModelForCausalLM
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
            if os.mkdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            # 确认目录为空后删除目录本身
            os.rmdir(path)


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

def load_data(data_path: str) -> List[TextNode]:
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

# # Reference:
# class ReadFiles:
#     """
#     Class to read files.
#     """

#     def __init__(self, path: str) -> None:
#         self._path = path
#         self.file_list = self.get_files()

#     def get_files(self):
#         # args：dir_path，目标文件夹路径
#         file_list = []
#         for filepath, dirnames, filenames in os.walk(self._path):
#             # os.walk 函数将递归遍历指定文件夹
#             for filename in filenames:
#                 # 通过后缀名判断文件类型是否满足要求
#                 if filename.endswith(".md"):
#                     # 如果满足要求，将其绝对路径加入到结果列表
#                     file_list.append(os.path.join(filepath, filename))
#                 elif filename.endswith(".txt"):
#                     file_list.append(os.path.join(filepath, filename))
#                 elif filename.endswith(".pdf"):
#                     file_list.append(os.path.join(filepath, filename))
#         return file_list

#     def get_content(self, max_token_len: int = 600, cover_content: int = 150):
#         docs = []
#         # 读取文件内容
#         for file in self.file_list:
#             content = self.read_file_content(file)
#             chunk_content = self.get_chunk(
#                 content, max_token_len=max_token_len, cover_content=cover_content)
#             docs.extend(chunk_content)
#         return docs

#     @classmethod
#     def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
#         chunk_text = []

#         curr_len = 0
#         curr_chunk = ''

#         token_len = max_token_len - cover_content
#         lines = text.splitlines()  # 假设以换行符分割文本为行

#         for line in lines:
#             line = line.replace(' ', '')
#             line_len = len(enc.encode(line))
#             if line_len > max_token_len:
#                 # 如果单行长度就超过限制，则将其分割成多个块
#                 num_chunks = (line_len + token_len - 1) // token_len
#                 for i in range(num_chunks):
#                     start = i * token_len
#                     end = start + token_len
#                     # 避免跨单词分割
#                     while not line[start:end].rstrip().isspace():
#                         start += 1
#                         end += 1
#                         if start >= line_len:
#                             break
#                     curr_chunk = curr_chunk[-cover_content:] + line[start:end]
#                     chunk_text.append(curr_chunk)
#                 # 处理最后一个块
#                 start = (num_chunks - 1) * token_len
#                 curr_chunk = curr_chunk[-cover_content:] + line[start:end]
#                 chunk_text.append(curr_chunk)
                
#             if curr_len + line_len <= token_len:
#                 curr_chunk += line
#                 curr_chunk += '\n'
#                 curr_len += line_len
#                 curr_len += 1
#             else:
#                 chunk_text.append(curr_chunk)
#                 curr_chunk = curr_chunk[-cover_content:]+line
#                 curr_len = line_len + cover_content

#         if curr_chunk:
#             chunk_text.append(curr_chunk)

#         return chunk_text

#     @classmethod
#     def read_file_content(cls, file_path: str):
#         # 根据文件扩展名选择读取方法
#         if file_path.endswith('.pdf'):
#             return cls.read_pdf(file_path)
#         elif file_path.endswith('.md'):
#             return cls.read_markdown(file_path)
#         elif file_path.endswith('.txt'):
#             return cls.read_text(file_path)
#         else:
#             raise ValueError("Unsupported file type")

#     @classmethod
#     def read_pdf(cls, file_path: str):
#         # 读取PDF文件
#         with open(file_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page_num in range(len(reader.pages)):
#                 text += reader.pages[page_num].extract_text()
#             return text

#     @classmethod
#     def read_markdown(cls, file_path: str):
#         # 读取Markdown文件
#         with open(file_path, 'r', encoding='utf-8') as file:
#             md_text = file.read()
#             html_text = markdown.markdown(md_text)
#             # 使用BeautifulSoup从HTML中提取纯文本
#             soup = BeautifulSoup(html_text, 'html.parser')
#             plain_text = soup.get_text()
#             # 使用正则表达式移除网址链接
#             text = re.sub(r'http\S+', '', plain_text) 
#             return text

#     @classmethod
#     def read_text(cls, file_path: str):
#         # 读取文本文件
#         with open(file_path, 'r', encoding='utf-8') as file:
#             return file.read()


# class Documents:
#     """
#     Class to get categorized JSON format documents.
#     """
#     def __init__(self, path: str = '') -> None:
#         self.path = path
    
#     def get_content(self):
#         with open(self.path, mode='r', encoding='utf-8') as f:
#             content = json.load(f)
#         return content