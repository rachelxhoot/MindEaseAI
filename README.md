# MindEaseAI Demo

## Embedding 向量化模块

- 作用：将文档片段向量化
- 目前实现：基于embedding模型的API对文本进行向量化
- To-do: 接入本地开源模型

## 文档加载和切分
- `read_file_content`支持的文件类型：pdf、md、txt
- `get_chunk` 按 Token 的长度来切分文档，按 `\n` 进行粗切分

## 数据库 + 向量检索

- 向量数据库： 存放文档片段和对应的向量表示
- 检索模块：根据 Query （问题）检索相关的文档片段
- Query：把用户提出的问题向量化，然后去数据库中检索相关的文档片段，最后返回检索到的文档片段

## LLM
- 目前实现：DeepSeek LLM API 调用
- To-do：接入本地开源模型

## Reference：
- [TinyRAG](https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyRAG)