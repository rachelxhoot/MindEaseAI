# MindEaseAI Demo

## Run
魔搭Notebook环境: `ubuntu22.04-py310-torch2.1.2-tf2.14.0-1.14.0`

在 Terminal 执行下面命令行
```bash
git clone https://github.com/rachelxhoot/MindEaseAI.git
cd MindEaseAI
pip install -r requirements.txt
```

### 准备model和vector database （只需执行一次）
（Optional）修改 `config.yaml` 中的 `embedding_model_name` 和 `model_name` 为要运行的模型
```bash
python3 prepare.py
```

### 运行 RAG App
```bash
python3 demo.py # 中间需要手动输入自己的dsw号
```

## Reference：
- [TinyRAG](https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyRAG)