# MindEaseAI Demo

## Run
ecs环境: `ubuntu22.04`

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

### 准备文件运行gradio
```bash
mkdir -p /var/www/gradio_app
chown root /var/www/gradio_app
chmod -R 755 /var/www/gradio_app
```

### 运行 RAG App
```bash
python3 demo_chat_engine.py 
```

### 如果ecs无法打开local path
在阿里云打开7860端口（gradio 默认端口）
```bash
ssh -L 7860:localhost:7860 username@your_ecs_public_ip
```
成功后本地浏览器中输入 http://localhost:7860/ 即可访问Web界面。

## Reference：
- [TinyRAG](https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyRAG)
- [PsyQA](https://github.com/thu-coai/PsyQA)
