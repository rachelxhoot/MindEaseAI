# Baseline

## 使用 ipex-llm 推理大语言模型
### 安装环境

`bash install.sh`

`conda activate ipex`
### 模型准备
`python3 model_prepare.py`

### 模型运行推理

- `python3 inference/run.py`
- 流式输出: `python3 inference/run_stream.py`

## 简单的 App 搭建
### Gradio

`pip install gradio`

`python3 app/run_gradio_stream.py` (根据提示输入dsw号即可，不需修改源码)

### Streamlit
`pip install streamlit`
`streamlit run app/run_streamlit_stream.py`

## 基于 Llamaindex 构建一个简易的 RAG 系统

### 环境准备

`conda activate ipex`

`pip install PyMuPDF llama-index-vector-stores-chroma llama-index-readers-file llama-index-embeddings-huggingface llama-index`

### embedding 模型准备
`python3 model_prepare.py --embedding`

### run rag

data path: `rag/data/`，需修改 `run_rag.py -> Config`中对应的参数

`python3 rag/run_rag.pys`