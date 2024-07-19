# 导入操作系统模块，用于设置环境变量
import os
# 设置环境变量 OMP_NUM_THREADS 为 8，用于控制 OpenMP 线程数
os.environ["OMP_NUM_THREADS"] = "8"

# 导入时间模块
import time
# 导入 Streamlit 模块，用于创建 Web 应用
import streamlit as st
# 从 transformers 库中导入 AutoTokenizer 类
from transformers import AutoTokenizer
# 从 transformers 库中导入 TextStreamer 类
from transformers import TextStreamer, TextIteratorStreamer
# 从 ipex_llm.transformers 库中导入 AutoModelForCausalLM 类
from ipex_llm.transformers import AutoModelForCausalLM
# 导入 PyTorch 库
import torch
from threading import Thread

# 指定模型路径
load_path = "model/qwen2chat_int4"
# 加载低比特率模型
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
# 从预训练模型中加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

# 定义生成响应函数
def generate_response(messages, message_placeholder):
    # 将用户的提示转换为消息格式
    # messages = [{"role": "user", "content": prompt}]
    # 应用聊天模板并进行 token 化
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    
    # 创建 TextStreamer 对象，跳过提示和特殊标记
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 使用 zip 函数同时遍历 model_inputs.input_ids 和 generated_ids
    generation_kwargs = dict(inputs=model_inputs.input_ids, max_new_tokens=512, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return streamer

# Streamlit 应用部分
# 设置应用标题
st.title("大模型聊天应用")

# 初始化聊天历史，如果不存在则创建一个空列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入部分
if prompt := st.chat_input("你想说点什么?"):
    # 将用户消息添加到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    response  = str()
    # 创建空的占位符用于显示生成的响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 调用模型生成响应
        streamer = generate_response(st.session_state.messages, message_placeholder)
        for text in streamer:
            response += text
            message_placeholder.markdown(response + "▌")
    
        message_placeholder.markdown(response)
    
    # 将助手的响应添加到聊天历史
    st.session_state.messages.append({"role": "assistant", "content": response})
