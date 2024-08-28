'''
Chat Engine is a multi-step chat mode built on top of a retriever over your data.
Ref:https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/
'''

import os
import time
from datetime import datetime

from typing import Optional
from threading import Thread, Event
import gradio as gr



from RAG.utils import Config
from RAG.LLM import setup_local_llm

from RAG.VectorBase import load_vector_database, VectorDBRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.storage.chat_store import SimpleChatStore


config = Config()

os.environ["OMP_NUM_THREADS"] = config.get("omp_num_threads")

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name=config.get("embedding_model_path"))

# 设置语言模型
llm = setup_local_llm(config)

# 加载向量数据库
persist_dir = config.get("persist_dir")
vector_store = load_vector_database(persist_dir, "load")

# store memory
# https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/

chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3900,
    chat_store=chat_store,)
# vector_store[type]
# index = VectorStoreIndex.from_vector_store(embed_model=embed_model, vector_store=vector_store)


# 创建一个停止事件，用于控制生成过程的中断
stop_event = Event()

# 定义用户输入处理函数
def user(user_message, history):
    return "", history + [[user_message, None]]  # 返回空字符串和更新后的历史记录

# 定义机器人回复生成函数
def bot(history, type):
    stop_event.clear()  # 重置停止事件
    prompt = history[-1][0]  # 获取最新的用户输入
    
    config.question = prompt
    query_str = prompt
    
    print(f"Query engine created with retriever: {type}")
    print(f"Query string length: {len(query_str)}")
    print(f"Query string: {query_str}")

    ######################chat_engine#################################

    # 设置检索器
    retriever = VectorDBRetriever(
        vector_store[type], embed_model, query_mode="default", similarity_top_k=1
    )

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm, streaming=True)

    custom_prompt = PromptTemplate(
        """\
你是一个心理咨询AI助手, 你的目标是帮助用户舒服地分享他们的想法和情感。你的回答应该是同情、鼓励和支持的，同时保持温暖和温和的语气。

要求：你的回复应该是简单温和的一句话，至少包括2个部分
(1) 承认他们的情绪并确认他们的感受
(2) 引导来访者继续阐述他们的感受和体验 (必须有这一步)

示例：

用户：我最近一直感到很焦虑。

回应：听到你感到焦虑，我很抱歉。焦虑确实让人很难受。你觉得是什么原因让你感到特别焦虑呢？我在这里聆听你。

<对话记录>
{chat_history}

<专业知识>
{question}

<用户问题>

"""
    )
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=custom_prompt,
        chat_history=memory.get_all(),
        memory=memory,
        verbose=True,
        llm=llm,
    )
    ######################chat_engine end#################################


    print(f"\n用户输入: {prompt}")
    print("模型输出: ", end="", flush=True)
    start_time = time.time()  # 记录开始时间

    response_stream = chat_engine.stream_chat(prompt)


    generated_text = ""
    for new_text in  response_stream.response_gen:  # 迭代生成的文本流
        if stop_event.is_set():  # 检查是否需要停止生成
            print("\n生成被用户停止")
            break
        generated_text += new_text
        print(new_text, end="", flush=True)
        history[-1][1] = generated_text  # 更新历史记录中的回复
        yield history  # 逐步返回更新的历史记录

    end_time = time.time()
    print(f"\n\n生成完成，用时: {end_time - start_time:.2f} 秒")
    
    
    

# 定义停止生成函数
def stop_generation():
    stop_event.set()  # 设置停止事件
    
    
def save_memory_json():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_store.persist(persist_path=f"chat_store/{date_str}.json")
    print(memory.to_string().encode('utf-8').decode('unicode_escape'))

# 使用Gradio创建Web界面
with gr.Blocks() as demo:
    gr.Markdown("# MindEaseAI Chatbot")
    chatbot = gr.Chatbot(label="AI 心理咨询助手", show_label=True)  # 聊天界面组件
    type_selector = gr.Dropdown(
        choices=[("家庭关系", "type1"), ("个人成长与心理", "type2"), ("人际关系与社会适应", "type3"), ("职业与学习", "type4")],
        label="选择类型"
    )
    msg = gr.Textbox(placeholder="请输入您的问题或感受...", label="您的消息")  # 用户输入文本框
    clear = gr.Button("清除")  # 清除按钮
    stop = gr.Button("停止生成")  # 停止生成按钮
    save_memory = gr.Button("保存记录")

    # 设置用户输入提交后的处理流程
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, type_selector], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)  # 清除按钮功能
    stop.click(stop_generation, queue=False)  # 停止生成按钮功能
    save_memory.click(save_memory_json, queue=False)

if __name__ == "__main__":
    print("启动 Gradio 界面...")
    demo.queue()  # 启用队列处理请求
    # 提示用户输入DSW号
    dsw_number = input("请输入DSW号 (例如: 525085)")
    root_path = f"/dsw-{dsw_number}/proxy/7860/"
    demo.launch(root_path=root_path)  # 兼容魔搭情况下的路由
