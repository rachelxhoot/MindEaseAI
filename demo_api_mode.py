import time

from typing import Optional
from threading import Thread, Event
import gradio as gr
from transformers import AutoTokenizer, TextIteratorStreamer
from openai import OpenAI

from RAG.utils import Config

from RAG.VectorBase import load_vector_database, VectorDBRetriever
from RAG.LLM import APIChat
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import VectorStoreIndex
import os

config = Config()
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# loads BAAI/bge-small-en
# embed_model = HuggingFaceEmbedding()

# loads BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

persist_dir = config.get("persist_dir")
vector_store = load_vector_database(persist_dir, "load")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
##############################rag start#########################

# 加载模型和tokenizer
from dotenv import load_dotenv
load_dotenv(".env", override=True)  # 自动从 .env 文件加载环境变量
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

def single_response(prompt):
    from openai import OpenAI
    client = OpenAI(
        api_key = api_key,
        base_url = base_url
    )
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": str(prompt),
                }
            ],
            model="deepseek-chat",
        )
        
        # Debug print statements
        print("API Response:", chat_completion)
        
        if chat_completion.choices:
            return chat_completion.choices[0].message.content
        else:
            return "没有有效的响应"
    except Exception as e:
        return f"发生错误: {str(e)}"


# # 创建 TextIteratorStreamer，用于流式生成文本
# streamer = TextIteratorStreamer(tokenizer=embed_model.tokenizer_name ,skip_prompt=True, skip_special_tokens=True)

# 创建一个停止事件，用于控制生成过程的中断
stop_event = Event()

# 定义用户输入处理函数
def user(user_message, history):
    return "", history + [[user_message, None]]  # 返回空字符串和更新后的历史记录

# 定义机器人回复生成函数
def bot(history, type):
    stop_event.clear()  # 重置停止事件
    prompt = history[-1][0]  # 获取最新的用户输入
    
    ######################rag query################################


    index = VectorStoreIndex.from_vector_store(
        vector_store[type],
        embed_model=embed_model,
    )

    retriever = index.as_retriever()
    nodes = retriever.retrieve(prompt)

    context_strings = [node.text for node in nodes]

    
    

    
    ######################rag query end#################################


    '''
    chat with history
    (tiny llms have some problems with chatting, so try to do muti tests)
    '''
    # TODO: more awesome chat templates
    messages = []
    for user_msg, response in history:
        if user_msg and not response:  # 如果当前正在处理的用户消息没有响应
            break  # 结束循环，只考虑之前的对话
        messages.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ])
        
    messages.extend([
            {"role": "user", "content": 
             f'''{context_strings}\n 
            请根据前面的知识参考, 回应下面用户的消息: {prompt}

            你是一个心理咨询AI助手, 你的目标是帮助用户舒服地分享他们的想法和情感。你的回答应该是同情、鼓励和支持的，同时保持温暖和温和的语气。
            要求：你的回复应该是简单温和的一句话，至少包括2个部分: 承认他们的情绪并确认他们的感受;引导来访者继续阐述他们的感受和体验 (必须有这一步)
            

            示例：
            用户：我最近一直感到很焦虑。
            回应：听到你感到焦虑，我很抱歉。焦虑确实让人很难受。你觉得是什么原因让你感到特别焦虑呢？我在这里聆听你。
            
            
            再次强调：请一定要记得继续引导来访者倾诉！！！
            '''}
        ])
    print(messages) 
    
    print(f"\n用户输入: {prompt}")
    print("模型输出: ", end="", flush=True)
    start_time = time.time()  # 记录开始时间

    # 直接调用 single_response
    response = single_response(messages)
    print(response)
    
    end_time = time.time()
    print(f"\n\n生成完成，用时: {end_time - start_time:.2f} 秒")

    return history + [[prompt, response]]

# 定义停止生成函数
def stop_generation():
    stop_event.set()  # 设置停止事件

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
    
   # 设置用户输入提交后的处理流程
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, type_selector], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)  # 清除按钮功能
    stop.click(lambda: None, None, chatbot, queue=False)  # 停止生成按钮功能

if __name__ == "__main__":
    print("启动 Gradio 界面...")
    demo.queue()  # 启用队列处理请求
    demo.launch()  # 兼容魔搭情况下的路由
