import gradio as gr
import time
import os
from transformers import AutoTokenizer, TextIteratorStreamer
from ipex_llm.transformers import AutoModelForCausalLM
import torch
from threading import Thread, Event

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "8"  # 设置OpenMP线程数为8,用于控制并行计算

# 加载模型和tokenizer
load_path = "model/qwen2chat_int4"  # 模型路径
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)  # 加载低位模型
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)  # 加载对应的tokenizer

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用
model = model.to(device)  # 将模型移动到选定的设备上

# 创建 TextIteratorStreamer，用于流式生成文本
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 创建一个停止事件，用于控制生成过程的中断
stop_event = Event()

# 定义用户输入处理函数
def user(user_message, history):
    return "", history + [[user_message, None]]  # 返回空字符串和更新后的历史记录

# 定义机器人回复生成函数
def bot(history):
    stop_event.clear()  # 重置停止事件
    prompt = history[-1][0]  # 获取最新的用户输入
    messages = [{"role": "user", "content": prompt}]  # 构建消息格式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 应用聊天模板
    model_inputs = tokenizer([text], return_tensors="pt").to(device)  # 对输入进行编码并移到指定设备
    
    print(f"\n用户输入: {prompt}")
    print("模型输出: ", end="", flush=True)
    start_time = time.time()  # 记录开始时间

    # 设置生成参数
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,  # 最大生成512个新token
        do_sample=True,  # 使用采样
        top_p=0.7,  # 使用top-p采样
        temperature=0.95,  # 控制生成的随机性
    )

    # 在新线程中运行模型生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:  # 迭代生成的文本流
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

# 使用Gradio创建Web界面
with gr.Blocks() as demo:
    gr.Markdown("# Qwen 聊天机器人")
    chatbot = gr.Chatbot()  # 聊天界面组件
    msg = gr.Textbox()  # 用户输入文本框
    clear = gr.Button("清除")  # 清除按钮
    stop = gr.Button("停止生成")  # 停止生成按钮

    # 设置用户输入提交后的处理流程
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)  # 清除按钮功能
    stop.click(stop_generation, queue=False)  # 停止生成按钮功能

if __name__ == "__main__":
    print("启动 Gradio 界面...")
    demo.queue()  # 启用队列处理请求
    # 提示用户输入DSW号
    dsw_number = input("请输入DSW号 (例如: 525085)")
    root_path = f"/dsw-{dsw_number}/proxy/7860/"
    demo.launch(root_path=root_path)  # 兼容魔搭情况下的路由