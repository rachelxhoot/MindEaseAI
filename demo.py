import os
import time
import torch
from typing import Optional
from threading import Thread, Event
import gradio as gr
from transformers import AutoTokenizer, TextIteratorStreamer


from RAG.utils import Config

from RAG.VectorBase import load_vector_database
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery

from ipex_llm.transformers import AutoModelForCausalLM


config = Config()

os.environ["OMP_NUM_THREADS"] = config.get("omp_num_threads")

# 设置嵌入模型
embed_model = HuggingFaceEmbedding(model_name=config.get("embedding_model_path"))

# 设置语言模型
# llm = setup_local_llm(config)

# 加载向量数据库
vector_store = load_vector_database(persist_dir=config.get("persist_dir"))

##############################rag start#########################

# 加载模型和tokenizer
load_path = config.get("model_path")  # 模型路径
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
    
    
    ######################rag query#################################
    # 设置查询
    # config.get("question") = prompt
    query_str = prompt
    query_embedding = embed_model.get_query_embedding(prompt)

    # 执行向量存储检索
    print("开始执行向量存储检索")
    query_mode = "default"
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )
    query_result = vector_store.query(vector_store_query)

    # 处理查询结果
    print("开始处理检索结果")
    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))
        
    
    # Convert retrieved documents into context strings
    context_strings = [node.node.text for node in nodes_with_scores]
    context_scores = [node.score for node in nodes_with_scores]
    
    # Cutoff of the score/Hyperparameter, 
    # If the score is small, the query_result has no meaning for the question.
    # TODO: advanced tricks in 
    # https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/
    if context_scores[0] < 0.3:
        context_strings = [""]
    # Concatenate context strings to form a single context string for the LLM
    context_string = "\n".join(context_strings)
    
    print("Scores list")
    print(context_scores)
    '''
    # 设置检索器
    retriever = rag.VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=1
    )

    print(f"Query engine created with retriever: {type(retriever).__name__}")
    print(f"Query string length: {len(query_str)}")
    print(f"Query string: {query_str}")

    # 创建查询引擎

    print("准备与llm对话")
    # synth = get_response_synthesizer(streaming=True,llm=llm)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    # 执行查询
    print("开始RAG最后生成")
    start_time_rag = time.time()
    response_rag = query_engine.query(query_str)
    print(f"\n\nRAG最后生成完成，用时: {end_time - start_time_rag:.2f} 秒")
    print(str(response_rag))
    '''
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
            {"role": "user", "content": f"{context_string}\n 你是一个心理咨询AI助手, 请根据前面的专业知识, 回应下面用户的消息: {prompt}"}
        ])
    print(messages) 
    #     messages = [{"role": "user", "content": prompt}]  # 构建消息格式
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
    gr.Markdown("# MindEaseAI Chatbot")
    chatbot = gr.Chatbot(label="AI 心理咨询助手", show_label=True)  # 聊天界面组件
    msg = gr.Textbox(placeholder="请输入您的问题或感受...", label="您的消息")  # 用户输入文本框
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
