# 设置OpenMP线程数为8
import os
os.environ["OMP_NUM_THREADS"] = "8"

import time
from transformers import AutoTokenizer
from transformers import TextStreamer

# 导入Intel扩展的Transformers模型
from ipex_llm.transformers import AutoModelForCausalLM
import torch

# 加载模型路径
load_path = "model/qwen2chat_int4"

# 加载4位量化的模型
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)

# 加载对应的tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

# 创建文本流式输出器
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 设置提示词
prompt = "给我讲一下心理咨询流程"

# 构建消息列表
messages = [{"role": "user", "content": prompt}]
    
# 使用推理模式
with torch.inference_mode():

    # 应用聊天模板,添加生成提示
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 对输入文本进行编码
    model_inputs = tokenizer([text], return_tensors="pt")
    
    print("start generate")
    st = time.time()  # 记录开始时间
    
    # 生成文本
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,  # 最大生成512个新token
        streamer=streamer,   # 使用流式输出
    )
    
    end = time.time()  # 记录结束时间
    
    # 打印推理时间
    print(f'Inference time: {end-st} s')