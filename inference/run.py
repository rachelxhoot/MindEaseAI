# 导入必要的库
import os
# 设置OpenMP线程数为8,优化CPU并行计算性能
os.environ["OMP_NUM_THREADS"] = "8"
import torch
import time
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# 指定模型加载路径
load_path = "model/qwen2chat_int4"
# 加载低位(int4)量化模型,trust_remote_code=True允许执行模型仓库中的自定义代码
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
# 加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

# 定义输入prompt
prompt = "给我讲一下心理咨询的流程"

# 构建符合模型输入格式的消息列表
messages = [{"role": "user", "content": prompt}]

# 使用推理模式,减少内存使用并提高推理速度
with torch.inference_mode():
    # 应用聊天模板,将消息转换为模型输入格式的文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 将文本转换为模型输入张量,并移至CPU (如果使用GPU,这里应改为.to('cuda'))
    model_inputs = tokenizer([text], return_tensors="pt").to('cpu')

    st = time.time()
    # 生成回答,max_new_tokens限制生成的最大token数
    generated_ids = model.generate(model_inputs.input_ids,
                                   max_new_tokens=512)
    end = time.time()

    # 初始化一个空列表,用于存储处理后的generated_ids
    processed_generated_ids = []

    # 使用zip函数同时遍历model_inputs.input_ids和generated_ids
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
        # 计算输入序列的长度
        input_length = len(input_ids)
        
        # 从output_ids中截取新生成的部分
        # 这是通过切片操作完成的,只保留input_length之后的部分
        new_tokens = output_ids[input_length:]
        
        # 将新生成的token添加到处理后的列表中
        processed_generated_ids.append(new_tokens)

    # 将处理后的列表赋值回generated_ids
    generated_ids = processed_generated_ids

    # 解码模型输出,转换为可读文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 打印推理时间
    print(f'Inference time: {end-st:.2f} s')
    # 打印原始prompt
    print('-'*20, 'Prompt', '-'*20)
    print(text)
    # 打印模型生成的输出
    print('-'*20, 'Output', '-'*20)
    print(response)