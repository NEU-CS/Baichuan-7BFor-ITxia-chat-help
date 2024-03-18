# 模型来源
该模型由百川智能Baichuan-7b-chat模型fine-tune而来，仓库里只有Lora adapter weights

# 数据集来源
数据集来自南京大学IT侠帮助QQ群，数据截取了2023年至2024-3月的问答  
通过导出QQ群聊消息记录，使用GPT-4帮助数据清洗，后续使用正则表达式进一步清洗  
目前数据集还比较小，清洗后有160条问答  

# 模型任务
模型能够做一些关于南京大学内部的IT问题，如"电脑如何清灰"，"如何连接校园网等"  
注意：问题前需要加入符号"#"表示一个问题的开始  

# Direct Use

```Python

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModelForCausalLM
basemodel = "baichuan-inc/Baichuan2-7B-Chat"
tokenzier = AutoTokenizer.from_pretrained(basemodel,model_max_length = 512,trust_remote_code=True)
tokenzier.pad_token = tokenzier.eos_token
tokenzier.padding_side = "right"
babcfig = BitsAndBytesConfig(load_in_4bit=True,llm_int4_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(basemodel,
        device_map = "cuda:0",
        quantization_config = babcfig,
        trust_remote_code=True)
model = PeftModelForCausalLM.from_pretrained(model,"Baichuan-7b-for-ITxia-chat-help")
messages = []
messages.append({"role": "user", "content": "#请问一下鼠标垫脏了怎么办？"})
model.chat(tokenzier,messages)
'鼠标垫脏了的话，可以尝试用湿布擦拭或者用软刷子刷洗。如果鼠标垫材质比较特殊的话，建议先查阅一下说明书或官方说明，看看是否有特定的清洁方法。如果是可拆卸的鼠标垫，可以先拆开来清洗，然后再按照原样组装回去。...'

```
