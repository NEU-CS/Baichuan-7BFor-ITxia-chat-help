# 模型来源
该模型由百川智能Baichuan-7b-chat模型fine-tune而来，仓库里只有Lora adapter weights

# 数据集来源
数据集来自南京大学IT侠帮助群，数据截取了2023年至今的问答  

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