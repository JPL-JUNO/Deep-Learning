from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained(
    "./", trust_remote_code=True, revision='v1.1.0')
model = AutoModel.from_pretrained(
    "./", trust_remote_code=True, revision='v1.1.0').half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "如何学习语言大模型", history=history)
print(response)
