import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = 'output_dir'
#model_path = 'solar-25M'
model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = '안녕하세'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.95, temperature=0.1)
text = tokenizer.decode(outputs[0])
print(text)
