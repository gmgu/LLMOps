import torch
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer

save_name = 'solar-25M'

tokenizer = AutoTokenizer.from_pretrained('upstage/SOLAR-10.7B-v1.0')
config = LlamaConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4)
model = AutoModelForCausalLM.from_config(config=config)

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params / 1000**2} M')

tokenizer.save_pretrained(save_name)
model.save_pretrained(save_name)
