from itertools import chain
from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer


num_proc = 1
num_example = 1000
block_size = 128
text_column_name = 'text'
tokenizer = AutoTokenizer.from_pretrained('solar-25M')


def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output


def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


ds = load_dataset('maywell/korean_textbooks', 'claude_evol', split=f'train[0:{num_example}]')
ds = ds.map(tokenize_function, remove_columns=ds.features)
ds = ds.map(group_texts, batched=True, num_proc=num_proc)
ds.save_to_disk('lm_datasets')
