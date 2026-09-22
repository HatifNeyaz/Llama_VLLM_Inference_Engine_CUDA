import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

# We manually format the exact string Llama-3 expects
text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# encode() returns a flat list of integers, not a dictionary
tokens = tokenizer.encode(text, allowed_special="all")

print("\nCopy this exact line into your main.cu:")
print(f"std::vector<int> prompt = {{{', '.join(map(str, tokens))}}};")