import os
import sys
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct", local_files_only=False)

if len(sys.argv) > 1:
    # Read all arguments as a list of integers
    token_ids = [int(arg) for arg in sys.argv[1:]]
    print(tokenizer.decode(token_ids, skip_special_tokens=True))