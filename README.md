# TinyVLLM 1660Ti

A bare-metal, custom CUDA inference engine built specifically to run Llama-3 architectures (specifically the ~1B parameter variant) locally. 

## The Objective
This project demonstrates how large language model inference works under the hood. Instead of relying on massive frameworks like PyTorch or Hugging Face for the heavy lifting, this code manually allocates memory, loads `.safetensors` weights directly into the GPU, and executes the autoregressive generation loop using custom CUDA kernels and cuBLAS. It is designed for inferencing and serves as a highly educational breakdown of the Transformer architecture operating at the hardware level.

## Requirements

**Hardware & Compilers:**
* **NVIDIA GPU:** Currently hardcoded for Turing architecture (GTX 1660 Ti / `sm_75`). *Note: If you have a different GPU, you must change `set(CMAKE_CUDA_ARCHITECTURES "75")` in the `CMakeLists.txt` to match your architecture.*
* **NVIDIA CUDA Toolkit:** Required for `nvcc` and the `cuBLAS` library.
* **C++ Compiler:** Must support C++17.

**Dependencies:**
* **nlohmann/json:** A single-header C++ JSON library. You must place `json.hpp` in the project root or an `include/` directory to parse the `.safetensors` header.
* **Python 3.x:** Required for pre-processing the prompt and post-processing the output.
* **Hugging Face Transformers:** Python package required for the tokenizer (`pip install transformers`).
* **Model Weights:** A Llama-3 model (e.g., Llama-3.2-1B-Instruct) downloaded in `.safetensors` format and placed in the parent directory as `model.safetensors`.

## How It Works

This project splits the LLM pipeline into three distinct phases: prep, execution, and decoding. Because writing a text tokenizer in C++ is highly complex, Python handles the text-to-token translation, while the raw CUDA C++ handles the computationally heavy matrix math.

### Simplified Workflow Diagram

```text
1. PREP (Python)                   2. INFERENCE (CUDA C++)                 3. DECODE (Python)
                                                                           
[User Prompt]                      [model.safetensors]                     
      │                                     │                              
 (quick.py)                                 ▼                              
      │                         ┌───────────────────────┐                  
      ▼                         │      main.cu          │                  
[Hardcoded Token IDs] ─────────▶│                       │                  
                                │  • Embedding Lookup   │                  
                                │  • KV Cache Alloc     │                  
                                │  • 16x Layers Loop:   │                  
                                │    - RMSNorm          │                  
                                │    - QKV Matmuls      │                  
                                │    - RoPE             │                  
                                │    - Attention        │                  
                                │    - MLP (SiLU)       │                  
                                │  • LM Head (Argmax)   │                  
                                │                       │                  
                                └───────────┬───────────┘                  
                                            │                              
                                            ▼                              
                                  [Generated Token IDs] ──────────────────▶ (tokenizer.py)
                                                                                  │
                                                                                  ▼
                                                                        [Final Text Output]






## System Components

* **`quick.py`**: A helper script. You feed it your desired text prompt, and it formats it using Llama-3's specific chat templates (`<|start_header_id|>`, etc.). It outputs a C++ array of integer token IDs that you manually copy and paste into `main.cu`.
* **`main.cu`**: The core engine.
  * It parses the JSON header of `model.safetensors` to find the exact byte offsets of the model's weights.
  * It maps those weights (Embeddings, Q/K/V/O projections, MLP gates, RMSNorms) directly to GPU memory pointers in `__nv_bfloat16` format.
  * It statically allocates a KV Cache for a maximum context window of 1024 tokens.
  * It runs a token-by-token generation loop up to a hardcoded 50 tokens, utilizing custom kernels for operations like Piecewise RoPE and Softmax.
  * Once finished, it triggers a system call to Python for decoding.
* **`tokenizer.py`**: Called automatically by the C++ executable at the very end. It takes the newly generated integer tokens, converts them back into human-readable text using the Hugging Face tokenizer, and prints the final response.