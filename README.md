# TinyVLLM: Bare-Metal Llama-3 Inference

A lightweight, custom C++/CUDA inference engine designed to run Llama-3 models locally. This specific implementation is tailored for NVIDIA GTX 1660 Ti hardware (Turing architecture, `sm_75`).

## The Objective
The goal of this project is to perform LLM inference from scratch without relying on massive machine learning frameworks like PyTorch or TensorFlow. By directly writing custom CUDA kernels and leveraging the cuBLAS library for matrix multiplications, the code loads SafeTensors model weights, manages the KV Cache, and executes the Llama-3 transformer blocks directly on the GPU. This provides a deep, under-the-hood look at how autoregressive text generation actually works at the hardware level.

## Requirements
To build and run this project, you need the following tools and libraries:

**C++ & CUDA Environment:**
*   **NVIDIA CUDA Toolkit:** Requires `nvcc` and the `cuBLAS` library.
*   **CMake:** Version 3.18 or higher.
*   **C++ Compiler:** Must support C++17 (e.g., GCC, Clang, or MSVC).
*   **Nlohmann JSON:** The single-header `json.hpp` file for parsing the SafeTensors metadata.

**Python Environment (for Tokenization):**
*   **Python 3.x**
*   **HuggingFace Transformers:** `pip install transformers`

**Model Weights:**
*   A Llama-3 `.safetensors` file (e.g., Llama-3.2-1B-Instruct) placed in the parent directory (`../model.safetensors`).

## How It Works

The engine runs a classic autoregressive generation loop. It takes a sequence of token IDs, maps them to vector embeddings, passes them through 16 transformer layers, and calculates the probabilities (logits) for the next word. It uses a "greedy decoding" approach—picking the most likely next token, appending it to the sequence, and repeating the process.

### Simplified Workflow Diagram

```text
[Input String] 
      │
      ▼
(quick.py) Tokenizer ──► [Token IDs] (e.g., 128000, 271, 3923...)
                              │
                              ▼
                        [ GPU MEMORY ]
                              │
  ┌───────────────────────────┴───────────────────────────┐
  │ 1. Embedding Gather: Map Token IDs to Hidden States   │
  │ 2. For each of the 16 Transformer Layers:             │
  │    ├─ RMSNorm                                         │
  │    ├─ Self-Attention (Q, K, V Projections)            │
  │    ├─ RoPE (Rotary Positional Embeddings)             │
  │    ├─ Update KV Cache                                 │
  │    ├─ Softmax & Context Projection                    │
  │    ├─ Residual Addition                               │
  │    └─ MLP (Gate, Up, Down Projections with SiLU)      │
  │ 3. Final RMSNorm                                      │
  │ 4. LM Head (Logits Calculation)                       │
  └───────────────────────────┬───────────────────────────┘
                              │
                              ▼
                  Argmax (Find highest probability)
                              │
                              ▼
                      [New Token ID] ────(Loop back to GPU)
                              │
                              ▼
(tokenizer.py) HuggingFace Decoder ──► [Human Readable Text]
```




# Steps to Reproduce
* 1. Project Setup
Ensure your directory structure looks like this:

```text
├── model.safetensors      # Your downloaded Llama-3 weights
├── tokenizer.py           # Output decoder script
├── quick.py               # Input encoder script
└── cpp_project/
    ├── CMakeLists.txt
    ├── main.cu
    └── include/
        └── json.hpp       # Downloaded from nlohmann/json GitHub
```

* 2. Generate the Input Prompt
Before compiling, you can customize your prompt using the Python helper script.

Bash
python quick.py
This will output a C++ vector string. Copy that output and replace the std::vector<int> prompt = {...}; line in main.cu if you want a custom prompt.

* 3. Build the CUDA Engine
Navigate to your C++ project directory and build the executable using CMake.

Note: The CMakeLists.txt is currently hardcoded for the GTX 1660 Ti (set(CMAKE_CUDA_ARCHITECTURES "75")). If you are using a different GPU, change "75" to match your GPU's architecture (e.g., "80" for Ampere/RTX 30-series, "89" for Ada/RTX 40-series).

Bash
mkdir build
cd build
cmake ..
make
4. Run Inference
Execute the compiled binary. The program will map the model weights to VRAM, allocate the KV Cache, execute the generation loop for 50 tokens, and automatically call the Python script to decode the output.

Bash
./tiny_vllm