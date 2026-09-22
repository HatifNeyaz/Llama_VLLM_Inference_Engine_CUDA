#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <vector>
#include <cstdlib> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
using json = nlohmann::json;

struct LlamaWeights {
    __nv_bfloat16* embed_tokens;
    __nv_bfloat16* final_norm;
    __nv_bfloat16* lm_head;

    std::vector<__nv_bfloat16*> input_norm;
    std::vector<__nv_bfloat16*> q_proj;
    std::vector<__nv_bfloat16*> k_proj;
    std::vector<__nv_bfloat16*> v_proj;
    std::vector<__nv_bfloat16*> o_proj;           
    std::vector<__nv_bfloat16*> post_attn_norm;   
    std::vector<__nv_bfloat16*> mlp_gate;
    std::vector<__nv_bfloat16*> mlp_up;
    std::vector<__nv_bfloat16*> mlp_down;
};

__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *hidden_state, __nv_bfloat16 *embed_tokens) {
    int workIndex = threadIdx.x + (blockIdx.x * 2048);
    int token_id = gpu_input_tokens[blockIdx.x];
    hidden_state[workIndex] = embed_tokens[(token_id * 2048) + threadIdx.x];
    hidden_state[workIndex + 1024] = embed_tokens[(token_id * 2048) + threadIdx.x + 1024];
}

__global__ void rmsNormKernel(__nv_bfloat16* input, __nv_bfloat16* norm_weight, __nv_bfloat16* output) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    int base_idx = token_idx * 2048;

    float val1 = (float)input[base_idx + tid];
    float val2 = (float)input[base_idx + tid + 1024];
    
    __shared__ float sum_sq[1024];
    sum_sq[tid] = (val1 * val1) + (val2 * val2);
    __syncthreads();

    for (int stride = 512; stride > 0; stride /= 2) {
        if (tid < stride) {
            sum_sq[tid] += sum_sq[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq[0] / 2048.0f;
        inv_rms = 1.0f / sqrtf(mean_sq + 1e-5f); 
    }
    __syncthreads(); 

    float weight1 = (float)norm_weight[tid];
    float weight2 = (float)norm_weight[tid + 1024];

    output[base_idx + tid] = (__nv_bfloat16)((val1 * inv_rms) * weight1);
    output[base_idx + tid + 1024] = (__nv_bfloat16)((val2 * inv_rms) * weight2);
}

// Llama-3 Piecewise RoPE
__global__ void ropeKernel(__nv_bfloat16 *input, int proj_dim, int current_pos) {
    const int HEAD_DIM = 64; 
    const int HALF_DIM = 32;

    if (threadIdx.x < proj_dim / 2) {
        int head_id = threadIdx.x / HALF_DIM;      
        int dim_id = threadIdx.x % HALF_DIM;       
        
        int first_idx = head_id * HEAD_DIM + dim_id;
        int second_idx = first_idx + HALF_DIM;

        float factor = 32.0f;
        float low_freq_factor = 1.0f;
        float high_freq_factor = 4.0f;
        float original_max_position_embeddings = 8192.0f;
        
        float freq = 1.0f / (powf(500000.0f, ((float)(dim_id * 2) / HEAD_DIM)));
        float wavelen = (2.0f * 3.14159265359f) / freq;
        float low_freq_wavelen = original_max_position_embeddings / low_freq_factor;
        float high_freq_wavelen = original_max_position_embeddings / high_freq_factor;
        
        float scaled_freq = freq;
        if (wavelen > low_freq_wavelen) {
            scaled_freq = freq / factor;
        } else if (wavelen >= high_freq_wavelen) {
            float smooth = (original_max_position_embeddings / wavelen - 1.0f) / (high_freq_factor - 1.0f);
            scaled_freq = (1.0f - smooth) * (freq / factor) + smooth * freq;
        }

        float angle = current_pos * scaled_freq;
        float val1 = (float)input[first_idx];
        float val2 = (float)input[second_idx];

        input[first_idx]  = (__nv_bfloat16)(val1 * cosf(angle) - val2 * sinf(angle));
        input[second_idx] = (__nv_bfloat16)(val1 * sinf(angle) + val2 * cosf(angle));
    }
}

__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b) {
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024) {
        float val_a = (float)a[workIndex + i];
        float val_b = (float)b[workIndex + i];
        float silu_out = val_a * (1.0f / (1.0f + expf(-val_a)));
        a[workIndex + i] = (__nv_bfloat16)(silu_out * val_b);
    }
}

__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds) {
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = (__nv_bfloat16)((float)input[workIndex] + (float)input_embeds[workIndex]);
    input[workIndex + 1024] = (__nv_bfloat16)((float)input[workIndex + 1024] + (float)input_embeds[workIndex + 1024]);
}

__global__ void decodeAttentionKernel(__nv_bfloat16* q, __nv_bfloat16* k_cache, float* scores, int seq_len) {
    int head_id = blockIdx.x;           
    int kv_head_id = head_id / 4;       
    int token_idx = threadIdx.x;        

    if (token_idx < seq_len) 
    {
        float score = 0.0f;
        for (int i = 0; i < 64; i++) {
            float q_val = (float)q[head_id * 64 + i];
            float k_val = (float)k_cache[token_idx * 512 + kv_head_id * 64 + i];
            score += q_val * k_val;
        }
        score *= 0.125f; 
        scores[head_id * seq_len + token_idx] = score;
    }
}

__global__ void decodeSoftmaxKernel(float* scores, int seq_len) {
    int head_id = blockIdx.x; 
    if (threadIdx.x == 0) 
    {
        float max_val = -1e20f; 
        for (int i = 0; i < seq_len; i++) {
            max_val = fmaxf(max_val, scores[head_id * seq_len + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            scores[head_id * seq_len + i] = expf(scores[head_id * seq_len + i] - max_val);
            sum += scores[head_id * seq_len + i];
        }
        
        for (int i = 0; i < seq_len; i++) {
            scores[head_id * seq_len + i] /= sum;
        }
    }
}

__global__ void decodeContextKernel(float* scores, __nv_bfloat16* v_cache, __nv_bfloat16* out, int seq_len) {
    int head_id = blockIdx.x;           
    int kv_head_id = head_id / 4;       
    int dim_idx = threadIdx.x;          

    float context = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        float score = scores[head_id * seq_len + t];
        float v_val = (float)v_cache[t * 512 + kv_head_id * 64 + dim_idx];
        context += score * v_val;
    }
    out[head_id * 64 + dim_idx] = (__nv_bfloat16)context;
}

int checkGPUStatus() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) return 1;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    std::cout << "--- GPU STATUS ---\n";
    std::cout << "Device: " << prop.name << "\n";
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free VRAM: " << free_mem / (1024 * 1024) << " MB\n";
    std::cout << "------------------\n\n";
    return 0;
}

void matmul(cublasHandle_t handle, __nv_bfloat16* input, __nv_bfloat16* weight, __nv_bfloat16* output, int num_tokens, int out_dim, int in_dim) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, num_tokens, in_dim, &alpha,
                 weight, CUDA_R_16BF, in_dim, input, CUDA_R_16BF, in_dim, &beta,
                 output, CUDA_R_16BF, out_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void loadModelToGPU(const std::string& file_path, LlamaWeights& weights) {
    std::cout << "Loading model: " << file_path << "\n";
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return;

    std::streamsize total_size = file.tellg();
    file.seekg(0, std::ios::beg);

    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    
    std::string header_json_str(header_size, ' ');
    file.read(&header_json_str[0], header_size);
    json header = json::parse(header_json_str);

    size_t weights_start = 8 + header_size;
    size_t weights_len = total_size - weights_start;

    std::vector<char> cpu_buffer(weights_len);
    file.seekg(weights_start, std::ios::beg);
    file.read(cpu_buffer.data(), weights_len);

    void* gpu_buffer;
    cudaMalloc(&gpu_buffer, weights_len);
    cudaMemcpy(gpu_buffer, cpu_buffer.data(), weights_len, cudaMemcpyHostToDevice);

    weights.embed_tokens = (__nv_bfloat16*)((char*)gpu_buffer + header["model.embed_tokens.weight"]["data_offsets"][0].get<size_t>());
    weights.final_norm = (__nv_bfloat16*)((char*)gpu_buffer + header["model.norm.weight"]["data_offsets"][0].get<size_t>());
    
    if (header.contains("lm_head.weight")) {
        weights.lm_head = (__nv_bfloat16*)((char*)gpu_buffer + header["lm_head.weight"]["data_offsets"][0].get<size_t>());
    } else {
        weights.lm_head = weights.embed_tokens;
    }

    std::cout << "Mapping 16 Transformer Layers...\n";
    for (int i = 0; i < 16; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
        weights.input_norm.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".input_layernorm.weight"]["data_offsets"][0].get<size_t>()));
        weights.q_proj.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".self_attn.q_proj.weight"]["data_offsets"][0].get<size_t>()));
        weights.k_proj.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".self_attn.k_proj.weight"]["data_offsets"][0].get<size_t>()));
        weights.v_proj.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".self_attn.v_proj.weight"]["data_offsets"][0].get<size_t>()));
        weights.o_proj.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".self_attn.o_proj.weight"]["data_offsets"][0].get<size_t>()));
        weights.post_attn_norm.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".post_attention_layernorm.weight"]["data_offsets"][0].get<size_t>()));
        weights.mlp_gate.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".mlp.gate_proj.weight"]["data_offsets"][0].get<size_t>()));
        weights.mlp_up.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".mlp.up_proj.weight"]["data_offsets"][0].get<size_t>()));
        weights.mlp_down.push_back((__nv_bfloat16*)((char*)gpu_buffer + header[prefix + ".mlp.down_proj.weight"]["data_offsets"][0].get<size_t>()));
    }
    std::cout << "SUCCESS! All 16 layers mapped.\n";
}

int main() 
{
    if (checkGPUStatus() != 0) return 1;

    LlamaWeights weights;
    loadModelToGPU("../model.safetensors", weights);

    int MAX_CONTEXT = 1024; 
    std::cout << "\nAllocating Static KV Cache for " << MAX_CONTEXT << " tokens...\n";
    
    std::vector<__nv_bfloat16*> kv_cache_k(16);
    std::vector<__nv_bfloat16*> kv_cache_v(16);
    size_t cache_layer_size = MAX_CONTEXT * 512 * sizeof(__nv_bfloat16);
    
    for (int i = 0; i < 16; i++) {
        cudaMalloc(&kv_cache_k[i], cache_layer_size);
        cudaMalloc(&kv_cache_v[i], cache_layer_size);
    }
    std::cout << "KV Cache allocated safely in VRAM.\n";

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    int* gpu_tokens; cudaMalloc(&gpu_tokens, sizeof(int));
    __nv_bfloat16 *gpu_hidden_state, *gpu_q, *gpu_k, *gpu_v, *gpu_gate, *gpu_up;
    cudaMalloc(&gpu_hidden_state, 2048 * sizeof(__nv_bfloat16));
    
    __nv_bfloat16* gpu_norm_state;
    cudaMalloc(&gpu_norm_state, 2048 * sizeof(__nv_bfloat16));
    
    cudaMalloc(&gpu_q, 2048 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_k, 512 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_v, 512 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_gate, 8192 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_up, 8192 * sizeof(__nv_bfloat16));

    float* gpu_decode_scores; 
    cudaMalloc(&gpu_decode_scores, 32 * MAX_CONTEXT * sizeof(float));
    
    __nv_bfloat16 *gpu_context_out, *gpu_o_proj_out, *gpu_down, *gpu_logits;
    cudaMalloc(&gpu_context_out, 2048 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_o_proj_out, 2048 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_down, 2048 * sizeof(__nv_bfloat16));
    cudaMalloc(&gpu_logits, 128256 * sizeof(__nv_bfloat16));

    int best_token_id = 0;
    int MAX_GENERATION = 50;
    
    std::vector<int> generated_tokens; 
    std::vector<int> prompt = {128000, 128000, 128006, 882, 128007, 271, 3923, 374, 279, 6864, 315, 9822, 30, 128009, 128006, 78191, 128007, 271};
    int prompt_len = (int)prompt.size();

    std::cout << "\n--- GENERATION STARTED ---\n";

    for (int i = 0; i < prompt_len + MAX_GENERATION; i++) 
    {
        int current_token;
        if (i < prompt_len) {
            current_token = prompt[i];
        } else {
            current_token = best_token_id;
            if (current_token == 128009 || current_token == 128001) break;
        }

        cudaMemcpy(gpu_tokens, &current_token, sizeof(int), cudaMemcpyHostToDevice);
        embeddingGatherKernel<<<1, 1024>>>(gpu_tokens, gpu_hidden_state, weights.embed_tokens);

        for (int layer = 0; layer < 16; layer++) 
        {
            rmsNormKernel<<<1, 1024>>>(gpu_hidden_state, weights.input_norm[layer], gpu_norm_state);
            matmul(cublas_handle, gpu_norm_state, weights.q_proj[layer], gpu_q, 1, 2048, 2048);
            matmul(cublas_handle, gpu_norm_state, weights.k_proj[layer], gpu_k, 1, 512, 2048);
            matmul(cublas_handle, gpu_norm_state, weights.v_proj[layer], gpu_v, 1, 512, 2048);

            ropeKernel<<<1, 1024>>>(gpu_q, 2048, i); 
            ropeKernel<<<1, 256>>>(gpu_k, 512, i);

            size_t cache_offset = i * 512;
            cudaMemcpy(kv_cache_k[layer] + cache_offset, gpu_k, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            cudaMemcpy(kv_cache_v[layer] + cache_offset, gpu_v, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

            int seq_len = i + 1;
            decodeAttentionKernel<<<32, 1024>>>(gpu_q, kv_cache_k[layer], gpu_decode_scores, seq_len);
            decodeSoftmaxKernel<<<32, 1>>>(gpu_decode_scores, seq_len);
            decodeContextKernel<<<32, 64>>>(gpu_decode_scores, kv_cache_v[layer], gpu_context_out, seq_len);

            matmul(cublas_handle, gpu_context_out, weights.o_proj[layer], gpu_o_proj_out, 1, 2048, 2048);
            residualKernel<<<1, 1024>>>(gpu_hidden_state, gpu_o_proj_out);

            rmsNormKernel<<<1, 1024>>>(gpu_hidden_state, weights.post_attn_norm[layer], gpu_norm_state);

            matmul(cublas_handle, gpu_norm_state, weights.mlp_gate[layer], gpu_gate, 1, 8192, 2048);
            matmul(cublas_handle, gpu_norm_state, weights.mlp_up[layer], gpu_up, 1, 8192, 2048);
            siluKernel<<<1, 1024>>>(gpu_gate, gpu_up);
            
            matmul(cublas_handle, gpu_gate, weights.mlp_down[layer], gpu_down, 1, 2048, 8192);
            residualKernel<<<1, 1024>>>(gpu_hidden_state, gpu_down);
        }

        rmsNormKernel<<<1, 1024>>>(gpu_hidden_state, weights.final_norm, gpu_norm_state);
        matmul(cublas_handle, gpu_norm_state, weights.lm_head, gpu_logits, 1, 128256, 2048);

        cudaDeviceSynchronize();

        std::vector<__nv_bfloat16> cpu_logits(128256);
        cudaMemcpy(cpu_logits.data(), gpu_logits, 128256 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        float max_score = -1e20f;
        for(int v = 0; v < 128256; v++) {
            if((float)cpu_logits[v] > max_score) {
                max_score = (float)cpu_logits[v];
                best_token_id = v;
            }
        }

        if (i >= prompt_len - 1) {
            generated_tokens.push_back(best_token_id);
        }
    }
    
    std::cout << "\n[Generation Complete. Decoding output...]\n";
    std::string cmd = "python3 ../tokenizer.py";
    for(int t : generated_tokens) {
        cmd += " " + std::to_string(t);
    }
    int sys_ret = system(cmd.c_str());
    (void)sys_ret; 

    cublasDestroy(cublas_handle);
    return 0;
}