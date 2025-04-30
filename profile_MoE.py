import torch
from torch.profiler import profile, record_function, ProfilerActivity
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
import os

# os.environ["VLLM_USE_DEEP_GEMM"] = "1"


fp8_config = Fp8Config(
    is_checkpoint_fp8_serialized=False,  # PowerMoE-3B不是原生FP8格式
    activation_scheme="dynamic",
    weight_block_size=None  # 或根据需要设置
)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object. 采样策略类
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    llm = LLM(
        model="/models/models--ibm-research--PowerMoE-3b/snapshots/13fcb5a98001438bed01cf1ac4b423751dc4c2ea",  # 或下载后的本地路径
        tensor_parallel_size=1,  # 根据您的GPU数量调整
        dtype="float16",  # 初始加载为float16，代码会处理FP8转换
        quantization="fp8",
        enforce_eager=True
    )
    outputs = llm.generate(prompts, sampling_params)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("PowerMoE_trace.json")
    
