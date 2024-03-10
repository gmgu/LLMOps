# LLMOps

This repository provides an hour-long seminar about distributed training large language models

## Outline

1. Large Language Model and Large Multimodal Model
   - Examples: text-to-text, text-to-code, image-to-text, and so on
   - In-Context Learning and Scaling Law [1, 2, 3, 4, 5]
3. What is LLMOps [6]
   - Data, Training, Evaluation, Serving
5. When do we need to train LLM
   - Prompt Engineering, Retrieval-Augmented Generation, Parameter Efficient Fine-Tuning, Full Parameter Fine-Tuning
7. Challenges in Training LLM
   - Memory and Time requirements
9. Distributed Training Technique
    - Single Matrix Multiplication
    - Parallel Matrix Multiplication (Tensor Parallelism) [7]
11. Hands-On Tensor Parallelism
    - Training Megatron-LM with 2 GPUs [8]
13. More Distributed Training Techniques
    - Data Parallelism
    - Pipeline Parallelism [9]
    - Sequence Parallelism [10]
    - Sharded Data Paralleism [11, 12]

**References**

[1] In-context Learning and Induction Heads, https://arxiv.org/abs/2209.11895

[2] Language Models are Few-Shot Learners, https://arxiv.org/abs/2005.14165

[3] Training Compute-Optimal Large Language Models, https://arxiv.org/abs/2203.15556

[4] DeepSeek LLM: Scaling Open-Source Language Models with Longtermism, https://arxiv.org/abs/2401.02954

[5] Are Emergent Abilities of Large Language Models a Mirage?, https://arxiv.org/abs/2304.15004

[6] https://wandb.ai/site/articles/understanding-llmops-large-language-model-operations

[7] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism, https://arxiv.org/abs/1909.08053

[8] https://github.com/NVIDIA/Megatron-LM

[9] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, https://arxiv.org/abs/2104.04473

[10] Reducing Activation Recomputation in Large Transformer Models, https://arxiv.org/abs/2205.05198

[11] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, https://arxiv.org/abs/1910.02054

[12] PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel, https://arxiv.org/abs/2304.11277
