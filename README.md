# Vision Transformer Attention Mechanism Benchmark

This repository, maintained by woodminus, is a comprehensive benchmarking of various attention mechanisms used in Vision Transformers. It not only provides a re-implementation but also furnishes a performance benchmark on parameters, FLOPs and CPU/GPU throughput of different attention mechanisms.

### Requirements

- Pytorch 1.8+
- timm
- ninja
- einops
- fvcore
- matplotlib

### Testing Environment

- NVIDIA RTX 3090
- Intel® Core™ i9-10900X CPU @ 3.70GHz
- Memory 32GB
- Ubuntu 22.04
- PyTorch 1.8.1 + CUDA 11.1

### Setting

- input: 14 x 14 = 196 tokens (1/16 scale feature maps in common ImageNet-1K training)
- batc