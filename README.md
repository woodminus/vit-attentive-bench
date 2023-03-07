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
- batch size for speed testing (images/s): 64
- embedding dimension:768
- number of heads: 12

### Testing

For example, to test HiLo attention,

```bash
cd attentions/
python hilo.py
```

> By default, the script will test models on both CPU and GPU. FLOPs is measured by fvcore. You may want to edit the source file as needed.

Outputs:

```bash
Number of Params: 2.2 M
FLOPs = 298.3 M
throughput averaged with 30 times
batch_size 64 throughput on CPU 1029
throughput averaged with 30 times
batch_size 64 throughput on GPU 5104
```

### Supported Attentions

- Numerous attention mechanisms along with their respective papers and codes.

### Single Attention Layer Benchmark

| Name | Params (M) | FLOPs (M) | CPU Speed | GPU Speed | Demo |

- Various attention mechanisms along with their respective c