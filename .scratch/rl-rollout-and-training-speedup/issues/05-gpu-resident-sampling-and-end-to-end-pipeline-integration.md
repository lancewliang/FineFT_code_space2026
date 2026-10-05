# 05: GPU-Resident Sampling and End-to-End Pipeline Integration

**What to build:** Move the pre-allocated continuous stratified buffer directly into GPU VRAM (~1.01 GB) to perform in-VRAM fancy index sampling, eliminating all 48GB PCIe synchronization during the 600 update steps per epoch. Integrate the targeted inference, right-sized worker pool, asynchronous evaluator, and GPU-resident buffer into `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py` and `FineFT/script/train/train_commodity_fu_10.sh`, delivering an end-to-end 5~6x speedup (single epoch dropping from ~550s to <= 85s).

**Blocked by:** 01: Targeted Sub-Network Rollout Inference, 02: Worker Pool Right-Sizing and Shared Q-Table, 03: Asynchronous Non-blocking Greedy Evaluation, 04: Native Continuous Tensor Stratified Replay Buffer

**Status:** completed

- [x] `GPURegimeStratifiedReplayBuffer` resides in GPU memory, executing randomized fancy indexing and batch slicing natively on CUDA cores with zero host-to-device PCIe transfer during training iterations.
- [x] Rollout transitions collected across workers are uploaded to GPU memory via a single bulk DMA transfer at exploration completion.
- [x] Target network Polyak soft copy utilizes vectorized `torch._foreach_lerp_` across module parameters.
- [x] `train_commodity_fu_10.sh` is updated to run the integrated pipeline with `--diverse_num_workers 40`.
- [x] End-to-end multi-epoch integration test confirms epoch runtime <= 85s, total RAM <= 28GB with 0 MB swap, GPU compute utilization >= 90% during training, and TD/KL loss convergence matching baseline.
