# How to Run ConDiffPlan

## 1. Install Dependencies
From the project root:
```bash
pip install -r requirements.txt
```

## 2. Run Scripts (inside `Code/`)
```bash
cd Code
```

1. **Preprocess Dataset** (creates `Code/data_cache/`):
   ```bash
   python preprocess.py --data ../Dataset/ResPlan.pkl --split ../Dataset/split.json --cache-dir data_cache
   ```

2. **Train Model** (saves weights to `Code/checkpoints/best.pt`, `latest.pt`, `step_<N>.pt`):
   ```bash
   # Level 1 — Auto-Optimized (recommended default for any NVIDIA GPU:
   # auto BF16/FP16/FP32 + TF32, length-bucketed batching, GPU masks, fused Adam/EMA,
   # and HouseDiffusion continuous + discrete 8-bit denoising)
   python train.py --preset default --batch-size 64 --steps 500000

   # Level 2 — Maximum Speed (Linux CUDA: adds torch.compile Triton kernel fusion, ~1.3x-1.6x faster)
   python train.py --preset default --batch-size 64 --steps 500000 --compile

   # Level 3 — High-VRAM GPUs (24GB+, e.g. RTX 3090/4090/A100: larger batch + compile)
   python train.py --preset default --batch-size 128 --steps 500000 --compile

   # Level 4 — Mid/Low-VRAM GPUs (6-8GB, e.g. RTX 2060/3060Ti/3070: keeps effective batch 64)
   python train.py --preset default --batch-size 64 --micro-batch-size 32 --steps 500000

   # Level 5 — Very Low-VRAM GPUs (4-6GB, e.g. GTX 1650/1660/laptop GPUs: micro-batch + activation checkpointing)
   python train.py --preset default --batch-size 64 --micro-batch-size 16 --grad-checkpointing --steps 500000

   # Resume from latest checkpoint (also upgrades legacy continuous-only checkpoints seamlessly)
   python train.py --preset default --batch-size 64 --steps 500000 --resume checkpoints/latest.pt

   # Quick CPU smoke test
   python train.py --smoke-test
   ```

3. **Generate Floorplan** (outputs vector coordinates JSON + optional SVG/PNG):
   ```bash
   python sample.py --checkpoint checkpoints/best.pt --mode boundary --num-candidates 10 --output-json outputs/plan.json --export-svg outputs/plan.svg --export-png outputs/plan.png
   ```

4. **Evaluate Checkpoint**:
   ```bash
   python eval.py --checkpoint checkpoints/best.pt --limit 100
   ```

5. **Run Tests**:
   ```bash
   pytest -v
   ```
