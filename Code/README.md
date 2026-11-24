# ConDiffPlan for ResPlan

A conditional vector floorplan diffusion model based on **ConDiffPlan** (*ChatAssistDesign: A language-interactive framework for iterative vector floorplan generation via conditional diffusion*, Li et al., 2026), adapted to the 17,000-plan **ResPlan** vector-graph dataset.

## 1. Standard Installation (`venv` + `pip`)

From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Running the Code

From inside `Code/`:

```bash
cd Code

# (Optional) Preprocess ResPlan.pkl into Code/data_cache/
# Note: train.py also runs this automatically if data_cache/train.pt does not exist yet.
python preprocess.py --data ../Dataset/ResPlan.pkl --split ../Dataset/split.json --cache-dir data_cache

# Full training (default 6-layer, 512-dim model, 500k steps, T=1000)
python train.py \
  --preset default \
  --batch-size 64 \
  --steps 500000 \
  --lr 1e-4 \
  --lr-min 1e-6 \
  --timesteps 1000 \
  --checkpoint-dir checkpoints

# Quick CPU smoke test (32 plans, 5 steps)
python train.py --smoke-test

# Sample vector coordinates from a connection graph
python sample.py --checkpoint checkpoints/latest.pt --mode boundary --num-candidates 10 \
  --output-json outputs/plan.json --export-svg outputs/plan.svg --export-png outputs/plan.png

# Evaluate a trained checkpoint
python eval.py --checkpoint checkpoints/best.pt --limit 100

# Run unit & integration tests
pytest -v
```
