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
   # Full training (use GPU)
   python train.py --preset default --batch-size 64 --steps 500000

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
