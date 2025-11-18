# iitj-ai-nas-ga
Basic demo of Neural Architecture Search (NAS) using Genetic Algorithm (GA)

## Assignment Modifications

This codebase has been modified to include:

1. **Q1A: Roulette-Wheel Selection** - Replaced tournament selection with fitness-proportionate roulette-wheel selection
2. **Q2B: Weighted Fitness Function** - Separate penalties for Conv (weight=2.5) and FC (weight=1.0) parameters based on computational complexity

## Quick Start

```bash
python nas_run.py
```

Logs will be saved in `outputs/run_X/nas_run.log`