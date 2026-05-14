#!/bin/bash

python3 run_ablation.py --config ablation_config.yaml
python3 evaluation/evaluate_ablations.py
python3 generate_trajectory_preview.py --dir ablation_results --fps 15 --force