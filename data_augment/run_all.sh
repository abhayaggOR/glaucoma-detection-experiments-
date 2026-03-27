#!/bin/bash
echo "============================================="
echo "STARTING RUNS 15 (Differential LR)"
echo "============================================="
python3 train_diff_lr.py > training_runs15.log 2>&1

echo "============================================="
echo "STARTING RUNS 16 (Progressive Unfreezing)"
echo "============================================="
python3 train_prog_unfreeze.py > training_runs16.log 2>&1

echo "============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================="
