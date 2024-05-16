#!/bin/bash
CWD="$(realpath $(pwd)/../)"

ACCELERATOR="gpu"
DEVICES=1
EPOCHS=300
CHECKPOINT_METRIC="val_loss"
PATIENCE=30
LOG_DIR="${CWD}/mlruns/"

# BATCH_SIZE=128
# LEARNING_RATE=0.001