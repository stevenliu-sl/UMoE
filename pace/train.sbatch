#!/bin/bash
#SBATCH --job-name=umoe-train
#SBATCH --account=coc
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/ice1/2/4/sliu736/umoe/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_gt_username@gatech.edu

source ~/UMoE/env.sh
cd ~/UMoE

# ---------------- Run settings (keep identical across chained jobs) ----------------
RUN_NAME=umoe_small_baseline
CKPT=/storage/ice1/2/4/sliu736/umoe/ckpt
DATA_ROOT=/storage/ice1/2/4/sliu736/umoe/data

# Training data: one or more patterns, each in quotes
TRAIN_GLOBS=(
    "$DATA_ROOT/fineweb-edu/sample/10BT/*.parquet"
)
EVAL_DIR=$DATA_ROOT/fineweb-edu-eval

RUN_DIR=$CKPT/$RUN_NAME
MANIFEST=$RUN_DIR/train_files_manifest.txt

# ---------------- Stop if already finished ----------------
if [ -d "$RUN_DIR/final_checkpoint" ]; then
    echo "Training already complete; nothing to do."; exit 0
fi
mkdir -p "$RUN_DIR"

# ---------------- Resolve the patterns to an exact file list ----------------
CURRENT=$(for g in "${TRAIN_GLOBS[@]}"; do ls -1 $g 2>/dev/null; done | sort -u)
if [ -z "$CURRENT" ]; then
    echo "ERROR: no training files match TRAIN_GLOBS"; exit 1
fi

# ---------------- Guard: no eval shard in the training set ----------------
for f in $(find "$EVAL_DIR" -name "*.parquet" 2>/dev/null); do
    if echo "$CURRENT" | grep -q "/$(basename "$f")$"; then
        echo "ERROR: eval shard $(basename "$f") is also in the training set"; exit 1
    fi
done

# ---------------- Guard: training files must not change between jobs ----------------
CURRENT_MANIFEST=$(echo "$CURRENT" | xargs -I{} stat -c "%s %n" {})
if [ -f "$MANIFEST" ]; then
    if ! diff <(echo "$CURRENT_MANIFEST") "$MANIFEST" > /dev/null; then
        echo "ERROR: training files changed since this run started:"
        diff <(echo "$CURRENT_MANIFEST") "$MANIFEST"
        exit 1
    fi
    echo "Manifest check passed."
else
    echo "$CURRENT_MANIFEST" > "$MANIFEST"
    echo "Created manifest: $MANIFEST"
fi
echo "Training on $(echo "$CURRENT" | wc -l) files:"; echo "$CURRENT"

# ---------------- Resume only if a checkpoint exists ----------------
RESUME=""
ls -d "$RUN_DIR"/checkpoint-* > /dev/null 2>&1 && RESUME="--resume"
echo "Starting at $(date) with RESUME='$RESUME'"

python train.py \
    --config_name small-vanilla-UMoE-1024 \
    --run_name $RUN_NAME \
    --output_dir $CKPT \
    --data_files "${TRAIN_GLOBS[@]}" \
    --dataset_text_field text \
    --disable_iter_shuffle \
    --world_size 1 \
    --micro_batch_size 16 \
    --gradient_accumulation_steps 64 \
    --global_batch_size 1024 \
    --max_steps 950 \
    --save_steps 45 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --num_proc 16 \
    --disable_wandb \
    $RESUME
