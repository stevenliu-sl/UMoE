#!/bin/bash
# Usage (on a GPU node): bash ~/UMoE/pace/save_timing.sh <config_name> <micro_batch>
CONFIG=$1; MB=$2
BASE=/storage/ice1/2/4/sliu736/umoe
DATA=${DATA:-"$BASE/data/fineweb-edu/sample/10BT/*.parquet"}
cd ~/UMoE
for SAVE in 100000 10; do
  echo "save_steps=$SAVE:"
  python train.py --config_name "$CONFIG" --run_name savetest_$SAVE --output_dir "$BASE/ckpt" \
    --data_files "$DATA" --dataset_text_field text --disable_iter_shuffle --world_size 1 \
    --micro_batch_size "$MB" --gradient_accumulation_steps 1 --global_batch_size "$MB" \
    --max_steps 20 --save_steps $SAVE --logging_steps 10 --num_proc 16 --disable_wandb 2>&1 \
    | grep -o "'train_runtime': [0-9.]*"
done
du -sh "$BASE/ckpt/savetest_10/checkpoint-10"
rm -rf "$BASE"/ckpt/savetest_*
