#!/bin/bash
# Usage (on a GPU node): bash ~/UMoE/pace/batch_sweep.sh <config_name> "4 8 16 32"
CONFIG=$1
SIZES=${2:-"4 8 16 32"}
STEPS=${STEPS:-30}
BASE=/storage/ice1/2/4/sliu736/umoe
DATA=${DATA:-"$BASE/data/fineweb-edu/sample/10BT/*.parquet"}
OUT=$BASE/logs/batch_sweep_$(date +%Y%m%d_%H%M).csv

cd ~/UMoE
SEQ=$(python -c "import json; print(json.load(open('model_libs/configs/$CONFIG/config.json'))['max_position_embeddings'])")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "gpu,config,seq_len,micro_batch,steps,status,train_runtime_s,sec_per_step,tokens_per_s,peak_mem_gb_torch,peak_mem_mib_smi" > "$OUT"

for MB in $SIZES; do
    RUN=sweep_mb$MB
    LOG=$BASE/logs/${RUN}.log
    echo ">>> micro_batch=$MB ..."

    # Background GPU memory monitor
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$BASE/logs/${RUN}_mem.txt" &
    MON=$!

    python train.py --config_name "$CONFIG" --run_name "$RUN" --output_dir "$BASE/ckpt" \
        --data_files "$DATA" --dataset_text_field text --disable_iter_shuffle \
        --world_size 1 --micro_batch_size "$MB" --gradient_accumulation_steps 1 --global_batch_size "$MB" \
        --max_steps "$STEPS" --save_steps 100000 --logging_steps 10 --num_proc 16 --disable_wandb \
        > "$LOG" 2>&1
    RC=$?

    kill $MON 2>/dev/null
    PEAK_SMI=$(sort -n "$BASE/logs/${RUN}_mem.txt" | tail -n 1)
    rm -rf "$BASE/ckpt/$RUN"

    if [ $RC -ne 0 ]; then
        if grep -qi "out of memory" "$LOG"; then STATUS=OOM; else STATUS=ERROR; fi
        echo "$GPU,$CONFIG,$SEQ,$MB,$STEPS,$STATUS,,,,,$PEAK_SMI" >> "$OUT"
        echo "    $STATUS (details in $LOG)"
        [ "$STATUS" = "OOM" ] && break     # larger sizes will also fail
        continue
    fi

    RUNTIME=$(grep -o "'train_runtime': [0-9.]*" "$LOG" | tail -n 1 | awk '{print $2}')
    PEAK_T=$(grep -o "PEAK_MEM_GB [0-9.]*" "$LOG" | awk '{print $2}')
    if [ -z "$RUNTIME" ]; then
        echo "$GPU,$CONFIG,$SEQ,$MB,$STEPS,NO_RUNTIME,,,,$PEAK_T,$PEAK_SMI" >> "$OUT"
        echo "    finished, but train_runtime not found in $LOG"
        continue
    fi
    read SPS TPS < <(python -c "r=$RUNTIME; print(f'{r/$STEPS:.3f} {$MB*$SEQ*$STEPS/r:.0f}')")
    echo "$GPU,$CONFIG,$SEQ,$MB,$STEPS,OK,$RUNTIME,$SPS,$TPS,$PEAK_T,$PEAK_SMI" >> "$OUT"
    echo "    OK: ${SPS}s/step, ${TPS} tokens/s, peak ${PEAK_T} GB"
done

echo; echo "Results saved to $OUT"; echo
column -s, -t "$OUT"
