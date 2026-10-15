#!/usr/bin/env bash
# Distilled student ladder. Run from the worktree root after teachers are trained and
#   C:/pfspecies/data/teacher_kd.npz          (ensemble KD targets: labeled + YouTube pool)
#   C:/pfspecies/data/yt_kd_train.parquet     (unlabeled YouTube rows, soft targets only)
# exist. Usage: bash species_benchmark/run_students.sh [name ...]   (default: all)
# KD=0 trains the same ladder with plain cross-entropy (no teacher) into runs/ce_<name>.
set -u
P=${PYTHON:-C:/pfspecies/.venv/Scripts/python.exe}
D=C:/pfspecies/data
R=C:/pfspecies/runs
EP=${EPOCHS:-12}
KD=${KD:-1}
if [ "$KD" = 1 ]; then PFX=kd; KDARGS="--teacher $D/teacher_kd.npz --extra-train $D/yt_kd_train.parquet --kd-alpha 0.7 --kd-t 2.0"; else PFX=ce; KDARGS=""; fi

declare -A BB=(
  [mnv4s]="mobilenetv4_conv_small.e2400_r224_in1k"
  [atto]="convnext_atto.d2_in1k"
  [tinyvit5m]="tiny_vit_5m_224.dist_in22k_ft_in1k"
  [repvit]="repvit_m1_1.dist_450e_in1k"
  [effvit_b1]="efficientvit_b1.r224_in1k"
  [mnv4m]="mobilenetv4_conv_medium.e500_r256_in1k"
  [nano]="convnext_nano.in12k_ft_in1k"
  [pe_s]="vit_pe_core_small_patch16_384.fb"
  [vit_s]="vit_small_patch16_224.augreg_in21k_ft_in1k"
  [convnext_t]="convnext_tiny.in12k_ft_in1k"
  [bioclip_b]="bioclip"
)
declare -A LR=([pe_s]=1e-4 [vit_s]=1e-4 [bioclip_b]=2e-5)
declare -A EXTRA=([bioclip_b]="--text-init --head-lr-mult 50")

names=("$@")
[ ${#names[@]} -eq 0 ] && names=(mnv4s atto tinyvit5m repvit effvit_b1 mnv4m nano pe_s vit_s convnext_t bioclip_b)
for n in "${names[@]}"; do
  out=$R/${PFX}_$n
  [ -f $out/metrics.json ] && { echo "skip $n"; continue; }
  echo "== $n (${BB[$n]}) $(date)"
  $P -m species_benchmark.train --backbone "${BB[$n]}" --img-size 224 --epochs $EP --bs 128 \
    --lr ${LR[$n]:-5e-4} ${EXTRA[$n]:-} --workers 10 \
    $KDARGS \
    --out $out > C:/pfspecies/logs/${PFX}_$n.log 2>&1
  echo "   exit $? $(date)"
done
