# nohup python analysis/motivation_finding/create_tsne_result.py \
#     >log/analysis/motivation_finding/tsne.log 2>&1 &
export OPENBLAS_NUM_THREADS=32

nohup python analysis/motivation_finding/create_tsne_without_down_scaling.py \
    --uncertainty_type aleatoric \
    >log/analysis/motivation_finding/tsne_aleatoric.log 2>&1 &


nohup python analysis/motivation_finding/create_tsne_without_down_scaling.py \
    --uncertainty_type epistemic \
    >log/analysis/motivation_finding/tsne_epistemic.log 2>&1 &
