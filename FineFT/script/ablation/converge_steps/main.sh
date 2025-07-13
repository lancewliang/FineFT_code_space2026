nohup bash script/ablation/converge_steps/train_util.sh >log/ablation/train_util.log 2>&1 &

nohup bash script/ablation/converge_steps/test_util.sh >log/ablation/test_util.log 2>&1 &
