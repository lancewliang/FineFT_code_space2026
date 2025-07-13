# 定义symbols、start_date和end_date变量
symbols="ETHUSDT"
start_date="2024-07-15"
end_date="2024-07-26"
data_types=(
    'quotes'
    'trades'
    'derivative_ticker'
    'book_snapshot_25'

)
# 遍历data_types数组
for data_type in "${data_types[@]}"; do
    # 构建目录路径
    dir_path="log/download/${symbols}/${start_date}_${end_date}"

    # 检查目录是否存在，如果不存在，则创建
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
    fi

    # 构建日志文件路径
    log_path="${dir_path}/${data_type}.log"

    # 使用nohup命令执行Python脚本并重定向输出到日志文件
    nohup python download_operator/download.py \
        --symbols "$symbols" --start_date "$start_date" --end_date "$end_date" --data_types "$data_type" \
        >"$log_path" 2>&1 &

    # 输出信息到控制台
    echo "${symbols} download ${data_type} from ${start_date} to ${end_date} initiated."
done

echo "所有下载任务已启动。"
