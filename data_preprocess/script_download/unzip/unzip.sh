
# 定义根路径和日志路径
root_store_path="./DOWNLOAD_DATASET/binance-futures"
log_base_path="log/download/unzip"

# 定义币种和数据类型数组
datasets=("BNBUSDT" "BTCUSDT" "ETHUSDT" "DOTUSDT")
data_types=("quotes" "trades" "derivative_ticker" "book_snapshot_25")

# 迭代处理每个币种和数据类型
for dataset in "${datasets[@]}"; do
  for data_type in "${data_types[@]}"; do
    # 构建数据存储路径
    store_path="$root_store_path/$dataset/$data_type"
    
    # 构建日志路径
    log_dir="$log_base_path/$dataset"
    log_path="$log_dir/$data_type.log"
    
    # 检查并创建日志目录
    if [ ! -d "$log_dir" ]; then
      mkdir -p "$log_dir"
    fi
    
    # 运行命令
    nohup python download_operator/unzip.py --root_store_path "$store_path" >"$log_path" 2>&1 &
  done
done
