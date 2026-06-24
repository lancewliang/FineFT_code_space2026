# FineFT 数据准备脚本逻辑与算法分析

本文先分析核心 Python 数据处理脚本，再补充以下 shell 入口脚本的执行顺序和作用：

- [`preprocess_data.py`](preprocess_data.py)：从原始 `df.feather` 切分训练、验证、测试数据，并生成训练分块。
- [`slice_model.py`](slice_model.py)：对验证集做市场动态切片、合并和标签划分。
- [`vae_data_creation.py`](vae_data_creation.py)：把按动态标签切好的验证片段转换成 VAE 训练用的 `.npy` 文件，并生成测试集特征数组。
- [`create_data_adaboost.py`](create_data_adaboost.py)：把训练、验证、测试集转换成监督学习用的 `X/y` 数组。
- [`split_train_valid_test.sh`](../../FineFT/script/data/split_train_valid_test.sh)：批量运行基础 train/valid/test 切分。
- [`split_valid_multi_dynamics.sh`](../../FineFT/script/data/split_valid_multi_dynamics.sh)：批量运行验证集市场动态切片。
- [`vae_data_creation.sh`](../../FineFT/script/data/vae_data_creation.sh)：批量生成 VAE 训练数组。
- [`create_sl_data.sh`](../../FineFT/script/data/create_sl_data.sh)：批量生成监督学习数组。
- [`ablation.sh`](../../FineFT/script/data/ablation.sh)：基于主训练集生成更小的消融实验数据，并对消融验证集做动态切片。

主线 VAE 数据构建链：

```text
dataset/<symbol>/df.feather
  -> preprocess_data.py
  -> train.feather / valid.feather / test.feather / train/df_*.feather
  -> slice_model.py
  -> valid_processed.feather / valid/label_<k>/df_*.feather
  -> vae_data_creation.py
  -> VAE_data/label_<k>.npy / VAE_data/test.npy
```

监督学习数据构建链：

```text
dataset/<symbol>/df.feather
  -> preprocess_data.py
  -> train.feather / valid.feather / test.feather
  -> create_data_adaboost.py
  -> SL_data/X_train.npy / y_train.npy / X_valid.npy / y_valid.npy / X_test.npy / y_test.npy
```

## 1. preprocess_data.py

### 输入、参数与输出

默认输入为：

```text
dataset/<trading_pair>/df.feather
```

主要参数：

- `--data_path`：数据根目录，默认 `dataset`。
- `--trading_pair`：交易对目录名，默认 `BNBUSDT`。
- `--chunk_length`：训练分块基础长度，默认 `8640`。
- `--early_stop`：每个训练分块额外向后保留的数据长度，默认 `2160`。

输出文件：

```text
dataset/<trading_pair>/train.feather
dataset/<trading_pair>/valid.feather
dataset/<trading_pair>/test.feather
dataset/<trading_pair>/train/df_0.feather
dataset/<trading_pair>/train/df_1.feather
...
```

### 核心逻辑

脚本首先读取完整行情数据：

```python
df = pd.read_feather(os.path.join(args.data_path, args.trading_pair, "df.feather"))
```

然后按原始时间顺序切分：

- `train_length = int(total_length / 2)`，训练集取前 50%。
- `valid_length = int(total_length / 5)`，验证集取接下来的 20%。
- 测试集再取接下来的 20%。

也就是说，代码实际使用比例是：

```text
train : valid : test : unused = 50% : 20% : 20% : 10%
```

文件中注释写的是 `train valid test 3:1:1`，但实现并不是严格的 3:1:1。若按 3:1:1 理解，应是 60%/20%/20%；当前代码还剩最后 10% 数据未写入任何集合。

接着脚本会把训练集再切成多个分块。分块数量为：

```python
train_df_chunk_num = int(len(train_df) / args.chunk_length)
```

每个分块的范围为：

```python
train_df.iloc[
    i * chunk_length : (i + 1) * chunk_length + early_stop
]
```

因此每个训练分块的长度最多为 `chunk_length + early_stop`。相邻分块之间存在重叠：

```text
df_0: [0, chunk_length + early_stop)
df_1: [chunk_length, 2 * chunk_length + early_stop)
```

默认参数下，每块主长度是 `8640`，并额外包含后续 `2160` 条数据，相邻分块重叠 `2160` 条。这种设计通常用于让每个训练 episode 或回测窗口拥有一段额外后续行情，避免在主窗口末端立即截断环境。

### 算法特点

该脚本没有随机抽样或打乱数据，而是严格按时间顺序切分。这适合金融时间序列任务，因为随机切分会把未来信息泄漏到训练集中。

训练分块使用滑动窗口式的固定步长切分，但窗口长度大于步长，所以带重叠。该设计能增加训练窗口之间的连续性，但也意味着不同训练分块并非完全独立。

### 注意点

- 注释比例与实际比例不一致。实际代码不是 3:1:1，而是 5:2:2 并丢弃最后 10%。
- `train_df_chunk_num` 使用向下取整，训练集尾部不足一个 `chunk_length` 的部分不会单独生成一个分块。
- 最后一个分块若超出训练集长度，`iloc` 会自动截断，不会报错。
- 脚本只创建 `train/` 目录，不会清理旧分块；若参数变化后重跑，旧文件可能残留。

## 2. slice_model.py

### 输入、参数与输出

默认输入为：

```text
dataset/BTCUSDT/valid.feather
```

该脚本只处理验证集，用于把验证数据切分成不同“市场动态”片段。主要参数包括：

- `--key_indicator`：用于建模的核心价格列名，默认 `mark_price`，但实际运行时会被覆盖为 `bid1_price`。
- `--timestamp`：时间戳列名，默认 `index`，实际由 DataFrame index 生成。
- `--tic`：资产标识列名，默认 `symbol`。
- `--labeling_method`：动态标签方法，支持 `slope`、`quantile`、`DTW`，默认 `slope`。
- `--min_length_limit`：初始片段最小长度，默认 `288`。
- `--merging_metric`：合并距离度量，默认 `DTW_distance`。
- `--merging_threshold`：距离低于该阈值时允许合并，默认 `0.0003`。
- `--merging_dynamic_constraint`：相邻片段标签差距大于该值时禁止合并，默认 `1`。
- `--filter_strength`：Butterworth 低通滤波强度，默认 `1`。
- `--dynamic_number`：市场动态类别数，默认 `5`。
- `--max_length_expectation`：期望片段最大长度，默认 `864`。

主要输出：

```text
dataset/<symbol>/valid_processed.feather
dataset/<symbol>/valid/label_0/df_*.feather
dataset/<symbol>/valid/label_1/df_*.feather
...
dataset/<symbol>/valid/label_<dynamic_number-1>/df_*.feather
```

脚本构造了 `process_datafile_path`，理论上可输出带标签的完整验证集：

```text
valid_labeled_slice_and_merge_model_...feather
```

但保存逻辑当前被注释掉了，所以默认不会生成这个完整 labeled 文件。

### 入口层处理

`Linear_Market_Dynamics_Model.run()` 先读取 `valid.feather`，并强制补充三列：

```python
raw_data[self.tic] = raw_data["symbol"]
raw_data[self.key_indicator] = raw_data["bid1_price"]
raw_data[self.timestamp] = raw_data.index
```

这意味着虽然参数默认 `key_indicator` 是 `mark_price`，但实际建模使用的是 `bid1_price`。之后保存为：

```text
valid_processed.feather
```

再把该文件交给 `label_util.Worker` 执行真正的切片与标注。

### Worker 的核心算法

`slice_model.py` 的关键算法在 [`label_util.py`](label_util.py) 中，流程可以分为五步。

#### 2.1 预处理与低通滤波

`Worker.preprocess()` 按 `tic` 分组，对每个交易标的构建如下字段：

- `pct_return`：原始 `key_indicator` 的百分比变化率。
- `key_indicator_filtered`：低通滤波后的价格序列。
- `pct_return_filtered`：滤波价格序列的百分比变化率。

低通滤波采用四阶 Butterworth filter：

```python
b, a = butter(order, Wn, btype="low", analog=False)
y = filtfilt(b, a, data)
```

截止频率计算为：

```python
Wn_key_indicator = min(2 / (min_length_limit * filter_strength), 2)
```

默认 `min_length_limit=288`、`filter_strength=1` 时，`Wn = 2 / 288`。这会过滤掉短周期噪声，使后续拐点检测更关注较平滑的价格趋势。

#### 2.2 初始拐点检测

`find_index_of_turning()` 使用滤波后收益率符号变化作为拐点条件：

```python
pct_return_filtered[i] * pct_return_filtered[i + 1] < 0
```

若相邻两点收益率一正一负，则认为趋势方向发生反转，在 `i + 1` 处加入拐点。最终拐点列表始终包含起点 `0` 和终点 `len(data)`。

初始片段就是相邻拐点之间的序列。

#### 2.3 最小长度约束

得到初始片段后，代码会把长度小于 `min_length_limit` 的片段并入当前累计片段，形成新的拐点列表。

该步骤的目的不是根据相似性合并，而是先消除明显过短、可能由噪声造成的微小片段。

#### 2.4 片段斜率估计

对每个片段，代码用 `LinearRegression` 拟合滤波价格：

```text
key_indicator_filtered[t] ~= a * t + b
```

得到斜率 `a` 后，计算归一化斜率：

```python
normalized_coef = 100 * coef / segment_start_price
```

这个值可理解为“相对片段起始价格的趋势斜率百分比”，用于后续标签划分。

#### 2.5 基于 DTW 距离的迭代合并

如果 `merging_threshold != -1`，算法会最多进行 20 轮合并。每一轮中，对长度小于 `max_length_expectation` 的片段，分别计算它与左邻居、右邻居的距离。

默认距离是 `DTW_distance`，实现为：

1. 取两个片段中较短者作为 `shorter`，较长者作为 `longer`。
2. 用较短片段长度作为窗口，在较长片段上滑动。
3. 对每个窗口计算 `fastdtw(shorter, window)`。
4. 取平均距离，并用 `slice_length * mean(shorter)` 归一化。

当左或右邻居距离小于 `merging_threshold` 时，当前片段会并入距离更近的邻居。

如果启用 `merging_dynamic_constraint`，每轮合并前会先用 `quantile` 方法给当前片段临时打标签；若当前片段与候选邻居的标签差距大于约束值，则禁止这次合并。默认值为 `1`，表示只允许相邻或非常接近的动态类别合并，避免把走势差异过大的片段合在一起。

合并结束后，代码会删除空片段，重新计算每个最终片段的线性回归斜率和归一化斜率。

### 动态标签算法

最终标签由 `Dynamic_labeler` 生成，支持三种方法。

#### slope

`slope` 方法先对所有归一化斜率排序，去掉两端极端风险区间的一部分，再在中间范围内均匀划分阈值。

默认 `risk_bond=0.1` 时，大致使用 5% 到 95% 分位附近的斜率范围。然后将该范围切成 `dynamic_number` 个区间。归一化斜率越小，标签越靠前；越大，标签越靠后。

这种方法强调斜率的绝对大小和方向，适合把市场划分为下跌、震荡、上涨等趋势强度区间。

#### quantile

`quantile` 方法直接按归一化斜率分位数切分。例如 `dynamic_number=5` 时，使用 20%、40%、60%、80% 分位作为阈值。

这种方法会让各标签的片段数量更均衡，但标签边界完全由样本分布决定。

#### DTW

`DTW` 方法不使用斜率阈值，而是把每个片段的 `pct_return_filtered` 序列作为时间序列样本，使用 `TimeSeriesKMeans(metric="dtw")` 聚类成 `dynamic_number` 类。

该方法更关注片段形状相似性，而不是简单线性趋势。但计算成本更高，也更依赖时间序列聚类库和参数稳定性。

### 验证集切片输出

`worker.label()` 完成后，`slice_model.py` 把各 `tic` 的带标签数据合并回原始处理数据。随后扫描 `merged_data.label`，每当标签变化时，就把上一段连续同标签数据保存到：

```text
valid/label_<previous_label>/df_<counter>.feather
```

也就是说，输出目录中的每个 `.feather` 文件是验证集里一段连续、标签相同的市场动态片段。

### 注意点

- `raw_data[self.key_indicator] = raw_data["bid1_price"]` 会覆盖参数指定的 `key_indicator` 列含义；实际建模价格来自 `bid1_price`。
- `valid_processed.feather` 会被写出，但完整的 labeled feather 保存代码被注释，默认只生成按标签拆分的片段目录。
- `os.makedirs(... "label_i")` 没有使用 `exist_ok=True`，如果 `valid/label_i` 已存在，重跑会报错。
- 扫描 `merged_data.label` 时，循环只在标签发生变化时写出上一段；最后一个连续标签片段没有在循环结束后额外写出。这会导致验证集末尾片段丢失。
- `merged_data.label[0]` 假设标签列存在且第一行有效；如果上游合并失败或数据为空，会直接报错。
- `slice_model.py` 导入了 `market_dynamics_modeling_analysis`，但对应分析调用被注释，默认不会执行额外分析或绘图。

## 3. vae_data_creation.py

### 输入、参数与输出

默认输入：

```text
dataset/<dataset_name>/valid/label_*/df_*.feather
dataset/<dataset_name>/state_features.npy
dataset/<dataset_name>/test.feather
```

主要参数：

- `--base_path`：输入数据根目录，默认 `dataset`。
- `--dataset_name`：数据集名称，默认 `BTCUSDT`。
- `--save_path`：输出数据根目录，默认 `dataset`。

输出目录：

```text
dataset/<dataset_name>/VAE_data/
```

输出文件：

```text
VAE_data/label_0.npy
VAE_data/label_1.npy
...
VAE_data/test.npy
```

### 核心逻辑

脚本首先读取特征列名：

```python
state_features = np.load(state_name_path)
```

`state_features.npy` 应该保存一组可用于状态表达的 DataFrame 列名。随后遍历：

```text
valid/label_*/
```

对每个标签目录，读取其中所有 `.feather` 片段，取出 `state_features` 对应列：

```python
single_label_data = df[state_features].values
```

再把同一标签下所有片段沿样本维拼接：

```python
single_label_data_all = np.concatenate(single_label_data_list, axis=0)
```

最后保存为：

```text
VAE_data/<label_dir_name>.npy
```

例如：

```text
valid/label_3/*.feather -> VAE_data/label_3.npy
```

处理完验证集标签数据后，脚本再读取测试集：

```python
df = pd.read_feather(test_path)
test_data = df[state_features].values
np.save(..., "test.npy")
```

测试集不会按动态标签切片，而是整体转换为一个特征矩阵。

### 算法特点

该脚本不再做建模或标签计算，只做特征选择、拼接和格式转换。输出的 `.npy` 是二维矩阵：

```text
(样本数, 状态特征数)
```

从相邻的 `vae_dataset.py` 可以看出，VAE 训练侧会用 `np.load()` 读取这些数组，并将每一行当作一个训练样本。

### 注意点

- `os.listdir(valid_path)` 和 `os.listdir(label_path)` 没有排序，拼接顺序依赖文件系统返回顺序。若后续训练不关心时间顺序，这通常影响不大；若需要可复现顺序，应显式排序。
- 如果某个 `label_*` 目录为空，`np.concatenate(single_label_data_list, axis=0)` 会报错。
- 如果 `state_features.npy` 中包含不存在于 `.feather` 文件的列，`df[state_features]` 会报错。
- 该脚本只使用验证集的动态标签片段生成 VAE 数据，没有使用训练集。
- 顶部设置了多个线程相关环境变量，意图是限制底层数值库线程数，降低并行导致的资源占用或非确定性。

## 4. create_data_adaboost.py

### 输入、参数与输出

该脚本由 `create_sl_data.sh` 批量调用，用于生成监督学习数据。默认输入：

```text
dataset/<dataset_name>/train.feather
dataset/<dataset_name>/valid.feather
dataset/<dataset_name>/test.feather
dataset/<dataset_name>/state_features.npy
```

主要参数：

- `--base_path`：输入数据根目录，默认 `dataset`。
- `--dataset_name`：数据集名称，默认 `BTCUSDT`。
- `--save_path`：输出数据根目录，默认 `dataset`。

输出目录：

```text
dataset/<dataset_name>/SL_data/
```

输出文件：

```text
SL_data/X_train.npy
SL_data/y_train.npy
SL_data/X_valid.npy
SL_data/y_valid.npy
SL_data/X_test.npy
SL_data/y_test.npy
```

### 核心逻辑

脚本读取 `state_features.npy` 作为特征列名，并分别处理 `train.feather`、`valid.feather`、`test.feather`。特征矩阵取当前时刻的状态特征：

```python
X = data[state_features].values[:-1]
```

目标值使用下一时刻与当前时刻的 `mark_price` 差值：

```python
y = data["mark_price"].shift(-1) - data["mark_price"]
y = y[:-1]
```

因此每个样本表示“用当前状态特征预测下一步 `mark_price` 变化”。`X` 和 `y` 都会去掉最后一行，因为最后一行没有下一时刻价格可用于构造目标。

### 注意点

- 该脚本依赖 `preprocess_data.py` 已经生成 `train.feather`、`valid.feather` 和 `test.feather`。
- 该脚本不依赖 `slice_model.py` 的市场动态标签，也不依赖 `VAE_data/`。
- `reward_feature` 在代码中固定为 `mark_price`，不能通过命令行参数修改。
- `SL_data` 目录只在不存在时创建；重跑会覆盖同名 `.npy` 文件。

## 整体数据与算法关系

这些脚本的设计目的不同：

- `preprocess_data.py` 负责时间序列级别的数据切分，保证训练、验证、测试按时间先后分离。
- `slice_model.py` 负责在验证集上识别市场动态。它使用滤波、拐点、线性斜率、DTW 距离和动态标签约束，把验证行情拆成多类连续片段。
- `vae_data_creation.py` 负责把这些动态片段转成 VAE 可直接读取的特征矩阵。
- `create_data_adaboost.py` 负责把基础 train/valid/test 数据转成监督学习可直接读取的特征和目标数组。

从算法角度看，最关键的是 `slice_model.py`。它不是简单按固定窗口切分，而是采用“拐点初分 + 最小长度过滤 + DTW 相似性合并 + 斜率或形状标签”的动态建模流程。这样生成的 VAE 数据按市场状态分组，后续模型可以学习不同市场动态下的状态分布边界。

## Shell 入口脚本的执行顺序和作用

这些 shell 脚本都假设从 `FineFT` 目录运行，因为命令使用了 `python datahandler/...`、`dataset/...` 和 `log/...` 这类相对路径。运行前应先进入项目子目录并激活环境：

```bash
cd FineFT
conda activate finetf
```

脚本中的每条命令都使用 `nohup ... >log/... 2>&1 &` 后台运行，所以同一个 shell 脚本内部的多币种任务会并行启动，调用者需要通过日志文件确认任务是否完成。

### 1. split_train_valid_test.sh

作用：批量运行基础数据切分。

```text
datahandler/preprocess_data.py --trading_pair BTCUSDT
datahandler/preprocess_data.py --trading_pair BNBUSDT
datahandler/preprocess_data.py --trading_pair ETHUSDT
datahandler/preprocess_data.py --trading_pair DOTUSDT
```

输入：

```text
dataset/<symbol>/df.feather
```

输出：

```text
dataset/<symbol>/train.feather
dataset/<symbol>/valid.feather
dataset/<symbol>/test.feather
dataset/<symbol>/train/df_*.feather
```

这是主线数据准备的第一步。后续 `split_valid_multi_dynamics.sh`、`vae_data_creation.sh`、`create_sl_data.sh` 和 `ablation.sh` 都直接或间接依赖它的输出。

### 2. split_valid_multi_dynamics.sh

作用：批量对基础验证集做市场动态切片。

```text
datahandler/slice_model.py --data_path dataset/<symbol>/valid.feather
```

输入：

```text
dataset/<symbol>/valid.feather
```

输出：

```text
dataset/<symbol>/valid_processed.feather
dataset/<symbol>/valid/label_<k>/df_*.feather
```

它必须在 `split_train_valid_test.sh` 完成之后运行，因为它读取基础切分得到的 `valid.feather`。该步骤是 VAE 数据生成的前置步骤，但不是监督学习数据生成的前置步骤。

### 3. vae_data_creation.sh

作用：批量把动态标签验证片段转换成 VAE 训练数组，并把测试集转换成 `test.npy`。

```text
datahandler/vae_data_creation.py --dataset_name <symbol>
```

输入：

```text
dataset/<symbol>/valid/label_*/df_*.feather
dataset/<symbol>/state_features.npy
dataset/<symbol>/test.feather
```

输出：

```text
dataset/<symbol>/VAE_data/label_<k>.npy
dataset/<symbol>/VAE_data/test.npy
```

它应在 `split_valid_multi_dynamics.sh` 完成之后运行。否则 `valid/label_*` 目录还不存在，无法生成按动态标签分组的 VAE 数据。

### 4. create_sl_data.sh

作用：批量生成监督学习数据。

```text
datahandler/create_data_adaboost.py --dataset_name <symbol>
```

输入：

```text
dataset/<symbol>/train.feather
dataset/<symbol>/valid.feather
dataset/<symbol>/test.feather
dataset/<symbol>/state_features.npy
```

输出：

```text
dataset/<symbol>/SL_data/X_train.npy
dataset/<symbol>/SL_data/y_train.npy
dataset/<symbol>/SL_data/X_valid.npy
dataset/<symbol>/SL_data/y_valid.npy
dataset/<symbol>/SL_data/X_test.npy
dataset/<symbol>/SL_data/y_test.npy
```

它只依赖 `split_train_valid_test.sh` 的基础切分结果，不依赖 `split_valid_multi_dynamics.sh` 或 `vae_data_creation.sh`。因此在基础切分完成后，`create_sl_data.sh` 可以和验证集动态切片/VAE 分支并行准备。

### 5. ablation.sh

作用：构建消融实验用的小规模数据分支，并对消融验证集做市场动态切片。

第一组命令运行：

```text
datahandler/ablation_data_slice.py --trading_pair <symbol>
```

输入：

```text
dataset/<symbol>/train.feather
```

输出：

```text
dataset/ablation/<symbol>/train.feather
dataset/ablation/<symbol>/valid.feather
dataset/ablation/<symbol>/train/df_*.feather
```

`ablation_data_slice.py` 不是从原始 `df.feather` 重新切分，而是从主线已经生成的 `dataset/<symbol>/train.feather` 再切出更小的消融训练集和消融验证集。默认比例为：

```text
ablation train : ablation valid : unused = 40% : 20% : 40%
```

默认训练分块也更短：

```text
chunk_length = 864
early_stop = 216
```

第二组命令运行：

```text
datahandler/slice_model.py --data_path dataset/ablation/<symbol>/valid.feather
```

输入：

```text
dataset/ablation/<symbol>/valid.feather
```

输出：

```text
dataset/ablation/<symbol>/valid_processed.feather
dataset/ablation/<symbol>/valid/label_<k>/df_*.feather
```

因此，消融分支的逻辑顺序是：

```text
split_train_valid_test.sh
  -> ablation_data_slice.py
  -> slice_model.py --data_path dataset/ablation/<symbol>/valid.feather
```

但 `ablation.sh` 当前把两组命令都用 `&` 直接放到后台，没有在中间 `wait`。这意味着 `slice_model.py` 可能在 `ablation_data_slice.py` 写完 `dataset/ablation/<symbol>/valid.feather` 之前启动，存在竞态风险。更稳妥的运行方式是先等待第一组消融切分日志完成，再运行第二组动态切片命令。

## 建议的运行顺序

主线 VAE 数据建议按以下顺序运行：

```bash
bash script/data/split_train_valid_test.sh
# 等待 log/data/<symbol>/train_valid_test_split.log 完成且无错误
bash script/data/split_valid_multi_dynamics.sh
# 等待 log/data/<symbol>/valid_split.log 完成且无错误
bash script/data/vae_data_creation.sh
```

监督学习数据只依赖基础切分：

```bash
bash script/data/split_train_valid_test.sh
# 等待基础切分完成
bash script/data/create_sl_data.sh
```

消融实验数据依赖基础切分：

```bash
bash script/data/split_train_valid_test.sh
# 等待基础切分完成
bash script/data/ablation.sh
```

由于 `ablation.sh` 内部缺少阶段间等待，若要避免竞态，建议手动拆成两阶段执行：先运行四个 `ablation_data_slice.py` 命令并等待完成，再运行四个 `slice_model.py --data_path dataset/ablation/<symbol>/valid.feather` 命令。

单个交易对的主线等价命令如下：

```bash
python datahandler/preprocess_data.py --data_path dataset --trading_pair BNBUSDT
python datahandler/slice_model.py --data_path dataset/BNBUSDT/valid.feather
python datahandler/vae_data_creation.py --base_path dataset --dataset_name BNBUSDT --save_path dataset
python datahandler/create_data_adaboost.py --base_path dataset --dataset_name BNBUSDT --save_path dataset
```

如果要重复运行 `slice_model.py`，建议先备份或清理已有的：

```text
dataset/<symbol>/valid/
dataset/<symbol>/valid_processed.feather
```

否则已有 `label_*` 目录可能导致创建目录时报错。
