# RL/DiHFT/VAE/main.py 逻辑流程说明

本文档说明 `RL/DiHFT/VAE/main.py` 的入口逻辑、训练流程和分析流程。该脚本用于为 DiHFT 的高层路由模块训练或使用按市场动态标签划分的 VAE 模型。

整体上，VAE 的作用是学习某个验证集动态标签 `label_<k>` 对应的数据分布，并用重构似然分数 `log p(x)` 判断样本更像当前标签内分布，还是更像测试集或其他标签分布。

```text
dataset/<dataset_name>/VAE_data/label_<k>.npy
        |
        v
构造 One_Dim_Dataset 和 DataLoader
        |
        v
构造 MLP_VAE
        |
        +-- 训练分支: train -> periodic test -> save model_latest.pth
        |
        +-- 普通分析分支: load model_latest.pth -> id/test logpx -> AUROC/AUPRC/FPR80
        |
        +-- 交叉分析分支: load model_latest.pth -> 对多个 label 数据计算 ood_logpx_<i>.npy
```

## 1. 脚本定位

核心入口文件：

```text
RL/DiHFT/VAE/main.py
```

它本身只负责解析参数、拼路径和选择执行分支，实际工作委托给：

- `RL/DiHFT/VAE/process.py`：数据加载、模型构造、训练循环、分析指标。
- `RL/DiHFT/VAE/vae.py`：`MLP_VAE` 模型、loss、单 epoch 训练/测试、样本打分。
- `RL/DiHFT/VAE/util.py`：ROC/PR 指标和张量形状辅助函数。
- `datahandler/vae_dataset.py`：读取 `.npy` 并封装成 PyTorch Dataset。

模型和分析结果默认输出到：

```text
result/DiHFT/vae_results/<dataset_name>/label_<label_index>/
```

输入数据来自数据准备链路生成的：

```text
dataset/<dataset_name>/VAE_data/label_<label_index>.npy
dataset/<dataset_name>/VAE_data/test.npy
```

## 2. 参数含义

主要参数如下：

- `--if_train`：是否执行训练分支，默认 `False`。
- `--if_cross_analyze`：是否执行交叉分析分支，默认 `True`。
- `--dataset_name`：交易对名称，默认 `BTCUSDT`。
- `--data_base_path`：数据根目录，默认 `dataset`。
- `--label_index`：当前 VAE 要拟合的市场动态标签编号，默认 `0`。
- `--total_label_number`：市场动态标签总数，默认 `5`。
- `--base_model_path`：模型结果根目录，默认 `result/DiHFT`。
- `--z_dim`：VAE 潜变量维度，默认 `512`。
- `--hidden_dims`：MLP 隐藏层维度，默认 `[4096, 2048, 1024, 1024]`。
- `--batch_size`：训练 batch size，默认 `128`；非训练分支会被强制改为 `1`。
- `--loss`：重构 loss 类型，默认 `NLL`，也支持 `BCE`。
- `--epochs`：训练轮数，默认 `2000`。
- `--log_interval`：训练日志和测试间隔，默认 `100`。
- `--save_interval`：模型保存间隔，默认 `50`。
- `--learning_rate`：Adam 学习率，默认 `1e-5`。

`sample_ratio` 和 `prr` 相关参数在当前 1D VAE 主流程中基本没有实际使用。

## 3. Piplineruner 初始化流程

`main.py` 中的核心封装类是：

```python
class Piplineruner
```

初始化时会先保存 `args`。如果不是训练分支：

```python
if not self.args.if_train:
    self.args.batch_size = 1
```

这意味着普通分析和交叉分析都会逐样本计算 log likelihood。这样保存出来的 `id_logpx.npy`、`ood_logpx.npy` 或 `ood_logpx_<i>.npy` 的每个元素可以对应一个单独样本。

然后根据 `label_index` 构造当前标签名：

```text
label_<label_index>
```

并生成三个关键路径：

```text
single_label_save_path =
result/DiHFT/vae_results/<dataset_name>/label_<label_index>/

train_data_path =
dataset/<dataset_name>/VAE_data/label_<label_index>.npy

ood_test_dataset_path =
dataset/<dataset_name>/VAE_data/test.npy
```

最后调用 `prepare_model()` 完成数据、模型和优化器准备。

## 4. 数据加载流程

数据集类是 `One_Dim_Dataset`：

```python
numpy_data = np.load(dataset_path)
self.data = torch.from_numpy(numpy_data).float()
self.input_dim = numpy_data[0].shape[0]
```

因此 VAE 输入文件必须是二维数组：

```text
(样本数, 特征维度)
```

`prepare_model()` 会加载三份 Dataset：

```python
train_data = One_Dim_Dataset(train_data_path)
test_data = One_Dim_Dataset(train_data_path)
ood_test_data = One_Dim_Dataset(ood_test_dataset_path)
```

注意 `train_data` 和 `test_data` 都来自同一个 `label_<k>.npy`。也就是说，这里没有单独的 ID validation 文件；训练时的 test loss 实际是在同一标签数据上做重构误差监控。

DataLoader 构造规则：

- `train_loader`：读取 `label_<k>.npy`，训练分支默认 `batch_size=128`，shuffle。
- `test_loader`：同样读取 `label_<k>.npy`，用于 ID 分数或 ID loss。
- `ood_test_loader`：读取 `test.npy`，用于 OOD 分数或 OOD loss。
- 非训练分支中 `batch_size` 会被入口强制设为 `1`。

## 5. 模型结构

VAE 模型为 `vae.py` 中的 `MLP_VAE`。默认结构可以概括为：

```text
输入 x: INPUT_DIM
  |
  v
Encoder MLP: INPUT_DIM -> 4096 -> 2048 -> 1024 -> 1024
  |
  +--> mu:     1024 -> z_dim
  |
  +--> logvar: 1024 -> z_dim
  |
  v
重参数采样 z = mu + eps * exp(0.5 * logvar)
  |
  v
Decoder MLP: z_dim -> 1024 -> 1024 -> 2048 -> 4096
  |
  +--> recon_mu:       4096 -> INPUT_DIM
  |
  +--> recon_logvar:   4096 -> INPUT_DIM
```

默认 loss 为 `NLL`，包括两部分：

```text
loss = Gaussian negative log likelihood + KL divergence
```

其中重构方差会通过 `softclip` 限制范围，降低数值不稳定风险：

```python
recon_logvar = softclip(recon_logvar, -6.0)
recon_logvar = -softclip(-recon_logvar, 0.0)
```

分析阶段使用的分数是：

```text
logpx = -loss
```

因此分数越大，表示 VAE 认为样本越像当前模型学习到的标签分布。

## 6. 训练分支

入口逻辑：

```python
if args.if_train:
    piplinerunner = Piplineruner(args)
    piplinerunner.train()
    args.if_train = False
    piplinerunner = Piplineruner(args)
    piplinerunner.analyze_test()
```

训练分支分两步：

1. 构造训练模式的 `Piplineruner`，使用训练 batch size。
2. 调用 `train_test()` 训练 VAE。

`train_test()` 的核心循环是：

```python
for epoch in range(1, args.epochs + 1):
    VAEs.train(...)
    if epoch % args.log_interval == 0:
        VAEs.test(...)
```

每个 epoch 中，`VAEs.train()` 会：

1. 遍历 `train_loader`。
2. 前向计算 `recon_mu, recon_logsigma, mu, logvar`。
3. 计算 VAE loss。
4. 反向传播并调用 Adam 更新参数。
5. 按 `log_interval` 打印 batch 级训练 loss。
6. 如果 `epoch % save_interval == 0`，保存模型：

```text
result/DiHFT/vae_results/<dataset_name>/label_<k>/<epoch>.pth
result/DiHFT/vae_results/<dataset_name>/label_<k>/model_latest.pth
```

`VAEs.test()` 会分别计算：

- ID loss：来自 `label_<k>.npy`。
- OOD loss：来自 `test.npy`。

训练结束后，入口会把 `args.if_train` 改为 `False`，重新初始化一个 batch size 为 `1` 的 runner，并立即执行普通分析分支 `analyze_test()`。这样训练完成后会自动输出 `id_logpx.npy`、`ood_logpx.npy` 并打印 AUROC/AUPRC/FPR80。

## 7. 普通分析分支

当入口进入：

```python
elif not args.if_cross_analyze:
    piplinerunner = Piplineruner(args)
    piplinerunner.analyze_test()
```

或训练完成后的自动分析时，会执行普通分析。

`analyze_test()` 会加载当前标签目录下的：

```text
model_latest.pth
```

然后在三类数据上调用 `VAEs.analyze()`：

- `train_loader`：当前代码会计算 `train_mus, train_logpx`，但后续没有保存或使用。
- `test_loader`：作为 ID 分布，输出 `id_logpx`。
- `ood_test_loader`：作为 OOD 分布，输出 `ood_logpx`。

保存结果：

```text
result/DiHFT/vae_results/<dataset_name>/label_<k>/id_logpx.npy
result/DiHFT/vae_results/<dataset_name>/label_<k>/ood_logpx.npy
```

随后用 ID 分数和 OOD 分数计算二分类指标：

```text
y_true  = [ID, ID, ..., OOD, OOD, ...]
y_score = concat(id_logpx, ood_logpx)
```

因为 ID 和 OOD 数量可能不同，代码会先取两者最短长度 `min_len`，再截断对齐。

输出指标：

- `AUROC`：区分 ID/OOD 的 ROC AUC。
- `AUPRC`：PR AUC。
- `FPR80`：TPR 第一次达到 0.8 时的 FPR。

## 8. 交叉分析分支

默认入口会进入交叉分析：

```python
elif args.if_cross_analyze:
    piplinerunner = Piplineruner(args)
    piplinerunner.analyze_cross_test()
```

设计意图是：加载 `label_<label_index>` 训练出的 VAE，然后对所有标签数据 `label_0.npy` 到 `label_<total_label_number-1>.npy` 分别计算 log likelihood。

理想流程可以表示为：

```text
model_latest.pth for label_k
        |
        +--> label_0.npy -> ood_logpx_0.npy
        +--> label_1.npy -> ood_logpx_1.npy
        +--> label_2.npy -> ood_logpx_2.npy
        +--> label_3.npy -> ood_logpx_3.npy
        +--> label_4.npy -> ood_logpx_4.npy
```

`cross_analyze()` 会：

1. 加载 `model_latest.pth`。
2. 遍历传入的 dataloader list。
3. 对每个 label 数据计算 `ood_logpx`。
4. 保存为：

```text
result/DiHFT/vae_results/<dataset_name>/label_<k>/ood_logpx_0.npy
result/DiHFT/vae_results/<dataset_name>/label_<k>/ood_logpx_1.npy
...
```

这些交叉标签得分后续可被高层路由或拒绝机制使用，用来判断当前市场片段与各类动态标签的匹配程度。

## 9. 启动脚本关系

仓库中相关 shell 脚本包括：

```text
script/train/DiHFT/low_level/VAE.sh
script/train/DiHFT/low_level/VAE_util.sh
script/test/DiHFT/low_level/main_VAE_cross_analyze.sh
script/test/DiHFT/low_level/VAE_cross_analyze_util.sh
```

`VAE_util.sh` 和 `VAE_cross_analyze_util.sh` 都会对四个交易对、五个 label 并行启动：

```text
BNBUSDT label_0..4
BTCUSDT label_0..4
DOTUSDT label_0..4
ETHUSDT label_0..4
```

每个 `label_index` 使用：

```bash
CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/VAE/main.py \
    --dataset_name "${dataset_name}" \
    --label_index "${label_index}" \
    >"${log_dir}/train_label_${label_index}.log" 2>&1 &
```

日志默认写入：

```text
log/DiHFT/<dataset_name>/VAE/
```

## 10. 与高层模块的关系

高层路由相关脚本中多处默认读取：

```text
result/DiHFT/vae_results
```

例如：

```text
RL/DiHFT/high_level/vae_routing_util.py
RL/DiHFT/high_level/vae_routing_final_result_macro_action.py
RL/DiHFT/high_level/test_high_level_rejection.py
RL/DiHFT/ablation/safe_routing.py
```

因此 `main.py` 的主要产物不是直接交易策略，而是 VAE 模型和每个标签/测试样本的 log likelihood 文件。这些文件为高层策略提供“当前状态属于哪个市场动态”的辅助信号。

## 11. 当前代码注意点

以下是阅读代码时发现的实现细节或潜在问题，理解流程和排查运行错误时需要特别留意。

### 11.1 默认运行并不会训练

`main.py` 的默认参数是：

```text
if_train = False
if_cross_analyze = True
```

因此直接运行：

```bash
python RL/DiHFT/VAE/main.py
```

会进入交叉分析分支，而不是训练分支。若当前 label 目录下没有 `model_latest.pth`，会因为找不到模型文件而失败。

### 11.2 `argparse` 的 bool 参数容易误用

代码使用：

```python
parser.add_argument("--if_train", type=bool, default=False)
```

这种写法下，命令行传入 `--if_train False` 时，`bool("False")` 仍然是 `True`。所以字符串形式的 `False` 反而会开启训练。

更稳妥的写法通常是 `action="store_true"` / `action="store_false"`，或者自定义字符串转布尔函数。

### 11.3 `save_interval` 命令行传参可能导致类型问题

`save_interval` 定义为：

```python
parser.add_argument("--save_interval", type=str, default=50)
```

默认值是整数 `50`，训练时可以正常执行：

```python
epoch % args.save_interval
```

但如果命令行传入 `--save_interval 50`，解析后会变成字符串 `"50"`，再执行取模会报类型错误。这里更合理的类型应该是 `int`。

### 11.4 `learning_rate` 命令行传参类型不匹配

`learning_rate` 定义为：

```python
parser.add_argument("--learning_rate", type=int, default=1e-5)
```

默认值 `1e-5` 是浮点数；但如果从命令行传入 `--learning_rate 1e-5`，`int("1e-5")` 会解析失败。这里更合理的类型应该是 `float`。

### 11.5 CPU 设备字符串拼写为 `cpus`

`process.py` 中设备选择写成：

```python
torch.device("cuda" if torch.cuda.is_available() else "cpus")
```

PyTorch 的 CPU 设备名应为 `"cpu"`。如果没有 CUDA，该代码会因为 `"cpus"` 不是合法 device 而失败。

另外代码中：

```python
kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
```

`device` 是 `torch.device` 对象，和字符串 `"cuda"` 比较通常为 `False`，所以即使有 CUDA，也不会启用这组 DataLoader kwargs。

### 11.6 交叉分析当前会重复读取同一个 label

`analyze_cross_test()` 中循环写法是：

```python
for i in range(self.args.total_label_number):
    label_name = "label_{}".format(self.args.label_index)
```

这里循环变量 `i` 没有被用于构造 label 名，导致 `label_path_list` 里会重复加入当前 `label_index` 对应的同一个 `.npy` 文件。

如果设计目标是对所有标签做交叉分析，应该更像：

```python
label_name = "label_{}".format(i)
```

否则保存出的 `ood_logpx_0.npy` 到 `ood_logpx_4.npy` 名字不同，但内容都来自同一个 label 数据。

### 11.7 `VAEs.analyze()` 会提前停止一个 batch

`vae.py` 的 `analyze()` 中有：

```python
if batch_idx == len(data_loader) - 2:
    break
```

这会在倒数第二个 batch 处退出循环。若 `batch_size=1`，最终会少分析最后一个样本。若数据量很小，还可能导致采样行为和预期不一致。

### 11.8 `train_data` 和 `test_data` 使用同一个文件

`prepare_model()` 中：

```python
train_data = One_Dim_Dataset(train_data_path)
test_data = One_Dim_Dataset(train_data_path)
```

因此训练日志里的 ID test loss 并不是严格意义上的独立验证集 loss，而是同一份 label 数据上的重构监控。

## 12. 总结

`RL/DiHFT/VAE/main.py` 是 DiHFT 中 VAE 子系统的入口调度脚本。它围绕单个市场动态标签 `label_<k>` 构造 VAE，将该标签的数据作为 ID 分布进行拟合，再通过负 loss 形式的 `logpx` 评估测试集或其他标签数据与当前标签分布的相似度。

从数据流角度看，它的关键产物是：

```text
model_latest.pth
id_logpx.npy
ood_logpx.npy
ood_logpx_<i>.npy
```

这些结果会被后续高层路由、拒绝机制或消融实验读取，用于在不同市场动态之间进行选择或过滤。
