# refactor-multi-contract-scale-save-robust-scaler

## 背景与目标

当前 commodity split-stage 的 `muti_contract_scale_save.py` 仍然按文件独立拟合缩放参数，导致同一特征在不同合约、不同 split 上出现离散跳档，进而制造了 VAE 的假 OOD。已确认原始 CSV 的行情单位正常，问题出在缩放阶段。

本变更的目标是把 `muti_contract_scale_save.py` 改成一个可复现、train-only、可审计的 scaler 模块，使用 train 全量拟合的 robust scaler，并原地覆盖现有 `SCALE_SAVE` 输出。

## 用户场景

1. 作为数据预处理流程的维护者，我希望 `train/valid/test` 使用同一套缩放参数，避免 test 合约因为独立 fit 而被错误拉到不同尺度。
2. 作为模型训练使用者，我希望 VAE 输入分布稳定，减少由预处理引入的假 OOD。
3. 作为排障人员，我希望能够从 manifest 和 diagnostics 追溯每个 state feature 的 center、scale、fallback 和 clip 情况。

## 设计方向

推荐方案是重构 `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`，保留其 CLI 和输出路径不变，但内部改为两阶段：

1. 仅扫描 `train/*.feather`，对所有 train 样本的 selected state features 进行全量 fit。
2. 对 train/valid/test 的每个输入文件统一 apply 这套 train-only scaler，输出仍写到 `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather`，并同步写同 basename `.csv`。

Scaler 采用 robust 方案：

- `center = train_median`
- `scale = train_iqr = q75 - q25`
- `scale` 过小时 fallback 到 `train_std`
- `std` 也过小时 fallback 到 `1.0`
- 默认启用 clip，范围 `[-20, 20]`

同时落盘两类审计文件：

- `SCALE_SAVE/{symbol}/{target_freq}/scaler_manifest.json`
- `SCALE_SAVE/{symbol}/{target_freq}/scale_diagnostics.csv`

## 关键决策

- 只修改 `muti_contract_scale_save.py`，不修改老版 `scale_save.py`。
- 缩放参数只从 train 全量拟合一次，valid/test 只能 apply，不能重新 fit。
- 输出路径原地覆盖现有 `SCALE_SAVE` 目录结构。
- 默认使用 robust scaler + clip `[-20, 20]`，并允许 CLI 覆盖或关闭 clip。
- manifest 记录每个 feature 的 center、scale、方法、fallback 与 clip 配置。
- diagnostics 记录每个输出文件的 clip 比例和基础统计，便于审计。

## 范围边界

**包含：**
- `muti_contract_scale_save.py` 的缩放语义重构
- train-only scaler fit/apply 分离
- manifest 与 diagnostics 落盘
- 新增或更新相关测试

**不包含（本次）：**
- `scale_save.py` 老逻辑改写
- full process 调度调整
- VAE 训练逻辑修改
- 原始 CSV 下载或行情清洗逻辑修改

## 验收标准

- [ ] `muti_contract_scale_save.py` 仍可原地输出 `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather/.csv`
- [ ] train/valid/test 使用同一套 train-only scaler 参数
- [ ] `wap_1` / `awap` 不再出现按文件独立 fit 导致的 10 倍尺度跳档
- [ ] `scaler_manifest.json` 和 `scale_diagnostics.csv` 能完整记录 fit/apply 结果
- [ ] 现有 scale-save 相关测试更新后通过，并新增 train-only robust scaler 回归测试
