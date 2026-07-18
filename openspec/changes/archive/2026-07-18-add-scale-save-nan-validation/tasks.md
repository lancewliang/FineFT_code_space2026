## 1. Implementation

- [x] 1.1 Add focused scale-save CLI tests covering input-stage NaN failure, output-stage NaN failure, and the existing successful output path. <!-- 已实现: 新增两个 CLI 失败路径测试并复用 helper 保留成功路径覆盖 -->
- [x] 1.2 Add a small Polars DataFrame NaN validation helper in `data_preprocess/operator_futures/scale_describe_save/scale_save.py` that reports stage, path, and NaN columns. <!-- 已实现: 新增浮点列 NaN 检测和 ValueError 报错 helper -->
- [x] 1.3 Call the validation helper immediately after reading the main input feather and after building final `out`, before any output file is written. <!-- 已实现: 在 input 读取后和 output 写入前调用校验 helper -->
- [x] 1.4 Run focused scale-save tests and a syntax/import check under the `finetf` conda environment. <!-- 已实现: 聚焦 pytest、py_compile 和 OpenSpec strict 校验均通过 -->
