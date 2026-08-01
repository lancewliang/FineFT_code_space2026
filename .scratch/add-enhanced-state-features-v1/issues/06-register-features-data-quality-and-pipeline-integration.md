# 06 — 特征注册、质量校验与全流水线集成

**What to build:** 将 8 类增强特征统一注册至 `expected_columns.py` 与 `DataQualityValidator`，实现 Fail-fast / NaN 校验，并确保特征贯穿 concat、Dataset Split、Feature Selection 及 Scale Save 全流程。

**Blocked by:** 01, 02, 03, 04, 05

**Status:** ready-for-agent

- [ ] 在 `expected_columns.py` 中注册所有增强特征名称模板。
- [ ] 在 `DataQualityValidator` 中接入增强特征的无非法值校验。
- [ ] 验证特征成功进入候选特征库、拼接处理、Feature Selection 矩阵以及 Scale Save 裁剪导出。
