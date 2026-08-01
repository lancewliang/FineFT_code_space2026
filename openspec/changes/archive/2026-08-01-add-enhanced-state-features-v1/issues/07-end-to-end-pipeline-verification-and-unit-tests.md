# 07 — 端到端测试套件与 Feature Selection / Scale Save 验证

**What to build:** 编写完整单元测试套件 `test_enhanced_state_features.py`，并运行商品期货 10min/30min 数据的完整预处理、特征选择与 Scale Save 端到端验证。

**Blocked by:** 06 — 特征注册、质量校验与全流水线集成

**Status:** ready-for-agent

- [ ] 编写 `test_enhanced_state_features.py` 覆盖 8 类增强特征的算子输入输出与边缘测试。
- [ ] 运行 Pytest 全部测试用例并保持绿灯。
- [ ] 在测试数据集上完整跑通 10min 和 30min 的 Feature Selection 与 Scale Save，产出正确的 manifest 与包含增强特征的精简数据集。
