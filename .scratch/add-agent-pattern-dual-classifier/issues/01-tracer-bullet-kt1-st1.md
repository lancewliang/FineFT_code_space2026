# 01 — KT1/ST1 可信数据 tracer bullet

**What to build:** 打通一条可运行、可审计的最小纵切：从原始 volume 和合约身份传递、Detail 行与动作空间 sidecar，到 KT1 突破即时与 ST1 突破即时分类，最后产生一个具有稳定身份和守恒 PnL 的窗口行。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] 原始 volume 作为非模型列逐行原样传递，行数和行序不变，且不进入 State Feature。
- [ ] Detail 行包含非空合约、原始 volume、执行前后仓位、mark price、已实现 PnL、手续费和浮动 PnL。
- [ ] epoch sidecar 记录 epoch、最大持仓、仓位选项数、杠杆选项和完整有序 signed position 档位，且与评估环境一致。
- [ ] KT1 纯分类行为覆盖第 1–5 步基准、第 6–10 步突破、最终延伸和突破侧保持比例，并对上涨/下跌对称。
- [ ] ST1 纯分类行为覆盖突破当步或下一步的空仓至同向近满仓开仓，开仓观测计入至少 10 步同号持仓。
- [ ] 加仓、反手、平仓、变号或持仓观测不足的输入不被误判为 ST1；同号减仓不中断持仓计数。
- [ ] tracer 输出每个窗口恰好一行，K 线和策略形态为合法 JSON 数组，gross/net PnL 按执行账本计算且不重复扣除滑点。
- [ ] `window_id` 仅使用已约定的逻辑追溯字段生成，对阈值、Selection 状态和数据根目录变化保持稳定。
- [ ] 纯函数测试覆盖 KT1/ST1 主路径、多空对称和阈值边界；一个端到端测试证明数据能从上游到达窗口产物。
