# 01: 终局边界语义解耦与 ADR-0060 正式确立 (Terminal Boundary Decoupling & ADR-0060 Adoption)

**What to build:** 确立三层边界解耦架构（数据预处理排除末尾 10 天交割真空、动态角色管理老仓退出、切片末尾按最后行情 MTM 盯市结算），维持高层交割月防御物理删除状态，并将 ADR-0060 恢复并置为 accepted（ADR-0058 维持 superseded）。更新终局边界前缀不变性测试契约，使得向后追加未来数据时绝不改变历史决策。

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] ADR-0060 正式确认 accepted，ADR-0058 保持 superseded，明确交割真空由预处理 10 天裁剪闭环、切片末尾按 MTM 盯市结算
- [x] 高层路由不设切片尾部提前平仓或空仓锁定规则，存续合约不因切片边界被提前锁定
- [x] 终局前缀不变性验证通过，追加未来行情时历史所有步的决策与风控掩码完全恒定
- [x] 切片最后一刻保留真实持仓状态与可用行情盯市估值，不产生虚构清算与手续费
