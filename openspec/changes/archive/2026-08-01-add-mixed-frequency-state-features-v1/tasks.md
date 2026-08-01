# 任务列表：add-mixed-frequency-state-features-v1

- [x] 新增 `Mixed-frequency State Feature` 生成逻辑，支持上一 `TradingDay` 和上一完整自然周特征
- [x] 新增 `MIXED_FREQUENCY_BASE` 生成逻辑，日基础数据一天一行，周基础数据一周一行
- [x] 将日基础数据与周基础数据拆成独立代码入口生成
- [x] 将日基础数据与周基础数据分别编排为 commodity full process shell 步骤
- [x] 将日混频特征与周混频特征拆成独立代码入口生成
- [x] 将日混频特征与周混频特征分别编排为 commodity full process shell 步骤
- [x] 实现上一日特征：`prev_day_return`、`prev_day_range_pct`、`prev_day_body_pct`、`prev_day_upper_shadow_pct`、`prev_day_lower_shadow_pct`、`prev_day_volume`、`prev_day_tradeval`、`prev_day_open_interest_change`、`prev_day_turnover_rate`
- [x] 实现上一自然周特征：`prev_week_return`、`prev_week_range_pct`、`prev_week_body_pct`、`prev_week_volume`、`prev_week_tradeval`、`prev_week_open_interest_change`、`prev_week_turnover_rate`
- [x] 在 daily merge 中支持按 `timestamp` join 混频特征到 future/state candidate feature frame
- [x] 在 commodity full process 脚本中编排混频特征生成，并在 merge 阶段要求混频特征产物存在
- [x] 确保混频特征不进入 Reward/Execution frame
- [x] 对缺失必要输入列、非数值价格、非有限输出执行 fail-fast 或 deterministic finite fallback
- [x] 编写上一 `TradingDay` 因果映射测试
- [x] 编写上一完整自然周因果映射测试，覆盖周一和周中 bar
- [x] 编写 daily merge 行为测试，验证混频列只进入 future/state candidate feature frame
- [x] 运行相关 commodity futures feature engineering 测试
