# 生肖机器学习预测模型使用指南

## 📚 模型简介

生肖ML预测模型是一个**混合智能预测系统**，结合了：
- ✅ **统计分析逻辑**：频率分析、轮转规律、冷热度、周期性
- 🤖 **机器学习算法**：随机森林、XGBoost、LightGBM、梯度提升

通过智能融合两种方法的优势，提供更准确的生肖预测。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

如果某些库安装失败，模型会自动降级为纯统计模式。

### 2. 基本使用

```python
from zodiac_ml_predictor import ZodiacMLPredictor

# 创建预测器（默认ML权重40%）
predictor = ZodiacMLPredictor(ml_weight=0.4)

# 获取预测
result = predictor.predict()

# 查看结果
print(f"TOP6生肖: {[z for z, s in result['top6_zodiacs']]}")
print(f"推荐号码: {result['top18_numbers'][:12]}")
```

### 3. 命令行运行

```bash
python zodiac_ml_predictor.py
```

---

## ⚙️ 高级配置

### 调整ML权重

模型支持灵活的权重配比：

```python
# 纯统计模型（ML权重=0）
predictor = ZodiacMLPredictor(ml_weight=0.0)

# 平衡模式（ML权重=0.4，推荐）⭐
predictor = ZodiacMLPredictor(ml_weight=0.4)

# ML优先模式（ML权重=0.6）
predictor = ZodiacMLPredictor(ml_weight=0.6)

# 纯ML模式（ML权重=1.0）
predictor = ZodiacMLPredictor(ml_weight=1.0)
```

**建议**：
- 🟢 初学者：`ml_weight=0.4`（平衡）
- 🟡 进阶用户：`ml_weight=0.5-0.6`（ML偏重）
- 🔴 保守用户：`ml_weight=0.2-0.3`（统计为主）

### 显式训练模型

```python
predictor = ZodiacMLPredictor(ml_weight=0.4)

# 手动触发训练
predictor.train_models()

# 查看训练状态
print(f"已训练: {predictor.is_trained}")
print(f"模型数量: {len(predictor.models)}")
```

---

## 🔍 核心特性

### 1. 统计分析维度

- ✅ 多时间窗口频率（5/10/20/30/50期）
- ✅ 生肖轮转规律（相邻、对冲）
- ✅ 冷热度分析
- ✅ 周期性检测
- ✅ 连续性惩罚
- ✅ 热度均衡

### 2. 机器学习特征

提取 **100+ 维特征**：

| 特征类别 | 数量 | 说明 |
|---------|------|------|
| 频率特征 | 60 | 各生肖在不同时间窗口的出现次数 |
| 间隔特征 | 12 | 距离上次出现的期数 |
| 位置特征 | 13 | 生肖位置和相对距离 |
| 周期特征 | 24 | 平均周期和周期方差 |
| 其他特征 | 10+ | 号码分布、多样性等 |

### 3. 模型融合

采用**集成学习**策略：

```
最终评分 = 统计权重 × 统计评分 + ML权重 × ML预测概率
```

**优势**：
- 统计模型提供稳定基线
- ML模型捕捉复杂模式
- 加权融合提升准确率

---

## 📊 预测结果说明

### 返回字段

```python
result = {
    'model': '生肖预测模型(统计+机器学习)',
    'version': '2.0',
    'ml_enabled': True,              # ML是否启用
    'ml_weight': 0.4,                # ML权重
    'stat_weight': 0.6,              # 统计权重
    'total_periods': 325,            # 总期数
    'last_date': '2025/12/23',       # 最新日期
    'last_number': 42,               # 最新号码
    'last_zodiac': '鼠',             # 最新生肖
    'top6_zodiacs': [                # TOP6生肖
        ('龙', 12.45),
        ('虎', 11.23),
        # ...
    ],
    'top18_numbers': [5, 17, 29, ...],  # 推荐号码
    'all_scores': {...},             # 所有生肖评分
    'stat_scores': {...},            # 统计评分
    'ml_probs': {...}                # ML预测概率
}
```

### 解读

- **top6_zodiacs**：按评分排序的TOP6生肖
  - 前2个：强推⭐⭐
  - 第3-4个：推荐⭐
  - 第5-6个：备选✓

- **top18_numbers**：基于TOP6生肖推荐的号码
  - 前6个：强推
  - 第7-12个：推荐
  - 第13-18个：备选

---

## 🧪 测试与验证

### 运行测试套件

```bash
python test_zodiac_ml.py
```

测试内容：
1. ✅ 基本预测功能
2. ✅ 不同权重配比
3. ✅ 模型训练过程
4. ✅ 最近10期验证
5. ✅ 纯统计 vs 混合对比

### 自定义验证

```python
import pandas as pd
from zodiac_ml_predictor import ZodiacMLPredictor

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

# 回测最近N期
N = 20
correct = 0

for i in range(N):
    # 使用前面的数据预测
    train_df = df.iloc[:-N+i]
    actual = df.iloc[-N+i]['animal']
    
    # 保存临时数据
    train_df.to_csv('data/temp.csv', index=False, encoding='utf-8-sig')
    
    # 预测
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict(csv_file='data/temp.csv')
    
    predicted_top6 = [z for z, s in result['top6_zodiacs']]
    
    if actual in predicted_top6:
        correct += 1

print(f"TOP6命中率: {correct}/{N} = {correct/N*100:.1f}%")
```

---

## 💡 使用建议

### 选号策略

| 策略 | 选择范围 | 适用场景 |
|------|---------|---------|
| 保守型 | TOP2生肖 | 风险厌恶，追求稳定 |
| 平衡型 ⭐ | TOP3生肖 | 一般推荐 |
| 进取型 | TOP6生肖 + TOP12号码 | 追求高覆盖 |

### 权重调优

根据历史验证结果调整：

```python
# 测试不同权重
for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
    predictor = ZodiacMLPredictor(ml_weight=w)
    # 验证并记录结果
```

选择命中率最高的权重配比。

---

## 🔧 技术细节

### 机器学习模型

1. **随机森林** (Random Forest)
   - n_estimators=100
   - max_depth=10
   - 适合处理非线性关系

2. **梯度提升** (Gradient Boosting)
   - n_estimators=100
   - max_depth=5
   - learning_rate=0.1

3. **XGBoost**
   - 高性能梯度提升
   - 自动处理缺失值

4. **LightGBM**
   - 快速训练
   - 低内存占用

### 特征工程

```python
# 核心特征示例
features = {
    'freq_30_龙': 3,           # 龙在最近30期的出现次数
    'gap_龙': 5,               # 距离龙上次出现的期数
    'relative_pos_龙': 2,      # 龙相对上期生肖的位置
    'avg_cycle_龙': 8.5,       # 龙的平均出现周期
    'unique_zodiacs_10': 9,    # 最近10期的生肖多样性
    # ... 100+ 特征
}
```

### 数据要求

- **最少期数**：10期（用于特征提取）
- **推荐期数**：50期以上（模型训练）
- **理想期数**：100期以上（充分学习）

---

## ❓ 常见问题

### Q1: 机器学习库安装失败怎么办？

**A**: 模型会自动降级为纯统计模式，不影响基本功能。

```bash
# 可选：只安装核心库
pip install pandas numpy scikit-learn
```

### Q2: 为什么每次预测结果不同？

**A**: 如果启用了ML，模型有一定随机性。可设置随机种子：

```python
# 在模型类中已设置 random_state=42
```

### Q3: 如何提升预测准确率？

**A**: 
1. 增加训练数据量
2. 调整ML权重
3. 添加更多特征维度
4. 使用集成学习

### Q4: 可以预测多期吗？

**A**: 当前版本只支持单期预测。多期预测准确率会显著下降。

---

## 📈 性能预期

| 指标 | 理论值 | 目标值 | 说明 |
|------|-------|--------|------|
| TOP3命中率 | 25.0% | 30%+ | 超过理论5%+ |
| TOP6命中率 | 50.0% | 55%+ | 超过理论5%+ |
| TOP12号码命中率 | 24.5% | 30%+ | 超过理论5%+ |

*实际性能取决于数据质量和参数配置*

---

## 🔄 更新日志

### v2.0 (2025-12-24)
- ✅ 新增机器学习模块
- ✅ 实现统计+ML混合预测
- ✅ 支持多模型融合
- ✅ 添加100+维特征工程
- ✅ 提供灵活权重配置

### v1.0
- 基础统计分析模型

---

## 📞 支持

- 📧 问题反馈：提交Issue
- 📖 文档：参考本指南
- 🧪 测试：运行 `test_zodiac_ml.py`

---

**祝您使用愉快！** 🎉
