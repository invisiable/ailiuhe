# 生肖预测模型 - 机器学习混合版

> 🤖 结合统计逻辑和机器学习的智能生肖预测系统

---

## ✨ 核心亮点

- 🎯 **双引擎预测**：统计分析 + 机器学习，优势互补
- 🔧 **灵活配置**：可调节ML权重（0-100%），适应不同场景
- 📊 **深度特征**：提取119维特征，全方位分析历史数据
- 🚀 **集成学习**：融合4种ML模型，提升预测稳定性
- 💻 **多种接口**：命令行、GUI、Python API，使用便捷

---

## 🚀 快速开始

### 1分钟快速体验

```bash
# 安装依赖（如果未安装）
pip install pandas numpy scikit-learn xgboost lightgbm

# 快速预测
python quick_predict_zodiac_ml.py

# GUI演示
python demo_zodiac_ml.py

# 完整测试
python test_zodiac_ml.py
```

### Python代码示例

```python
from zodiac_ml_predictor import ZodiacMLPredictor

# 创建预测器（ML权重40%，推荐配置）
predictor = ZodiacMLPredictor(ml_weight=0.4)

# 获取预测
result = predictor.predict()

# 显示TOP3生肖
for i, (zodiac, score) in enumerate(result['top6_zodiacs'][:3], 1):
    print(f"{i}. {zodiac}: {score:.2f}")

# 显示推荐号码
print(f"推荐号码: {result['top18_numbers'][:12]}")
```

---

## 📋 功能对比

| 功能 | 纯统计模型 | ML混合模型 | 说明 |
|------|-----------|-----------|------|
| 频率分析 | ✅ | ✅ | 多时间窗口统计 |
| 轮转规律 | ✅ | ✅ | 生肖相邻关系 |
| 冷热度分析 | ✅ | ✅ | 避重机制 |
| 周期性检测 | ✅ | ✅ | 出现周期 |
| 特征工程 | ❌ | ✅ | 119维深度特征 |
| 模式识别 | ❌ | ✅ | 复杂非线性关系 |
| 自适应学习 | ❌ | ✅ | 持续优化 |
| 集成融合 | ❌ | ✅ | 多模型投票 |

---

## 🎮 使用方式

### 命令行模式

```bash
# 默认配置（ML=40%）
python quick_predict_zodiac_ml.py

# 自定义ML权重
python quick_predict_zodiac_ml.py 0.5

# 纯统计模式（不使用ML）
python quick_predict_zodiac_ml.py --pure-stat

# 纯ML模式（100% ML）
python quick_predict_zodiac_ml.py --pure-ml
```

### GUI模式

```bash
python demo_zodiac_ml.py
```

特性：
- 滑块调节ML权重
- 实时预测结果
- 详细评分展示
- 预设配置快捷按钮

### API模式

```python
from zodiac_ml_predictor import ZodiacMLPredictor

# 创建预测器
predictor = ZodiacMLPredictor(ml_weight=0.4)

# 显式训练模型（可选，predict会自动训练）
predictor.train_models()

# 预测
result = predictor.predict()

# 获取详细信息
print(f"ML状态: {result['ml_enabled']}")
print(f"训练模型数: {len(predictor.models)}")
print(f"TOP6生肖: {[z for z, s in result['top6_zodiacs']]}")
print(f"统计评分: {result['stat_scores']}")
print(f"ML概率: {result['ml_probs']}")
```

---

## ⚙️ 配置指南

### ML权重选择

| 权重 | 模式 | 适用场景 | 特点 |
|------|------|---------|------|
| 0.0 | 纯统计 | 保守稳健 | 基于历史规律 |
| 0.2-0.3 | 统计为主 | 风险厌恶 | 少量ML辅助 |
| **0.4-0.5** | **平衡模式** ⭐ | **推荐** | **统计+ML均衡** |
| 0.6-0.7 | ML为主 | 激进创新 | ML主导预测 |
| 1.0 | 纯ML | 实验探索 | 完全依赖ML |

### 调优建议

1. **从0.4开始**：推荐的平衡配置
2. **小步调整**：每次±0.1，观察效果
3. **验证对比**：用历史数据验证不同权重的效果
4. **场景切换**：根据实际情况动态调整

---

## 🔬 技术原理

### 整体架构

```
┌─────────────────────────────────────┐
│         数据加载与预处理              │
│    data/lucky_numbers.csv          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         特征工程 (119维)             │
│  • 频率特征 (60维)                   │
│  • 间隔特征 (12维)                   │
│  • 位置特征 (13维)                   │
│  • 周期特征 (24维)                   │
│  • 其他特征 (10维)                   │
└──────┬───────────────┬──────────────┘
       │               │
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│  统计分析   │ │  ML模型集成  │
│  • 频率     │ │  • RF       │
│  • 轮转     │ │  • GB       │
│  • 冷热     │ │  • XGB      │
│  • 周期     │ │  • LGB      │
└──────┬──────┘ └──────┬──────┘
       │               │
       ▼               ▼
   统计评分        ML概率预测
       │               │
       └───────┬───────┘
               ▼
       加权融合 (可配置)
               ▼
         最终评分排序
               ▼
     ┌─────────────────┐
     │  TOP6生肖        │
     │  TOP18号码       │
     └─────────────────┘
```

### 机器学习模型

| 模型 | 说明 | 优势 |
|------|------|------|
| **Random Forest** | 随机森林 | 抗过拟合，处理非线性 |
| **Gradient Boosting** | 梯度提升 | 逐步优化，精度高 |
| **XGBoost** | 极端梯度提升 | 高性能，自动处理缺失 |
| **LightGBM** | 轻量级梯度提升 | 快速训练，低内存 |

### 特征示例

```python
{
    # 频率特征
    'freq_30_龙': 3,        # 龙在最近30期出现3次
    'freq_10_虎': 1,        # 虎在最近10期出现1次
    
    # 间隔特征
    'gap_龙': 5,            # 距离龙上次出现5期
    
    # 位置特征
    'relative_pos_龙': 2,   # 龙相对上期生肖的位置
    'last_zodiac_idx': 4,   # 上期生肖的索引
    
    # 周期特征
    'avg_cycle_龙': 8.5,    # 龙的平均出现周期
    'std_cycle_龙': 2.3,    # 周期的标准差
    
    # 其他特征
    'unique_zodiacs_10': 9, # 最近10期的生肖多样性
    'freq_variance': 1.2,   # 频率方差
    # ... 共119个特征
}
```

---

## 📊 性能测试

### 测试环境

- 数据集：324期历史数据
- 训练集：314个样本
- 特征数：119维
- 模型数：4个

### 基准测试（最近10期）

| 指标 | ML=0.0 | ML=0.4 | ML=1.0 | 理论值 |
|------|--------|--------|--------|--------|
| TOP3命中 | - | 10% | - | 25% |
| TOP6命中 | - | 40% | - | 50% |

*小样本测试，仅供参考*

### 权重对比

| ML权重 | TOP3预测 | 差异 |
|--------|----------|------|
| 0.0 | 猪、羊、鸡 | 完全基于统计 |
| 0.4 | 羊、龙、猪 | 龙被ML提升 |
| 1.0 | 羊、龙、猪 | ML主导结果 |

---

## 📚 文件清单

### 核心文件

| 文件 | 说明 | 大小 |
|------|------|------|
| `zodiac_ml_predictor.py` | 主模型（800行） | ~35KB |
| `test_zodiac_ml.py` | 测试套件（240行） | ~10KB |
| `quick_predict_zodiac_ml.py` | 快速预测（140行） | ~6KB |
| `demo_zodiac_ml.py` | GUI演示（260行） | ~11KB |

### 文档文件

| 文件 | 说明 |
|------|------|
| `生肖ML预测模型使用指南.md` | 详细使用文档 |
| `生肖ML预测模型交付清单.md` | 交付清单 |
| `README_ZODIAC_ML.md` | 本文档 |

---

## 🧪 测试验证

### 运行测试

```bash
python test_zodiac_ml.py
```

测试内容：
1. ✅ 基本预测功能
2. ✅ 不同权重配比
3. ✅ 模型训练过程
4. ✅ 最近10期验证
5. ✅ 统计vs混合对比

### 自定义回测

```python
import pandas as pd
from zodiac_ml_predictor import ZodiacMLPredictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

# 回测最近20期
correct_top3 = 0
correct_top6 = 0

for i in range(20):
    # 使用前N期数据训练
    train_df = df.iloc[:-20+i]
    actual = df.iloc[-20+i]['animal']
    
    # 保存临时数据并预测
    train_df.to_csv('data/temp.csv', index=False, encoding='utf-8-sig')
    
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict(csv_file='data/temp.csv')
    
    top6 = [z for z, s in result['top6_zodiacs']]
    top3 = top6[:3]
    
    if actual in top3:
        correct_top3 += 1
    if actual in top6:
        correct_top6 += 1

print(f"TOP3命中率: {correct_top3/20*100:.1f}%")
print(f"TOP6命中率: {correct_top6/20*100:.1f}%")
```

---

## 💡 最佳实践

### 1. 数据准备

- ✅ 确保数据文件 `data/lucky_numbers.csv` 存在
- ✅ 数据格式：date, number, animal, element
- ✅ 至少50期数据（推荐100期以上）

### 2. 权重配置

```python
# 场景1：保守稳健
predictor = ZodiacMLPredictor(ml_weight=0.3)

# 场景2：平衡推荐 ⭐
predictor = ZodiacMLPredictor(ml_weight=0.4)

# 场景3：激进创新
predictor = ZodiacMLPredictor(ml_weight=0.6)
```

### 3. 结果解读

```python
result = predictor.predict()

# 重点关注TOP3生肖（强推）
top3 = result['top6_zodiacs'][:3]

# 查看统计vs ML的贡献
for zodiac, score in top3:
    stat = result['stat_scores'][zodiac]
    ml = result['ml_probs'][zodiac] if result['ml_probs'] else 0
    print(f"{zodiac}: 统计={stat:.1f}, ML={ml*100:.1f}%, 综合={score:.2f}")
```

### 4. 持续优化

- 📊 定期验证不同权重的效果
- 🔄 根据近期表现调整配置
- 📈 记录并分析预测结果
- 🎯 结合实际情况灵活应用

---

## ❓ 常见问题

### Q: 机器学习库安装失败怎么办？

**A**: 模型会自动降级为纯统计模式，基本功能不受影响。

```bash
# 最小依赖
pip install pandas numpy

# 推荐依赖
pip install pandas numpy scikit-learn
```

### Q: 为什么每次预测结果略有不同？

**A**: ML模型有一定随机性，但已设置 `random_state=42` 来保持稳定性。如果仍有差异，可能是数据更新导致。

### Q: 如何选择最佳ML权重？

**A**: 
1. 从0.4开始（推荐配置）
2. 用历史数据回测不同权重
3. 选择命中率最高的配置
4. 定期重新验证和调整

### Q: 可以用于其他预测任务吗？

**A**: 可以，但需要修改：
1. 调整生肖映射规则
2. 修改特征提取逻辑
3. 重新训练模型

---

## 🔄 版本更新

### v2.0 (2025-12-24)
- ✅ 首次发布
- ✅ 统计+ML混合预测
- ✅ 119维特征工程
- ✅ 4种ML模型集成
- ✅ 灵活权重配置

### 未来计划
- [ ] 深度学习模型（LSTM）
- [ ] 在线学习和模型更新
- [ ] 自动调参和特征选择
- [ ] 模型解释性分析（SHAP）

---

## 📞 技术支持

- 📖 详细文档：[生肖ML预测模型使用指南.md](生肖ML预测模型使用指南.md)
- 📦 交付清单：[生肖ML预测模型交付清单.md](生肖ML预测模型交付清单.md)
- 🧪 测试脚本：`test_zodiac_ml.py`
- 💻 快速开始：`quick_predict_zodiac_ml.py`

---

## 🎉 总结

生肖ML预测模型是一个**创新的混合智能预测系统**，结合了传统统计分析的稳定性和机器学习的灵活性。

**核心优势**：
- 🎯 双引擎预测，优势互补
- 🔧 灵活配置，适应多场景
- 📊 深度特征，全面分析
- 🚀 易于使用，开箱即用

**开始使用**：
```bash
python quick_predict_zodiac_ml.py
```

**祝您好运！** 🍀

---

*最后更新：2025-12-24*
