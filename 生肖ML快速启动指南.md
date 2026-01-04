# 🚀 生肖ML预测模型 - 快速启动指南

> 5分钟上手机器学习混合预测系统

---

## 📦 第一步：安装依赖

```bash
# 完整安装（推荐）
pip install pandas numpy scikit-learn xgboost lightgbm

# 最小安装（仅统计模式）
pip install pandas numpy
```

---

## ⚡ 第二步：快速开始

### 方式1: 一键预测（最简单）

```bash
python quick_predict_zodiac_ml.py
```

**输出示例：**
```
🤖 生肖预测 - 机器学习混合模型
================================

⭐ 生肖预测 TOP 6:
⭐⭐ 1. 羊 [强推]  综合评分: 1.79
⭐⭐ 2. 龙 [强推]  综合评分: 1.50
⭐ 3. 猪 [推荐]  综合评分: 1.09

📋 推荐号码 TOP 18:
   强推 (1-6):   [20, 32, 5, 17, 29, 41]
   推荐 (7-12):  [12, 36, 48, 44, 9, 21]
```

### 方式2: GUI界面（最直观）

```bash
python demo_zodiac_ml.py
```

**功能：**
- ✅ 滑块调节ML权重
- ✅ 预设配置快捷按钮
- ✅ 实时预测结果
- ✅ 详细评分对比

### 方式3: Python代码（最灵活）

```python
from zodiac_ml_predictor import ZodiacMLPredictor

# 创建预测器
predictor = ZodiacMLPredictor(ml_weight=0.4)

# 获取预测
result = predictor.predict()

# 查看结果
print(f"TOP6生肖: {[z for z, s in result['top6_zodiacs']]}")
print(f"推荐号码: {result['top18_numbers'][:12]}")
```

---

## 🎯 第三步：理解结果

### 预测结果说明

```python
result = {
    'top6_zodiacs': [
        ('羊', 1.79),  # (生肖, 综合评分)
        ('龙', 1.50),
        # ...
    ],
    'top18_numbers': [20, 32, 5, 17, ...],  # 推荐号码
    'ml_enabled': True,      # ML是否启用
    'stat_scores': {...},    # 统计评分
    'ml_probs': {...},       # ML预测概率
}
```

### 如何选号？

| 策略 | 选择 | 说明 |
|------|------|------|
| **保守型** | TOP2生肖 | 6-10个号码，稳健 |
| **平衡型** ⭐ | TOP3生肖 | 9-15个号码，推荐 |
| **进取型** | TOP12号码 | 12个号码，高覆盖 |

---

## ⚙️ 第四步：调整配置

### 不同ML权重

```bash
# 纯统计模式（ML=0%）
python quick_predict_zodiac_ml.py --pure-stat

# 平衡模式（ML=40%，推荐）
python quick_predict_zodiac_ml.py

# ML优先（ML=60%）
python quick_predict_zodiac_ml.py 0.6

# 纯ML模式（ML=100%）
python quick_predict_zodiac_ml.py --pure-ml
```

### Python代码调整

```python
# 纯统计
predictor = ZodiacMLPredictor(ml_weight=0.0)

# 平衡模式（推荐）⭐
predictor = ZodiacMLPredictor(ml_weight=0.4)

# ML优先
predictor = ZodiacMLPredictor(ml_weight=0.6)

# 纯ML
predictor = ZodiacMLPredictor(ml_weight=1.0)
```

---

## 🧪 第五步：测试验证

### 运行完整测试

```bash
python test_zodiac_ml.py
```

**测试内容：**
- ✅ 基本预测功能
- ✅ 不同权重对比
- ✅ 模型训练过程
- ✅ 最近10期验证
- ✅ 统计vs混合对比

### 运行综合示例

```bash
python examples_zodiac_ml.py
```

**示例内容：**
- 示例1: 基础使用
- 示例2: 不同权重对比
- 示例3: 详细信息获取
- 示例4: 手动训练模型
- 示例5: 统计vs混合对比
- 示例6: 号码推荐策略
- 示例7: 简单验证
- 示例8: 所有生肖评分

---

## 📊 常用命令速查

| 命令 | 说明 |
|------|------|
| `python quick_predict_zodiac_ml.py` | 快速预测 |
| `python quick_predict_zodiac_ml.py 0.5` | 自定义权重 |
| `python quick_predict_zodiac_ml.py --pure-stat` | 纯统计 |
| `python demo_zodiac_ml.py` | GUI界面 |
| `python test_zodiac_ml.py` | 完整测试 |
| `python examples_zodiac_ml.py` | 综合示例 |

---

## 💡 使用技巧

### 技巧1: 找到最佳权重

```python
# 测试不同权重
for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
    predictor = ZodiacMLPredictor(ml_weight=w)
    result = predictor.predict()
    top3 = [z for z, s in result['top6_zodiacs'][:3]]
    print(f"ML={w}: {top3}")
```

### 技巧2: 查看详细评分

```python
result = predictor.predict()

# 对比统计vs ML
for zodiac, final in result['top6_zodiacs'][:3]:
    stat = result['stat_scores'][zodiac]
    ml = result['ml_probs'][zodiac]
    print(f"{zodiac}: 统计={stat:.1f}, ML={ml*100:.1f}%, 综合={final:.2f}")
```

### 技巧3: 历史验证

```python
import pandas as pd

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

# 回测最近N期
N = 10
for i in range(N):
    train_df = df.iloc[:-N+i]
    actual = df.iloc[-N+i]['animal']
    
    train_df.to_csv('data/temp.csv', index=False, encoding='utf-8-sig')
    
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict(csv_file='data/temp.csv')
    
    predicted = [z for z, s in result['top6_zodiacs']]
    hit = "✓" if actual in predicted else "✗"
    print(f"期{i+1}: 实际={actual}, 预测={predicted[:3]}, {hit}")
```

---

## ❓ 常见问题

### Q1: 提示缺少库怎么办？

```bash
# 完整安装
pip install pandas numpy scikit-learn xgboost lightgbm

# 如果失败，尝试最小安装
pip install pandas numpy scikit-learn
```

### Q2: 如何查看模型信息？

```python
predictor = ZodiacMLPredictor(ml_weight=0.4)
predictor.train_models()  # 显式训练

print(f"模型数量: {len(predictor.models)}")
print(f"模型列表: {list(predictor.models.keys())}")
```

### Q3: 如何保存预测结果？

```python
result = predictor.predict()

# 保存到文件
import json
with open('prediction_result.json', 'w', encoding='utf-8') as f:
    # 转换为可序列化格式
    save_data = {
        'top6_zodiacs': [[z, float(s)] for z, s in result['top6_zodiacs']],
        'top18_numbers': result['top18_numbers'],
        'ml_enabled': result['ml_enabled'],
    }
    json.dump(save_data, f, ensure_ascii=False, indent=2)
```

---

## 📚 深入学习

### 详细文档

- 📖 [使用指南](生肖ML预测模型使用指南.md) - 详细配置和高级功能
- 📦 [交付清单](生肖ML预测模型交付清单.md) - 完整功能列表
- 🎓 [开发总结](生肖ML模型开发完成总结.md) - 技术细节
- 📘 [README](README_ZODIAC_ML.md) - 快速入门

### 代码示例

- `examples_zodiac_ml.py` - 8个完整示例
- `test_zodiac_ml.py` - 完整测试套件
- `demo_zodiac_ml.py` - GUI示例

---

## 🎯 下一步

1. **选择使用方式**：命令行 / GUI / Python代码
2. **运行第一个预测**：获得TOP6生肖和推荐号码
3. **实验不同权重**：找到适合自己的配置
4. **历史验证**：用过往数据测试准确率
5. **实际应用**：结合实际情况灵活使用

---

## ✅ 检查清单

- [ ] 已安装必要的依赖库
- [ ] 已成功运行 `quick_predict_zodiac_ml.py`
- [ ] 理解TOP6生肖和推荐号码的含义
- [ ] 知道如何调整ML权重
- [ ] 尝试过不同的配置和对比

---

**准备就绪！开始使用吧！** 🚀

有问题？查看详细文档或运行示例代码。

祝您预测顺利！ 🍀
