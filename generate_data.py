"""
生成幸运数字训练数据
包含日期、数字、生肖、五行元素
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 12生肖
animals = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']

# 五行
elements = ['金', '木', '水', '火', '土']

# 五行对应的数字
element_numbers = {
    '金': [3, 4, 11, 12, 25, 26, 33, 34, 41, 42],
    '木': [7, 8, 15, 16, 23, 24, 37, 38, 45, 46],
    '水': [13, 14, 21, 22, 29, 30, 43, 44],
    '火': [1, 2, 9, 10, 17, 18, 31, 32, 39, 40, 47, 48],
    '土': [5, 6, 19, 20, 27, 28, 35, 36, 49]
}

# 反向映射：从数字到五行
number_to_element = {}
for element, numbers in element_numbers.items():
    for num in numbers:
        number_to_element[num] = element

# 设置随机种子
np.random.seed(42)

# 生成300天的历史数据
base_date = datetime(2024, 1, 1)
data = []

# 初始幸运数字
current_number = 25

for i in range(300):
    date = base_date + timedelta(days=i)
    
    # 幸运数字 - 带有一定规律的随机变化
    change = np.random.choice([-5, -3, -2, -1, 0, 1, 2, 3, 5, 8], 
                              p=[0.05, 0.1, 0.15, 0.2, 0.1, 0.2, 0.1, 0.05, 0.03, 0.02])
    current_number = max(1, min(49, current_number + change))
    number = int(current_number)
    
    # 生肖 - 按日期循环
    animal = animals[i % 12]
    
    # 五行 - 根据数字确定（使用数字对应的五行）
    element = number_to_element.get(number, elements[i % 5])  # 如果数字没有对应五行，按循环
    
    data.append({
        'date': date.strftime('%Y-%m-%d'),
        'number': number,
        'animal': animal,
        'element': element
    })

# 创建DataFrame并保存
df = pd.DataFrame(data)
df.to_csv('data/lucky_numbers.csv', index=False, encoding='utf-8-sig')

print(f"✅ 幸运数字数据生成成功！")
print(f"📊 数据量: {len(df)} 行")
print(f"📊 数字范围: {df['number'].min()} - {df['number'].max()}")
print(f"📊 平均值: {df['number'].mean():.2f}")
print(f"\n生肖列表: {', '.join(animals)}")
print(f"五行列表: {', '.join(elements)}")
print(f"\n五行与数字对应关系:")
for element, numbers in element_numbers.items():
    print(f"  {element}: {numbers}")
print(f"\n前10行数据预览:")
print(df.head(10))
print(f"\n后10行数据预览:")
print(df.tail(10))
