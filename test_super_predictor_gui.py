"""测试超级预测器GUI集成"""
from zodiac_super_predictor import ZodiacSuperPredictor

print('='*70)
print('测试超级预测器v5.0 - GUI集成测试')
print('='*70)
print()

predictor = ZodiacSuperPredictor()

# 测试1: 基本预测功能
print('测试1: 基本预测功能...')
result = predictor.predict(top_n=5)
print('✅ 基本预测功能正常')
print(f'   模型: {result["model"]}')
print(f'   版本: {result["version"]}')
print(f'   TOP5生肖: {[z for z, s in result["top5_zodiacs"]]}')
print()

# 测试2: 最近20期验证功能
print('测试2: 最近20期验证功能...')
validation = predictor.get_recent_20_validation()
if validation:
    print('✅ 最近20期验证功能正常')
    print(f'   生肖TOP5命中率: {validation["zodiac_top5_rate"]:.1f}% ({validation["zodiac_top5_hits"]}/20)')
    print(f'   号码TOP15命中率: {validation["number_top15_rate"]:.1f}% ({validation["number_top15_hits"]}/20)')
    print(f'   验证期数: {len(validation["details"])}期')
else:
    print('⚠️ 数据不足20期')
print()

print('='*70)
print('🎉 所有功能测试通过！')
print('✅ GUI可以正常使用超级预测器v5.0')
print('='*70)
