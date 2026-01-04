"""
快速预测工具 - 命令行版本
快速获取下一期预测结果
"""

from final_hybrid_predictor import FinalHybridPredictor


def quick_predict():
    """快速预测"""
    predictor = FinalHybridPredictor()
    
    print("\n" + "="*60)
    print("🔮 快速预测 - 下一期 TOP15")
    print("="*60)
    
    # 获取信息
    info = predictor.get_prediction_info()
    print(f"\n📅 最新一期: {info['latest_period']['date']} - 开出 {info['latest_period']['number']}")
    
    # 生成预测
    top15 = predictor.predict()
    top5 = top15[:5]
    
    print(f"\n🎯 TOP 5:  {top5}")
    print(f"📊 TOP 15: {top15}")
    
    print(f"\n💡 基于 {info['total_records']} 期历史数据")
    print(f"✓ 验证成功率: TOP15={info['success_rate']['top15']}")
    print("="*60 + "\n")
    
    return top15


if __name__ == '__main__':
    quick_predict()
