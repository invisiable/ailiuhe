"""
推荐生肖TOP4投注策略 - v2.1（连续未中切换版本）
新切换机制：连续4期未中就切换模型
"""

import pandas as pd
import numpy as np
from collections import Counter, deque
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from hybrid_adaptive_predictor import HybridAdaptivePredictor


class RecommendedZodiacTop4StrategyV2_1:
    """推荐的生肖TOP4投注策略 - 连续未中切换版本"""
    
    def __init__(self, use_emergency_backup=True, consecutive_miss_threshold=4):
        """
        Args:
            use_emergency_backup: 是否启用应急备份
            consecutive_miss_threshold: 连续未中多少期触发切换（默认4期）
        """
        # 主力模型：重训练v2.0
        self.primary_predictor = RetrainedZodiacPredictor()
        self.primary_model = self.primary_predictor  # 用于GUI显示
        
        # 备份模型：混合自适应v3.1
        self.backup_predictor = HybridAdaptivePredictor() if use_emergency_backup else None
        self.backup_model = self.backup_predictor  # 用于GUI显示
        
        # 当前使用的模型
        self.current_model = 'primary'  # 'primary' 或 'backup'
        
        # 性能监控
        self.recent_performance = deque(maxlen=10)  # 最近10期命中记录
        self.consecutive_misses = 0  # 连续未中次数
        self.consecutive_miss_threshold = consecutive_miss_threshold  # 触发切换的连续未中次数
        self.consecutive_hits_for_recovery = 2  # 连续命中几次后考虑切回主模型
        self.consecutive_hits = 0  # 连续命中次数
        self.switch_history = []  # 模型切换历史记录
        
        # 投注配置（固定1倍）
        self.bet_per_zodiac = 4  # 每个生肖4元
        self.num_zodiacs = 4  # 投注4个生肖
        self.total_bet_per_period = 16  # 每期总投注16元
        self.win_amount = 46  # 中奖金额46元
        self.net_profit_per_win = 30  # 净利润30元 (46-16)
        
    def predict_top4(self, animals):
        """预测TOP4生肖"""
        if self.current_model == 'primary':
            result = self.primary_predictor.predict_from_history(animals, top_n=4, debug=False)
            predictor_name = '重训练v2.0'
        else:
            result = self.backup_predictor.predict_from_history(animals, top_n=4, debug=False)
            predictor_name = result.get('predictor', '混合自适应v3.1')
        
        return {
            'top4': result['top4'],
            'predictor': predictor_name,
            'model': self.current_model
        }
    
    def update_performance(self, is_hit):
        """更新性能记录"""
        self.recent_performance.append(1 if is_hit else 0)
        
        # 更新连续未中/命中计数
        if is_hit:
            self.consecutive_misses = 0
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0
            self.consecutive_misses += 1
        
        # 同时更新备份模型的监控（如果存在）
        if self.backup_predictor:
            self.backup_predictor.update_performance(is_hit)
    
    def check_and_switch_model(self):
        """检查性能并决定是否切换模型（基于连续未中次数）"""
        if not self.backup_predictor:
            return False, "无备份模型"
        
        # 如果使用主模型且连续未中达到阈值，切换到备份
        if self.current_model == 'primary' and self.consecutive_misses >= self.consecutive_miss_threshold:
            old_model = self.current_model
            self.current_model = 'backup'
            switch_info = {
                'from': old_model,
                'to': self.current_model,
                'reason': f"主模型连续{self.consecutive_misses}期未中",
                'consecutive_misses': self.consecutive_misses
            }
            self.switch_history.append(switch_info)
            # 切换后重置计数器
            self.consecutive_misses = 0
            self.consecutive_hits = 0
            return True, switch_info['reason'] + "，切换到备份模型"
        
        # 如果使用备份模型且连续命中或表现恢复，切回主模型
        if self.current_model == 'backup':
            # 条件1：连续命中达到阈值
            if self.consecutive_hits >= self.consecutive_hits_for_recovery:
                old_model = self.current_model
                self.current_model = 'primary'
                switch_info = {
                    'from': old_model,
                    'to': self.current_model,
                    'reason': f"备份模型连续{self.consecutive_hits}期命中",
                    'consecutive_hits': self.consecutive_hits
                }
                self.switch_history.append(switch_info)
                # 切换后重置计数器
                self.consecutive_misses = 0
                self.consecutive_hits = 0
                return True, switch_info['reason'] + "，切回主模型"
            
            # 条件2：备份模型也连续未中太多，切回主模型试试
            if self.consecutive_misses >= self.consecutive_miss_threshold:
                old_model = self.current_model
                self.current_model = 'primary'
                switch_info = {
                    'from': old_model,
                    'to': self.current_model,
                    'reason': f"备份模型也连续{self.consecutive_misses}期未中",
                    'consecutive_misses': self.consecutive_misses
                }
                self.switch_history.append(switch_info)
                # 切换后重置计数器
                self.consecutive_misses = 0
                self.consecutive_hits = 0
                return True, switch_info['reason'] + "，切回主模型"
        
        return False, f"当前使用{self.current_model}模型，连续未中{self.consecutive_misses}期，连续命中{self.consecutive_hits}期，无需切换"
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.recent_performance:
            return {
                'recent_hits': 0,
                'recent_total': 0,
                'recent_rate': 0,
                'current_model': self.current_model,
                'consecutive_misses': self.consecutive_misses,
                'consecutive_hits': self.consecutive_hits
            }
        
        hits = sum(self.recent_performance)
        total = len(self.recent_performance)
        rate = hits / total * 100
        
        return {
            'recent_hits': hits,
            'recent_total': total,
            'recent_rate': rate,
            'current_model': self.current_model,
            'consecutive_misses': self.consecutive_misses,
            'consecutive_hits': self.consecutive_hits
        }
    
    def get_current_model_name(self):
        """获取当前模型的名称"""
        if self.current_model == 'primary':
            return '重训练v2.0（主力模型）'
        else:
            return '混合自适应v3.1（备份模型）'


if __name__ == '__main__':
    # 快速测试
    print("测试连续未中切换策略 v2.1")
    print("="*80)
    
    strategy = RecommendedZodiacTop4StrategyV2_1(
        use_emergency_backup=True,
        consecutive_miss_threshold=4
    )
    
    print(f"✓ 策略创建成功")
    print(f"  主力模型: {strategy.primary_model.__class__.__name__}")
    print(f"  备份模型: {strategy.backup_model.__class__.__name__ if strategy.backup_model else 'None'}")
    print(f"  切换阈值: 连续{strategy.consecutive_miss_threshold}期未中")
    print(f"  恢复条件: 连续{strategy.consecutive_hits_for_recovery}期命中")
    
    # 模拟测试
    print("\n模拟连续未中场景：")
    test_results = [False, False, False, False, True, False, True, True]
    
    for i, hit in enumerate(test_results):
        strategy.update_performance(hit)
        switched, msg = strategy.check_and_switch_model()
        stats = strategy.get_performance_stats()
        
        print(f"  第{i+1}期: {'✓中' if hit else '✗未中'} - 连续未中{stats['consecutive_misses']}期, "
              f"连续命中{stats['consecutive_hits']}期 - {msg}")
        
        if switched:
            print(f"    >>> {msg}")
