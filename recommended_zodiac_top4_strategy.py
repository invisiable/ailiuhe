"""
推荐生肖TOP4投注策略 - v2.0
基于100期验证结果，使用最佳模型（重训练v2.0）+ 固定1倍投注
"""

import pandas as pd
import numpy as np
from collections import Counter, deque
from retrained_zodiac_predictor import RetrainedZodiacPredictor
from hybrid_adaptive_predictor import HybridAdaptivePredictor


class RecommendedZodiacTop4Strategy:
    """推荐的生肖TOP4投注策略"""
    
    def __init__(self, use_emergency_backup=True):
        """
        Args:
            use_emergency_backup: 是否启用应急备份（在主模型失效时切换到混合模型）
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
        self.check_period = 10  # 每10期检查一次
        self.emergency_threshold = 0.30  # 命中率低于30%触发应急
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
        
        # 同时更新备份模型的监控（如果存在）
        if self.backup_predictor:
            self.backup_predictor.update_performance(is_hit)
    
    def check_and_switch_model(self):
        """检查性能并决定是否切换模型"""
        if not self.backup_predictor:
            return False, "无备份模型"
        
        if len(self.recent_performance) < 5:
            return False, "数据不足"
        
        # 计算最近10期（或更少）的命中率
        recent_hits = sum(self.recent_performance)
        recent_total = len(self.recent_performance)
        recent_rate = recent_hits / recent_total
        
        # 如果使用主模型且表现不佳，切换到备份
        if self.current_model == 'primary' and recent_rate < self.emergency_threshold:
            old_model = self.current_model
            self.current_model = 'backup'
            switch_info = {
                'from': old_model,
                'to': self.current_model,
                'reason': f"主模型命中率{recent_rate*100:.1f}%低于{self.emergency_threshold*100:.0f}%",
                'recent_rate': recent_rate
            }
            self.switch_history.append(switch_info)
            return True, switch_info['reason'] + "，切换到备份模型"
        
        # 如果使用备份模型且表现恢复，切回主模型
        if self.current_model == 'backup' and recent_rate >= 0.45:
            old_model = self.current_model
            self.current_model = 'primary'
            switch_info = {
                'from': old_model,
                'to': self.current_model,
                'reason': f"备份模型表现恢复({recent_rate*100:.1f}%)",
                'recent_rate': recent_rate
            }
            self.switch_history.append(switch_info)
            return True, switch_info['reason'] + "，切回主模型"
        
        return False, f"当前使用{self.current_model}模型，命中率{recent_rate*100:.1f}%，无需切换"
    
    def get_performance_stats(self):
        """获取性能统计"""
        if not self.recent_performance:
            return {
                'recent_hits': 0,
                'recent_total': 0,
                'recent_rate': 0,
                'current_model': self.current_model
            }
        
        hits = sum(self.recent_performance)
        total = len(self.recent_performance)
        rate = hits / total * 100
        
        return {
            'recent_hits': hits,
            'recent_total': total,
            'recent_rate': rate,
            'current_model': self.current_model
        }
    
    def get_current_model_name(self):
        """获取当前模型的名称"""
        if self.current_model == 'primary':
            return '重训练v2.0（主力模型）'
        else:
            return '混合自适应v3.1（备份模型）'
    
    def backtest(self, csv_file='data/lucky_numbers.csv', test_periods=100, start_from_end=True):
        """
        回测策略
        Args:
            csv_file: 数据文件
            test_periods: 测试期数
            start_from_end: 是否从最后往前测试
        """
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        if len(df) < test_periods + 50:
            raise ValueError(f"数据不足，需要至少{test_periods + 50}期")
        
        # 确定测试范围
        if start_from_end:
            start_idx = len(df) - test_periods
            end_idx = len(df)
        else:
            start_idx = 50
            end_idx = 50 + test_periods
        
        results = []
        balance = 0
        total_invested = 0
        total_won = 0
        model_switches = []
        
        print(f"\n{'='*80}")
        print(f"生肖TOP4投注策略回测 - 推荐策略v2.0")
        print(f"{'='*80}\n")
        print(f"测试期数: 第{start_idx+1}-{end_idx}期（共{test_periods}期）")
        print(f"主力模型: 重训练v2.0（100期验证命中率47%）")
        print(f"投注方式: 固定1倍（16元/期）")
        print(f"应急机制: {'启用' if self.backup_predictor else '禁用'}")
        print(f"{'='*80}\n")
        
        for i in range(start_idx, end_idx):
            period = i + 1
            
            # 使用i之前的数据进行预测
            history_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
            
            # 预测
            prediction = self.predict_top4(history_animals)
            top4 = prediction['top4']
            predictor_name = prediction['predictor']
            
            # 实际结果
            actual_zodiac = str(df.iloc[i]['animal']).strip()
            actual_date = df.iloc[i]['date']
            actual_number = df.iloc[i]['number']
            
            # 判断命中
            is_hit = actual_zodiac in top4
            
            # 计算盈亏
            bet_amount = self.total_bet_per_period
            win_amount = self.win_amount if is_hit else 0
            profit = win_amount - bet_amount
            
            balance += profit
            total_invested += bet_amount
            if is_hit:
                total_won += win_amount
            
            # 更新性能
            self.update_performance(is_hit)
            
            # 每10期检查是否需要切换模型
            if (i - start_idx + 1) % self.check_period == 0:
                switched, reason = self.check_and_switch_model()
                if switched:
                    model_switches.append({
                        'period': period,
                        'reason': reason,
                        'new_model': prediction['predictor']
                    })
            
            results.append({
                'period': period,
                'date': actual_date,
                'actual_number': actual_number,
                'actual_zodiac': actual_zodiac,
                'predicted_top4': ', '.join(top4),
                'is_hit': is_hit,
                'bet': bet_amount,
                'win': win_amount,
                'profit': profit,
                'balance': balance,
                'predictor': predictor_name
            })
            
            # 每20期显示进度
            if (i - start_idx + 1) % 20 == 0:
                stats = self.get_performance_stats()
                print(f"已回测 {i - start_idx + 1}/{test_periods} 期 | "
                      f"累计余额: {balance:+d}元 | "
                      f"最近{stats['recent_total']}期命中率: {stats['recent_rate']:.1f}%")
        
        # 统计结果
        hits = sum(1 for r in results if r['is_hit'])
        hit_rate = hits / len(results) * 100
        roi = (balance / total_invested) * 100 if total_invested > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"【回测结果】")
        print(f"{'='*80}\n")
        print(f"测试期数: {len(results)}期")
        print(f"命中次数: {hits}次")
        print(f"命中率: {hit_rate:.1f}%")
        print(f"理论命中率: 33.3%")
        print(f"vs理论值: {hit_rate - 33.3:+.1f}%\n")
        
        print(f"总投注: {total_invested}元")
        print(f"总中奖: {total_won}元")
        print(f"净盈利: {balance:+d}元")
        print(f"ROI: {roi:.1f}%\n")
        
        # 模型切换记录
        if model_switches:
            print(f"模型切换记录: {len(model_switches)}次\n")
            for switch in model_switches:
                print(f"  第{switch['period']}期: {switch['reason']}")
        else:
            print("全程使用主模型，无切换\n")
        
        # 分段分析
        print(f"{'='*80}")
        print(f"【分段分析】（每20期）")
        print(f"{'='*80}\n")
        
        for i in range(0, len(results), 20):
            segment = results[i:i+20]
            seg_hits = sum(1 for r in segment if r['is_hit'])
            seg_rate = seg_hits / len(segment) * 100
            seg_profit = sum(r['profit'] for r in segment)
            start_p = segment[0]['period']
            end_p = segment[-1]['period']
            
            bar = '█' * int(seg_rate / 5)
            print(f"第{start_p:>3}-{end_p:<3}期: {seg_hits:>2}/{len(segment)} ({seg_rate:>5.1f}%) "
                  f"盈利{seg_profit:>+4}元 {bar}")
        
        print(f"\n{'='*80}")
        
        # 保存结果
        results_df = pd.DataFrame(results)
        output_file = f'recommended_zodiac_top4_backtest_{test_periods}periods.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 详细结果已保存: {output_file}\n")
        
        return {
            'results': results,
            'hits': hits,
            'total': len(results),
            'hit_rate': hit_rate,
            'balance': balance,
            'roi': roi,
            'model_switches': model_switches
        }


if __name__ == '__main__':
    print("="*80)
    print("推荐生肖TOP4投注策略测试")
    print("="*80)
    
    # 创建策略实例
    strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
    
    # 回测最近100期
    result = strategy.backtest(test_periods=100, start_from_end=True)
    
    print("\n【策略说明】")
    print("="*80)
    print("1. 主力模型: 重训练v2.0（47%命中率，ROI 35.1%）")
    print("2. 投注方式: 固定1倍（16元/期）")
    print("3. 应急机制: 连续10期命中率<30%自动切换到混合模型")
    print("4. 风险控制: 低成本、稳收益、可持续")
    print("="*80)
