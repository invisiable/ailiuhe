#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于命中概率预测的动态倍投策略

核心思想：
1. 根据历史数据和近期表现预测下一期命中概率
2. 根据预测概率动态调整倍投倍数
3. 高概率期加大投注，低概率期降低投注
"""

import numpy as np
import pandas as pd
from collections import deque


class ProbabilityBettingStrategy:
    """基于概率预测的动态倍投策略"""
    
    def __init__(self, config):
        """
        初始化策略
        
        Args:
            config: 配置字典
                - base_bet: 基础投注金额
                - win_reward: 中奖奖励
                - max_multiplier: 最大倍数限制
                - lookback: 回看期数（用于特征提取）
                - prob_high_thresh: 高概率阈值（如0.40）
                - prob_low_thresh: 低概率阈值（如0.25）
                - high_mult: 高概率倍数系数（如1.5）
                - low_mult: 低概率倍数系数（如0.5）
        """
        self.cfg = config
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.fib_index = 0
        
        # 历史记录
        self.hit_history = deque(maxlen=config.get('lookback', 20))
        self.prob_history = []  # 记录预测概率和实际结果
        
        # 统计信息
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
        
        # 特征权重（可调整）
        self.weights = {
            'recent_rate': 0.4,      # 近期命中率权重
            'trend': 0.3,            # 趋势权重
            'streak_factor': 0.2,    # 连续性因素权重
            'global_rate': 0.1       # 全局命中率权重
        }
    
    def predict_next_probability(self):
        """
        预测下一期的命中概率
        
        基于以下特征：
        1. 近期命中率（最近N期）
        2. 趋势（命中率变化方向）
        3. 连续性（当前连中/连亏次数）
        4. 全局命中率基准
        
        Returns:
            float: 预测的命中概率 (0-1)
        """
        if len(self.hit_history) == 0:
            return 0.33  # 初始默认概率
        
        history_list = list(self.hit_history)
        
        # 特征1: 近期命中率（最近lookback期）
        recent_rate = sum(history_list) / len(history_list)
        
        # 特征2: 趋势（近期 vs 全局）
        if len(history_list) >= 10:
            recent_10 = sum(history_list[-10:]) / 10
            early_10 = sum(history_list[:10]) / 10
            trend_factor = (recent_10 - early_10) / 2  # 归一化趋势
        else:
            trend_factor = 0
        
        # 特征3: 连续性因素
        consecutive = 0
        if len(history_list) > 0:
            last_result = history_list[-1]
            for i in range(len(history_list) - 1, -1, -1):
                if history_list[i] == last_result:
                    consecutive += 1
                else:
                    break
            
            # 连续命中越多，下期概率可能下降（均值回归）
            # 连续未中越多，下期概率可能上升
            if last_result == 1:  # 连续命中
                streak_factor = -0.02 * min(consecutive, 5)
            else:  # 连续未中
                streak_factor = 0.03 * min(consecutive, 5)
        else:
            streak_factor = 0
        
        # 特征4: 全局基准率
        global_rate = 0.33  # 假设全局命中率基准
        
        # 综合计算预测概率
        predicted_prob = (
            self.weights['recent_rate'] * recent_rate +
            self.weights['trend'] * (0.33 + trend_factor) +
            self.weights['streak_factor'] * (0.33 + streak_factor) +
            self.weights['global_rate'] * global_rate
        )
        
        # 限制在合理范围内
        predicted_prob = max(0.15, min(0.65, predicted_prob))
        
        return predicted_prob
    
    def calculate_multiplier(self, predicted_prob):
        """
        根据预测概率计算倍投倍数
        
        策略：
        - 高概率（>high_thresh）: 基础倍数 × high_mult
        - 中概率（low_thresh ~ high_thresh）: 基础倍数
        - 低概率（<low_thresh）: 基础倍数 × low_mult
        
        Args:
            predicted_prob: 预测概率
            
        Returns:
            float: 投注倍数
        """
        # 获取Fibonacci基础倍数
        if self.fib_index >= len(self.fib_sequence):
            base_mult = min(self.fib_sequence[-1], self.cfg['max_multiplier'])
        else:
            base_mult = min(self.fib_sequence[self.fib_index], self.cfg['max_multiplier'])
        
        # 根据概率调整倍数
        high_thresh = self.cfg.get('prob_high_thresh', 0.40)
        low_thresh = self.cfg.get('prob_low_thresh', 0.25)
        high_mult = self.cfg.get('high_mult', 1.5)
        low_mult = self.cfg.get('low_mult', 0.5)
        
        if predicted_prob >= high_thresh:
            # 高概率：加大投注
            multiplier = min(base_mult * high_mult, self.cfg['max_multiplier'])
        elif predicted_prob <= low_thresh:
            # 低概率：降低投注
            multiplier = max(base_mult * low_mult, 1)
        else:
            # 中等概率：标准投注
            multiplier = base_mult
        
        return multiplier
    
    def process_period(self, hit):
        """
        处理一期投注
        
        Args:
            hit: 是否命中
            
        Returns:
            dict: 包含multiplier, bet, profit, predicted_prob, recent_rate等信息
        """
        # 预测下期命中概率
        predicted_prob = self.predict_next_probability()
        
        # 根据概率计算倍数
        multiplier = self.calculate_multiplier(predicted_prob)
        
        # 计算投注和收益
        bet = self.cfg['base_bet'] * multiplier
        self.total_bet += bet
        
        if hit:
            win = self.cfg['win_reward'] * multiplier
            self.total_win += win
            profit = win - bet
            self.balance += profit
            self.fib_index = 0  # 命中后重置Fibonacci索引
        else:
            profit = -bet
            self.balance += profit
            self.fib_index += 1  # 未中则升级
            
            # 更新最大回撤
            if self.balance < self.min_balance:
                self.min_balance = self.balance
                self.max_drawdown = abs(self.min_balance)
        
        # 添加到历史记录
        self.hit_history.append(1 if hit else 0)
        
        # 记录预测准确性
        self.prob_history.append({
            'predicted_prob': predicted_prob,
            'actual': 1 if hit else 0,
            'multiplier': multiplier
        })
        
        # 计算近期命中率
        recent_rate = sum(self.hit_history) / len(self.hit_history) if self.hit_history else 0.33
        
        return {
            'multiplier': multiplier,
            'bet': bet,
            'profit': profit,
            'predicted_prob': predicted_prob,
            'recent_rate': recent_rate,
            'fib_index': self.fib_index
        }
    
    def get_prediction_accuracy(self):
        """
        计算预测准确性指标
        
        Returns:
            dict: 包含MAE、RMSE、校准度等指标
        """
        if len(self.prob_history) == 0:
            return None
        
        predicted = np.array([p['predicted_prob'] for p in self.prob_history])
        actual = np.array([p['actual'] for p in self.prob_history])
        
        # 平均绝对误差
        mae = np.mean(np.abs(predicted - actual))
        
        # 均方根误差
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        
        # 校准度（将概率分bin，看实际命中率是否接近预测）
        bins = np.linspace(0, 1, 6)  # 5个区间
        calibration = []
        for i in range(len(bins) - 1):
            mask = (predicted >= bins[i]) & (predicted < bins[i+1])
            if np.sum(mask) > 0:
                avg_pred = np.mean(predicted[mask])
                avg_actual = np.mean(actual[mask])
                calibration.append({
                    'range': f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    'count': np.sum(mask),
                    'avg_predicted': avg_pred,
                    'avg_actual': avg_actual,
                    'bias': avg_actual - avg_pred
                })
        
        return {
            'mae': mae,
            'rmse': rmse,
            'calibration': calibration,
            'total_predictions': len(self.prob_history)
        }


def validate_probability_strategy(predictor, numbers, animals, elements, test_periods=100):
    """
    验证基于概率的倍投策略
    
    Args:
        predictor: TOP15预测器
        numbers: 历史开奖号码
        animals: 历史生肖
        elements: 历史五行
        test_periods: 测试期数
        
    Returns:
        dict: 验证结果
    """
    # 配置
    config = {
        'base_bet': 15,
        'win_reward': 47,
        'max_multiplier': 10,
        'lookback': 20,
        'prob_high_thresh': 0.40,
        'prob_low_thresh': 0.25,
        'high_mult': 1.5,
        'low_mult': 0.5
    }
    
    strategy = ProbabilityBettingStrategy(config)
    history = []
    
    start_idx = len(numbers) - test_periods
    
    for i in range(start_idx, len(numbers)):
        # 预测
        train_data = numbers[:i]
        predictions = predictor.predict(train_data)
        actual = numbers[i]
        
        # 判断命中
        hit = actual in predictions
        
        # 处理这一期
        result = strategy.process_period(hit)
        
        history.append({
            'period': i - start_idx + 1,
            'actual': actual,
            'predictions': predictions,
            'hit': hit,
            **result
        })
    
    # 统计
    wins = sum(1 for h in history if h['hit'])
    losses = len(history) - wins
    hit_rate = wins / len(history) if history else 0
    roi = (strategy.balance / strategy.total_bet * 100) if strategy.total_bet > 0 else 0
    
    # 预测准确性
    accuracy = strategy.get_prediction_accuracy()
    
    return {
        'history': history,
        'total_periods': len(history),
        'wins': wins,
        'losses': losses,
        'hit_rate': hit_rate,
        'total_bet': strategy.total_bet,
        'total_win': strategy.total_win,
        'total_profit': strategy.balance,
        'roi': roi,
        'max_drawdown': strategy.max_drawdown,
        'prediction_accuracy': accuracy
    }
