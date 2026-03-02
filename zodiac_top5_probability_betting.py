#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生肖TOP5概率预测投注策略

结合生肖TOP5预测和概率预测动态倍投
目标：提升收益，降低回撤
"""

import numpy as np
import pandas as pd
from collections import deque


class ZodiacTop5ProbabilityBetting:
    """生肖TOP5概率预测动态倍投策略"""
    
    def __init__(self, config):
        """
        初始化策略
        
        Args:
            config: 配置字典
                - base_bet: 基础投注金额（20元，5个生肖×4元）
                - win_reward: 中奖奖励（47元）
                - max_multiplier: 最大倍数限制
                - lookback: 回看期数
                - prob_high_thresh: 高概率阈值
                - prob_low_thresh: 低概率阈值
                - high_mult: 高概率倍数系数
                - low_mult: 低概率倍数系数
        """
        self.cfg = config
        self.fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.fib_index = 0
        
        # 历史记录
        self.hit_history = deque(maxlen=config.get('lookback', 20))
        self.prob_history = []
        
        # 统计信息
        self.total_bet = 0
        self.total_win = 0
        self.balance = 0
        self.min_balance = 0
        self.max_drawdown = 0
        
        # 特征权重（针对生肖TOP5优化）
        self.weights = {
            'recent_rate': 0.4,      # 近期命中率
            'trend': 0.3,            # 趋势
            'streak_factor': 0.2,    # 连续性因素
            'global_rate': 0.1       # 全局基准（生肖TOP5约42%）
        }
    
    def predict_next_probability(self):
        """
        预测下一期命中概率（针对生肖TOP5优化）
        
        Returns:
            float: 预测的命中概率 (0-1)
        """
        if len(self.hit_history) == 0:
            return 0.42  # 生肖TOP5理论命中率
        
        history_list = list(self.hit_history)
        
        # 特征1: 近期命中率
        recent_rate = sum(history_list) / len(history_list)
        
        # 特征2: 趋势
        if len(history_list) >= 10:
            recent_10 = sum(history_list[-10:]) / 10
            early_10 = sum(history_list[:10]) / 10
            trend_factor = (recent_10 - early_10) / 2
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
            
            # 生肖预测的连续性调整（略微不同于数字预测）
            if last_result == 1:  # 连续命中
                streak_factor = -0.015 * min(consecutive, 5)  # 略微下调
            else:  # 连续未中
                streak_factor = 0.025 * min(consecutive, 5)  # 略微上调
        else:
            streak_factor = 0
        
        # 特征4: 全局基准率（生肖TOP5）
        global_rate = 0.42
        
        # 综合计算
        predicted_prob = (
            self.weights['recent_rate'] * recent_rate +
            self.weights['trend'] * (0.42 + trend_factor) +
            self.weights['streak_factor'] * (0.42 + streak_factor) +
            self.weights['global_rate'] * global_rate
        )
        
        # 限制在合理范围（生肖TOP5：25%-65%）
        predicted_prob = max(0.25, min(0.65, predicted_prob))
        
        return predicted_prob
    
    def calculate_multiplier(self, predicted_prob):
        """
        根据预测概率计算倍投倍数
        
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
        
        # 根据概率调整倍数（生肖TOP5阈值）
        high_thresh = self.cfg.get('prob_high_thresh', 0.45)  # 生肖略高于数字
        low_thresh = self.cfg.get('prob_low_thresh', 0.30)   # 生肖略高于数字
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
            dict: 包含multiplier, bet, profit, predicted_prob等信息
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
            self.fib_index = 0  # 命中后重置
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
        recent_rate = sum(self.hit_history) / len(self.hit_history) if self.hit_history else 0.42
        
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
            dict: MAE、RMSE、校准度等指标
        """
        if len(self.prob_history) == 0:
            return None
        
        predicted = np.array([p['predicted_prob'] for p in self.prob_history])
        actual = np.array([p['actual'] for p in self.prob_history])
        
        # 平均绝对误差
        mae = np.mean(np.abs(predicted - actual))
        
        # 均方根误差
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        
        # 校准度
        bins = np.linspace(0, 1, 6)
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


def validate_zodiac_probability_strategy(zodiac_predictor, animals, test_periods=200):
    """
    验证生肖TOP5概率预测策略
    
    Args:
        zodiac_predictor: 生肖预测器
        animals: 历史生肖数据
        test_periods: 测试期数
        
    Returns:
        dict: 验证结果
    """
    # 配置
    config = {
        'base_bet': 20,
        'win_reward': 47,
        'max_multiplier': 10,
        'lookback': 20,
        'prob_high_thresh': 0.45,  # 生肖阈值
        'prob_low_thresh': 0.30,   # 生肖阈值
        'high_mult': 1.5,
        'low_mult': 0.5
    }
    
    strategy = ZodiacTop5ProbabilityBetting(config)
    history = []
    
    start_idx = len(animals) - test_periods
    
    for i in range(start_idx, len(animals)):
        # 预测
        train_data = animals[:i].tolist()
        result = zodiac_predictor.predict_from_history(train_data, top_n=5, debug=False)
        top5 = result['top5']
        
        # 实际结果
        actual = animals[i]
        
        # 判断命中
        hit = actual in top5
        
        # 处理这一期
        period_result = strategy.process_period(hit)
        
        history.append({
            'period': i - start_idx + 1,
            'actual': actual,
            'predictions': top5,
            'hit': hit,
            **period_result
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
        'prediction_accuracy': accuracy,
        'config': config
    }
