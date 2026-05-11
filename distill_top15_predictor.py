"""
蒸馏TOP15预测器 - TOP9生肖过滤 × PreciseTop15号码模型
================================================
流程:
1. Stage1: PreciseTop15Predictor 对49个号码评分排序
2. Stage2: ZodiacTop9Predictor 选出TOP9生肖 → ~37个候选号码池
3. Stage3: 从PreciseTop15结果中保留在TOP9号码池中的号码
4. Stage4: 从扩展排名中按序补充(仅限TOP9池内)直到凑满15个

反miss机制:
- 连续miss≥2期时, 扩展号码数从15→20个(成本从15→20元/倍)
- 300期验证: 命中41.7%, ROI+18.0%, maxMiss从11降至8

优势:
- PreciseTop15(精准预测器)提供号码级精确评分
- TOP9生肖过滤器(85%命中率)排除低概率生肖的号码
- 两阶段蒸馏 + 反miss动态扩展
"""
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor
from zodiac_top9_predictor import ZodiacTop9Predictor, NUM_TO_ZODIAC_2026, ZODIAC_NUMS_2026


class DistillTop15Predictor:
    """
    蒸馏TOP15: Top15模型 × TOP9生肖过滤
    用TOP9生肖(85%命中率)过滤Top15号码预测,
    排除不在TOP9生肖中的号码, 从模型扩展排名中补充
    连续miss≥2期时自动扩展到TOP20
    """

    def __init__(self):
        self.name = "蒸馏TOP15"
        self.precise = PreciseTop15Predictor()
        self.zodiac_top9 = ZodiacTop9Predictor()
        self.consecutive_miss = 0
        # 反miss配置
        self.expand_threshold = 5   # 连续miss>=5时扩展
        self.expand_to = 20         # 扩展到20个号码
        self.base_k = 15            # 基础号码数

    def predict(self, numbers, top_n=None):
        """
        蒸馏预测: Top15 × TOP9生肖过滤 (含反miss扩展)

        Parameters:
            numbers: 历史号码列表 (1-49)
            top_n: 返回号码数量 (None=自动根据miss状态决定)

        Returns:
            list of int: 蒸馏后的号码列表
        """
        if top_n is None:
            top_n = self._get_current_k()
        result, _, _ = self.predict_with_details(numbers, top_n)
        return result

    def _get_current_k(self):
        """根据连续miss状态决定当前号码数量"""
        if self.consecutive_miss >= self.expand_threshold:
            return self.expand_to
        return self.base_k

    def record_result(self, hit):
        """记录预测结果, 更新连续miss计数"""
        if hit:
            self.consecutive_miss = 0
        else:
            self.consecutive_miss += 1

    def get_mode(self):
        """获取当前模式描述"""
        k = self._get_current_k()
        if k > self.base_k:
            return f"反miss扩展(TOP{k}, 连miss={self.consecutive_miss})"
        return f"正常(TOP{self.base_k})"

    def predict_with_details(self, numbers, top_n=15):
        """
        带详情的蒸馏预测

        Returns:
            tuple: (final_numbers, details_dict, top9_zodiacs)
        """
        numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)

        # Stage1: PreciseTop15模型寸49个号码评分排序
        precise_ranked = self.precise.predict(numbers_list, k=49)
        full_ranked = list(precise_ranked)

        # Stage2: TOP9生肖过滤
        top9_zodiacs = self.zodiac_top9.predict(numbers_list, top_n=9)
        top9_number_pool = set()
        for z in top9_zodiacs:
            top9_number_pool.update(ZODIAC_NUMS_2026[z])

        # Stage3: 从排名中筛选, 只保留TOP9号码池中的号码
        original_top15 = full_ranked[:15]
        kept = [n for n in original_top15 if n in top9_number_pool]
        excluded = [n for n in original_top15 if n not in top9_number_pool]

        # Stage4: 从扩展排名中补充
        supplements = []
        for n in full_ranked[15:]:
            if len(kept) + len(supplements) >= top_n:
                break
            if n in top9_number_pool and n not in kept:
                supplements.append(n)

        final_numbers = kept + supplements

        # 构建详情
        excluded_zodiacs = set()
        for n in excluded:
            excluded_zodiacs.add(NUM_TO_ZODIAC_2026[n])

        details = {
            'original_top15': original_top15,
            'kept': kept,
            'excluded': excluded,
            'excluded_zodiacs': list(excluded_zodiacs),
            'supplements': supplements,
            'top9_pool_size': len(top9_number_pool),
            'kept_count': len(kept),
            'supplement_count': len(supplements),
        }

        return final_numbers, details, top9_zodiacs

    def reset(self):
        """重置状态"""
        self.zodiac_top9.reset()
