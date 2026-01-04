"""
GUI集成示例 - 生肖ML预测模型
展示如何将ML模型集成到现有GUI中
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from zodiac_ml_predictor import ZodiacMLPredictor


class ZodiacMLDemo:
    """生肖ML预测演示GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("生肖ML预测演示")
        self.root.geometry("900x700")
        
        self.predictor = None
        self.ml_weight = tk.DoubleVar(value=0.4)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建界面组件"""
        
        # 配置区
        config_frame = ttk.LabelFrame(self.root, text="⚙️ 模型配置", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # ML权重滑块
        ttk.Label(config_frame, text="机器学习权重:").grid(row=0, column=0, sticky='w')
        
        weight_frame = ttk.Frame(config_frame)
        weight_frame.grid(row=0, column=1, sticky='ew', padx=10)
        
        self.weight_scale = ttk.Scale(
            weight_frame, 
            from_=0, 
            to=1, 
            orient='horizontal',
            variable=self.ml_weight,
            command=self._update_weight_label
        )
        self.weight_scale.pack(side='left', fill='x', expand=True)
        
        self.weight_label = ttk.Label(weight_frame, text="40%")
        self.weight_label.pack(side='left', padx=5)
        
        # 预设按钮
        preset_frame = ttk.Frame(config_frame)
        preset_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(preset_frame, text="纯统计", 
                  command=lambda: self._set_weight(0.0)).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="平衡模式", 
                  command=lambda: self._set_weight(0.4)).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="ML优先", 
                  command=lambda: self._set_weight(0.6)).pack(side='left', padx=2)
        ttk.Button(preset_frame, text="纯ML", 
                  command=lambda: self._set_weight(1.0)).pack(side='left', padx=2)
        
        # 预测按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            btn_frame, 
            text="🔮 开始预测", 
            command=self._predict,
            style='Accent.TButton'
        ).pack(fill='x')
        
        # 结果显示区
        result_frame = ttk.LabelFrame(self.root, text="📊 预测结果", padding=10)
        result_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10),
            height=30
        )
        self.result_text.pack(fill='both', expand=True)
        
        # 状态栏
        self.status_label = ttk.Label(
            self.root, 
            text="就绪", 
            relief=tk.SUNKEN, 
            anchor='w'
        )
        self.status_label.pack(fill='x', side='bottom')
    
    def _update_weight_label(self, value):
        """更新权重标签"""
        weight = float(value)
        self.weight_label.config(text=f"{weight*100:.0f}%")
    
    def _set_weight(self, value):
        """设置权重"""
        self.ml_weight.set(value)
        self._update_weight_label(value)
    
    def _predict(self):
        """执行预测"""
        self.result_text.delete('1.0', tk.END)
        self.status_label.config(text="正在预测...")
        self.root.update()
        
        try:
            # 创建预测器
            weight = self.ml_weight.get()
            self.predictor = ZodiacMLPredictor(ml_weight=weight)
            
            self._log("="*80)
            self._log("🤖 生肖ML预测")
            self._log("="*80)
            self._log(f"配置: ML权重={weight*100:.0f}%, 统计权重={100-weight*100:.0f}%\n")
            
            # 执行预测
            result = self.predictor.predict()
            
            # 显示结果
            self._display_result(result)
            
            self.status_label.config(text="预测完成")
            
        except Exception as e:
            self._log(f"\n❌ 错误: {e}")
            self.status_label.config(text="预测失败")
    
    def _display_result(self, result):
        """显示预测结果"""
        
        # 模型信息
        self._log(f"模型: {result['model']}")
        self._log(f"ML状态: {'✓ 已启用' if result['ml_enabled'] else '✗ 未启用'}")
        
        if result['ml_enabled']:
            self._log(f"训练模型数: {len(self.predictor.models)}")
        
        # 最新一期
        self._log(f"\n📅 最新一期（第{result['total_periods']}期）")
        self._log(f"   日期: {result['last_date']}")
        self._log(f"   开出: {result['last_number']} - {result['last_zodiac']}")
        
        # 下一期预测
        self._log(f"\n🔮 下一期预测（第{result['total_periods']+1}期）")
        self._log("="*80)
        
        # TOP6生肖
        self._log("\n⭐ 生肖预测 TOP 6:\n")
        for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
            nums = self.predictor.zodiac_numbers[zodiac]
            
            if i <= 2:
                emoji = "⭐⭐"
                level = "强推"
            elif i <= 4:
                emoji = "⭐"
                level = "推荐"
            else:
                emoji = "✓"
                level = "备选"
            
            # 详细信息
            stat_score = result['stat_scores'][zodiac]
            info = f"{emoji} {i}. {zodiac} [{level}]  综合评分: {score:6.2f}"
            
            if result['ml_probs']:
                ml_prob = result['ml_probs'][zodiac]
                info += f"  (统计:{stat_score:5.1f}, ML:{ml_prob*100:4.1f}%)"
            
            self._log(info)
            self._log(f"      → 号码: {nums}\n")
        
        # 推荐号码
        self._log("\n📋 推荐号码 TOP 18:\n")
        top18 = result['top18_numbers']
        self._log(f"   强推 (1-6):   {top18[0:6]}")
        self._log(f"   推荐 (7-12):  {top18[6:12]}")
        self._log(f"   备选 (13-18): {top18[12:18]}")
        
        # 使用建议
        self._log("\n" + "="*80)
        self._log("💡 使用建议")
        self._log("="*80)
        self._log("   【保守型】选择 TOP2生肖 的号码")
        self._log("   【平衡型】选择 TOP3生肖 的号码 ⭐ 推荐")
        self._log("   【进取型】选择 TOP6生肖 + TOP12号码")
        
        self._log("\n" + "="*80)
    
    def _log(self, message):
        """输出日志"""
        self.result_text.insert(tk.END, message + '\n')
        self.result_text.see(tk.END)
        self.root.update()


def main():
    """主函数"""
    root = tk.Tk()
    app = ZodiacMLDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
