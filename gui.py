"""
图形用户界面模块
提供数据输入和训练结果输出的窗口界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


class TrainingGUI:
    """模型训练图形界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("机器学习模型训练系统")
        self.root.geometry("1000x700")
        
        self.trainer = ModelTrainer()
        self.data_loaded = False
        self.model_trained = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 1. 数据输入区域
        self.setup_data_input_section(main_frame)
        
        # 2. 训练配置区域
        self.setup_training_config_section(main_frame)
        
        # 3. 结果输出区域
        self.setup_output_section(main_frame)
        
        # 4. 控制按钮区域
        self.setup_control_buttons(main_frame)
        
    def setup_data_input_section(self, parent):
        """设置数据输入区域"""
        
        input_frame = ttk.LabelFrame(parent, text="数据输入", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(1, weight=1)
        
        # 文件选择
        ttk.Label(input_frame, text="数据文件:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.file_path_var).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=5
        )
        ttk.Button(input_frame, text="浏览...", command=self.browse_file).grid(
            row=0, column=2, padx=5
        )
        
        # 目标列选择
        ttk.Label(input_frame, text="目标列:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_column_var = tk.StringVar()
        self.target_column_combo = ttk.Combobox(
            input_frame, textvariable=self.target_column_var, state='readonly'
        )
        self.target_column_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 加载数据按钮
        ttk.Button(input_frame, text="加载数据", command=self.load_data).grid(
            row=1, column=2, padx=5, pady=5
        )
        
        # 数据信息显示
        self.data_info_var = tk.StringVar(value="未加载数据")
        ttk.Label(input_frame, textvariable=self.data_info_var, foreground="blue").grid(
            row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5
        )
        
    def setup_training_config_section(self, parent):
        """设置训练配置区域"""
        
        config_frame = ttk.LabelFrame(parent, text="训练配置", padding="10")
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 模型类型
        ttk.Label(config_frame, text="模型类型:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_type_var = tk.StringVar(value="auto")
        model_types = [
            ("自动选择", "auto"),
            ("线性回归", "linear"),
            ("逻辑回归", "logistic"),
            ("随机森林分类", "random_forest_classifier"),
            ("随机森林回归", "random_forest_regressor")
        ]
        
        model_frame = ttk.Frame(config_frame)
        model_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W, padx=5)
        
        for i, (text, value) in enumerate(model_types):
            ttk.Radiobutton(
                model_frame, text=text, variable=self.model_type_var, value=value
            ).grid(row=0, column=i, padx=5)
        
        # 测试集比例
        ttk.Label(config_frame, text="测试集比例:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(config_frame, textvariable=self.test_size_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5
        )
        
        # 随机种子
        ttk.Label(config_frame, text="随机种子:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.random_state_var = tk.StringVar(value="42")
        ttk.Entry(config_frame, textvariable=self.random_state_var, width=10).grid(
            row=1, column=3, sticky=tk.W, padx=5, pady=5
        )
        
    def setup_output_section(self, parent):
        """设置结果输出区域"""
        
        output_frame = ttk.LabelFrame(parent, text="训练结果输出", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # 创建Notebook用于多标签页
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 文本输出标签页
        text_frame = ttk.Frame(self.notebook)
        self.notebook.add(text_frame, text="训练日志")
        
        self.output_text = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, width=80, height=15
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # 图表标签页
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="可视化")
        self.chart_frame = chart_frame
        
    def setup_control_buttons(self, parent):
        """设置控制按钮区域"""
        
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, pady=10)
        
        # 开始训练按钮
        self.train_button = ttk.Button(
            button_frame, text="开始训练", command=self.start_training, width=15
        )
        self.train_button.grid(row=0, column=0, padx=5)
        
        # 保存模型按钮
        self.save_button = ttk.Button(
            button_frame, text="保存模型", command=self.save_model, width=15, state='disabled'
        )
        self.save_button.grid(row=0, column=1, padx=5)
        
        # 加载模型按钮
        ttk.Button(
            button_frame, text="加载模型", command=self.load_model, width=15
        ).grid(row=0, column=2, padx=5)
        
        # 清空输出按钮
        ttk.Button(
            button_frame, text="清空输出", command=self.clear_output, width=15
        ).grid(row=0, column=3, padx=5)
        
    def browse_file(self):
        """浏览并选择数据文件"""
        filename = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            # 读取CSV文件的列名
            try:
                df = pd.read_csv(filename)
                columns = list(df.columns)
                self.target_column_combo['values'] = columns
                if columns:
                    self.target_column_combo.current(len(columns) - 1)  # 默认选择最后一列
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败: {str(e)}")
    
    def load_data(self):
        """加载训练数据"""
        file_path = self.file_path_var.get()
        target_column = self.target_column_var.get()
        
        if not file_path:
            messagebox.showwarning("警告", "请先选择数据文件")
            return
        
        if not target_column:
            messagebox.showwarning("警告", "请选择目标列")
            return
        
        try:
            self.trainer.load_data(file_path, target_column)
            self.data_loaded = True
            
            info = (f"数据加载成功! "
                   f"样本数: {len(self.trainer.X)}, "
                   f"特征数: {len(self.trainer.feature_names)}, "
                   f"任务类型: {'分类' if self.trainer.is_classification else '回归'}")
            self.data_info_var.set(info)
            
            self.log_output(f"\n{'='*60}\n")
            self.log_output(f"数据加载成功\n")
            self.log_output(f"文件路径: {file_path}\n")
            self.log_output(f"目标列: {target_column}\n")
            self.log_output(f"样本数量: {len(self.trainer.X)}\n")
            self.log_output(f"特征列表: {', '.join(self.trainer.feature_names)}\n")
            self.log_output(f"任务类型: {'分类任务' if self.trainer.is_classification else '回归任务'}\n")
            self.log_output(f"{'='*60}\n")
            
            messagebox.showinfo("成功", info)
            
        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.log_output(f"\n错误: {str(e)}\n")
    
    def start_training(self):
        """开始训练模型"""
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据")
            return
        
        # 禁用训练按钮防止重复点击
        self.train_button.config(state='disabled')
        
        # 在新线程中训练模型
        thread = threading.Thread(target=self.train_model_thread)
        thread.daemon = True
        thread.start()
    
    def train_model_thread(self):
        """在后台线程中训练模型"""
        try:
            model_type = self.model_type_var.get()
            test_size = float(self.test_size_var.get())
            random_state = int(self.random_state_var.get())
            
            self.log_output(f"\n{'='*60}\n")
            self.log_output(f"开始训练模型...\n")
            self.log_output(f"模型类型: {model_type}\n")
            self.log_output(f"测试集比例: {test_size}\n")
            self.log_output(f"随机种子: {random_state}\n")
            self.log_output(f"{'='*60}\n")
            
            # 训练模型
            results = self.trainer.train_model(model_type, test_size, random_state)
            
            # 显示结果
            self.log_output(f"\n训练完成!\n")
            self.log_output(f"{'='*60}\n")
            self.log_output(f"模型类型: {results['model_type']}\n")
            self.log_output(f"训练样本数: {results['train_samples']}\n")
            self.log_output(f"测试样本数: {results['test_samples']}\n")
            self.log_output(f"特征数量: {results['feature_count']}\n")
            self.log_output(f"\n评估指标:\n")
            self.log_output(f"{results['metrics']}\n")
            
            if 'classification_report' in results:
                self.log_output(f"\n分类报告:\n")
                self.log_output(f"{results['classification_report']}\n")
            
            self.log_output(f"{'='*60}\n")
            
            self.model_trained = True
            self.root.after(0, lambda: self.save_button.config(state='normal'))
            self.root.after(0, lambda: messagebox.showinfo("成功", "模型训练完成!"))
            
            # 绘制特征重要性图（如果是随机森林模型）
            if hasattr(self.trainer.model, 'feature_importances_'):
                self.root.after(0, self.plot_feature_importance)
            
        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            self.log_output(f"\n{error_msg}\n")
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
        
        finally:
            # 重新启用训练按钮
            self.root.after(0, lambda: self.train_button.config(state='normal'))
    
    def plot_feature_importance(self):
        """绘制特征重要性图"""
        try:
            # 清空之前的图表
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(8, 5))
            
            importances = self.trainer.model.feature_importances_
            features = self.trainer.feature_names
            
            # 排序
            indices = importances.argsort()[::-1][:10]  # 只显示前10个最重要的特征
            
            ax.bar(range(len(indices)), importances[indices])
            ax.set_xlabel('特征')
            ax.set_ylabel('重要性')
            ax.set_title('特征重要性排名 (Top 10)')
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
            
            plt.tight_layout()
            
            # 嵌入到Tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.log_output(f"\n绘制图表失败: {str(e)}\n")
    
    def save_model(self):
        """保存训练好的模型"""
        if not self.model_trained:
            messagebox.showwarning("警告", "请先训练模型")
            return
        
        try:
            filepath = self.trainer.save_model()
            self.log_output(f"\n模型已保存至: {filepath}\n")
            messagebox.showinfo("成功", f"模型已保存至:\n{filepath}")
            
        except Exception as e:
            error_msg = f"保存模型失败: {str(e)}"
            self.log_output(f"\n{error_msg}\n")
            messagebox.showerror("错误", error_msg)
    
    def load_model(self):
        """加载已保存的模型"""
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.trainer.load_model(filename)
                self.model_trained = True
                self.save_button.config(state='normal')
                
                info = (f"模型: {self.trainer.model_type}, "
                       f"特征数: {len(self.trainer.feature_names)}, "
                       f"任务: {'分类' if self.trainer.is_classification else '回归'}")
                
                self.log_output(f"\n{'='*60}\n")
                self.log_output(f"模型加载成功\n")
                self.log_output(f"文件路径: {filename}\n")
                self.log_output(f"模型类型: {self.trainer.model_type}\n")
                self.log_output(f"特征列表: {', '.join(self.trainer.feature_names)}\n")
                self.log_output(f"任务类型: {'分类任务' if self.trainer.is_classification else '回归任务'}\n")
                self.log_output(f"{'='*60}\n")
                
                messagebox.showinfo("成功", f"模型加载成功!\n{info}")
                
            except Exception as e:
                error_msg = f"加载模型失败: {str(e)}"
                self.log_output(f"\n{error_msg}\n")
                messagebox.showerror("错误", error_msg)
    
    def clear_output(self):
        """清空输出区域"""
        self.output_text.delete(1.0, tk.END)
    
    def log_output(self, message):
        """输出日志信息"""
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)  # 自动滚动到底部
        self.output_text.update()


def main():
    """主函数"""
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
