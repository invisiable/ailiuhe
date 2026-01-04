"""快速测试GUI中的生肖预测（最近20期验证）"""
import tkinter as tk
from lucky_number_gui import LuckyNumberGUI

if __name__ == '__main__':
    print("启动幸运号码预测GUI...")
    print("请点击 '🐉 生肖预测' 按钮查看包含最近20期验证数据的预测结果")
    print("-" * 70)
    
    root = tk.Tk()
    app = LuckyNumberGUI(root)
    root.mainloop()
