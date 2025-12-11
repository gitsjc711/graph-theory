# main.py - 程序主入口
import tkinter as tk
from tkinter import messagebox
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入UI模块
try:
    from ui import GraphAlgorithmVisualizer

    print("成功导入UI模块")
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保ui.py文件在当前目录下")
    sys.exit(1)


def main():
    """程序主函数"""
    try:
        # 创建主窗口
        root = tk.Tk()

        # 设置窗口图标（可选）
        try:
            # 如果有图标文件
            if os.path.exists("icon.ico"):
                root.iconbitmap("icon.ico")
        except:
            pass

        # 创建应用程序实例
        app = GraphAlgorithmVisualizer(root)

        # 设置窗口关闭事件
        def on_closing():
            if messagebox.askokcancel("退出", "确定要退出程序吗？"):
                root.destroy()
                sys.exit(0)

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # 运行主循环
        root.mainloop()

    except Exception as e:
        messagebox.showerror("启动错误", f"程序启动失败:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print("程序运行")
    print("=" * 50)
    main()