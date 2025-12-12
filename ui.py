# ui.py - 用户界面模块
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib

matplotlib.use('TkAgg')  # 确保在使用TkAgg后端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import networkx as nx
import random
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入算法函数
try:
    from algorithm import (
        kruskal, prim, break_cycle,
        dijkstra, floyd, floyd_warshall,
        hungarian, kuhn_munkres
    )

    print("成功导入算法模块")
except ImportError as e:
    print(f"导入算法模块失败: {e}")
    print("请确保 algorithm.py 文件在当前目录下")


    # 定义虚拟函数，避免程序崩溃
    def dummy_algorithm(*args, **kwargs):
        return {"error": "算法函数未找到"}


    kruskal = prim = break_cycle = dijkstra = floyd = floyd_warshall = hungarian  = dummy_algorithm


def setup_chinese_font():
    """设置中文字体支持"""
    # 常见系统中文字体列表
    chinese_fonts = [
        'Microsoft YaHei',  # Windows 雅黑
        'SimHei',  # Windows 黑体
        'SimSun',  # Windows 宋体
        'FangSong',  # Windows 仿宋
        'KaiTi',  # Windows 楷体
        'STXihei',  # Mac 黑体
        'STKaiti',  # Mac 楷体
        'STSong',  # Mac 宋体
        'Arial Unicode MS',  # 通用
        'DejaVu Sans',  # 默认字体
    ]

    # 尝试设置中文字体
    for font in chinese_fonts:
        try:
            # 设置matplotlib的默认字体
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 测试字体是否可用
            test_font = FontProperties(fname=None, family=font)

            # 如果设置了中文字体，就使用它
            if font not in ['DejaVu Sans', 'Arial']:  # 排除默认的英文字体
                print(f"已设置中文字体: {font}")
                break

        except Exception as e:
            continue

    # 如果以上字体都不可用，尝试从系统字体目录加载
    if plt.rcParams['font.sans-serif'][0] in ['DejaVu Sans', 'Arial']:
        print("警告: 未找到中文字体，中文可能显示异常")
        print("建议安装中文字体，如: Microsoft YaHei")


# 调用设置字体函数
setup_chinese_font()


class GraphAlgorithmVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("图算法实现 v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # 图的存储结构
        self.node_count = 0
        self.adj_matrix = None
        self.graph_type = tk.StringVar(value="undirected")  # undirected or directed
        self.node_labels = []  # 节点标签A-J
        self.weighted_graph = tk.BooleanVar(value=True)  # 是否带权图

        # 算法选择变量
        self.algorithm_var = tk.StringVar()

        # 可视化相关
        self.fig = None
        self.ax = None
        self.canvas = None
        self.G = None  # NetworkX图对象
        self.pos = None  # 节点位置

        self.setup_ui()
        self.initialize_graph()

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 右侧可视化区域
        viz_frame = ttk.LabelFrame(main_frame, text="图可视化", padding="10")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.setup_control_panel(control_frame)
        self.setup_visualization_area(viz_frame)

    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 算法选择
        ttk.Label(parent, text="选择算法:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        # 算法列表
        algorithms = [
            "Kruskal算法",
            "Prim算法",
            "破圈法",
            "Dijkstra算法",
            "Floyd算法",
            "Floyd-Warshall算法",
            "匈牙利算法"
        ]

        algorithm_combo = ttk.Combobox(parent, textvariable=self.algorithm_var,
                                       values=algorithms, state="readonly")
        algorithm_combo.pack(fill=tk.X, pady=(0, 10))
        algorithm_combo.current(0)

        # 图类型选择
        ttk.Label(parent, text="图类型:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))

        type_frame = ttk.Frame(parent)
        type_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(type_frame, text="无向图", variable=self.graph_type,
                        value="undirected").pack(side=tk.LEFT)
        ttk.Radiobutton(type_frame, text="有向图", variable=self.graph_type,
                        value="directed").pack(side=tk.LEFT)

        # 绑定图类型变化事件
        def on_graph_type_change(*args):
            self.update_matrix_display()
            self.draw_graph()

        self.graph_type.trace('w', on_graph_type_change)

        # 节点数量设置
        ttk.Label(parent, text="节点数量 (1-10):", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))

        node_frame = ttk.Frame(parent)
        node_frame.pack(fill=tk.X, pady=(0, 10))

        self.node_var = tk.IntVar(value=5)  # 默认改为5个节点
        node_scale = ttk.Scale(node_frame, from_=1, to=10, variable=self.node_var,  # 最大改为10
                               orient=tk.HORIZONTAL, command=self.on_node_count_change)
        node_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.node_label = ttk.Label(node_frame, text="5", width=3)  # 默认显示5
        self.node_label.pack(side=tk.RIGHT, padx=(5, 0))

        # 起始节点选择（使用字母下拉框）
        ttk.Label(parent, text="起始节点:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))

        # 创建下拉选择框，选项为节点标签
        self.start_node_var = tk.StringVar(value="A")
        self.start_node_combo = ttk.Combobox(
            parent,
            textvariable=self.start_node_var,
            values=self.node_labels[:self.node_count],  # 只显示当前节点数量的字母
            state="readonly",  # 设置为只读，防止用户输入无效内容
            width=8
        )
        self.start_node_combo.pack(anchor=tk.W, pady=(0, 10))

        # 图权重选择
        ttk.Label(parent, text="图权重:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))

        weight_frame = ttk.Frame(parent)
        weight_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(weight_frame, text="带权图", variable=self.weighted_graph,
                        value=True).pack(side=tk.LEFT)
        ttk.Radiobutton(weight_frame, text="无权图(权重=1)", variable=self.weighted_graph,
                        value=False).pack(side=tk.LEFT)

        # 邻接矩阵输入
        ttk.Label(parent, text="邻接矩阵输入:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))

        matrix_frame = ttk.Frame(parent)
        matrix_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.matrix_text = scrolledtext.ScrolledText(matrix_frame, height=10, width=30)
        self.matrix_text.pack(fill=tk.BOTH, expand=True)

        # 按钮区域
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="应用矩阵", command=self.apply_matrix).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                                  padx=2)
        ttk.Button(button_frame, text="随机生成", command=self.generate_random_graph).pack(side=tk.LEFT, fill=tk.X,
                                                                                           expand=True, padx=2)
        ttk.Button(button_frame, text="运行算法", command=self.run_algorithm).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                                   padx=2)
        ttk.Button(button_frame, text="清空重置", command=self.reset_graph).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                                 padx=2)

        # 结果显示
        ttk.Label(parent, text="算法结果:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

        self.result_text = scrolledtext.ScrolledText(parent, height=8, width=30)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def setup_visualization_area(self, parent):
        """设置可视化区域"""
        # 创建Matplotlib图形
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # 创建Tkinter画布
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 工具栏框架
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(toolbar_frame, text="重新绘制", command=self.draw_graph).pack(side=tk.LEFT)
        ttk.Button(toolbar_frame, text="保存图片", command=self.save_image).pack(side=tk.LEFT)

    def initialize_graph(self):
        """初始化图数据"""
        self.node_count = 5  # 默认改为5个节点
        self.node_labels = [chr(65 + i) for i in range(10)]  # 只生成A-J，对应10个节点

        # 初始化邻接矩阵
        self.adj_matrix = np.zeros((self.node_count, self.node_count), dtype=float)
        np.fill_diagonal(self.adj_matrix, 0)

        # 设置起始节点下拉框的初始值
        if hasattr(self, 'start_node_combo'):
            self.start_node_combo.config(values=self.node_labels[:self.node_count])
            self.start_node_var.set("A")

        self.update_matrix_display()
        self.draw_graph()

    def on_node_count_change(self, event=None):
        """节点数量变化处理"""
        self.node_count = int(self.node_var.get())
        self.node_label.config(text=str(self.node_count))

        # 调整邻接矩阵大小
        new_size = self.node_count
        if self.adj_matrix is None or self.adj_matrix.shape[0] != new_size:
            new_matrix = np.zeros((new_size, new_size), dtype=float)
            if self.adj_matrix is not None:
                min_size = min(self.adj_matrix.shape[0], new_size)
                new_matrix[:min_size, :min_size] = self.adj_matrix[:min_size, :min_size]
            self.adj_matrix = new_matrix

        # 更新起始节点下拉框的选项
        if hasattr(self, 'start_node_combo'):
            self.start_node_combo.config(values=self.node_labels[:self.node_count])
            # 如果当前选择的起始节点字母超出了新范围，则重置为'A'
            current_selection = self.start_node_var.get()
            if current_selection not in self.node_labels[:self.node_count]:
                self.start_node_var.set("A")

        self.update_matrix_display()
        self.draw_graph()

    def update_matrix_display(self):
        """更新矩阵显示"""
        self.matrix_text.delete(1.0, tk.END)

        if self.adj_matrix is not None:
            # 表头
            header = "    " + " ".join(f"{self.node_labels[i]:>4}" for i in range(self.node_count)) + "\n"
            self.matrix_text.insert(tk.END, header)
            self.matrix_text.insert(tk.END, "    " + "-" * (self.node_count * 5) + "\n")

            # 矩阵内容
            for i in range(self.node_count):
                row_str = f"{self.node_labels[i]} | "
                for j in range(self.node_count):
                    if self.adj_matrix[i][j] == float('inf'):
                        row_str += "  ∞ "
                    else:
                        row_str += f"{self.adj_matrix[i][j]:4.1f}"
                self.matrix_text.insert(tk.END, row_str + "\n")

    def apply_matrix(self):
        """应用用户输入的矩阵"""
        try:
            text = self.matrix_text.get(1.0, tk.END).strip()
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            # 解析矩阵
            new_matrix = []
            for line in lines:
                # 跳过表头和分隔线
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) > 1:
                        nums = parts[1].strip().split()
                        row = []
                        for num in nums:
                            if num == '∞':
                                row.append(float('inf'))
                            else:
                                row.append(float(num))
                        new_matrix.append(row)

            if new_matrix:
                n = len(new_matrix)
                if n != len(new_matrix[0]):
                    messagebox.showerror("错误", "矩阵必须是方阵！")
                    return

                if n > 10:  # 修改为10个节点限制
                    messagebox.showerror("错误", "节点数量不能超过10个！")
                    return

                self.node_count = n
                self.node_var.set(n)
                self.node_label.config(text=str(n))
                self.adj_matrix = np.array(new_matrix)

                # 更新起始节点下拉框
                if hasattr(self, 'start_node_combo'):
                    self.start_node_combo.config(values=self.node_labels[:self.node_count])
                    self.start_node_var.set("A")

                self.draw_graph()
                messagebox.showinfo("成功", "矩阵应用成功！")
        except Exception as e:
            messagebox.showerror("错误", f"解析矩阵失败: {str(e)}")

    def generate_random_adj_matrix(self, n, density):
        """1. 直接随机生成无向图的邻接矩阵"""
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):  # 无向图，只处理上三角
                if random.random() < density:
                    if self.weighted_graph.get():
                        weight = random.randint(1, 10)  # 带权：1-10
                    else:
                        weight = 1  # 不带权：1
                    matrix[i][j] = weight
                    matrix[j][i] = weight  # 无向图对称

        return matrix

    def is_graph_connected(self, matrix):
        """2. 判断邻接矩阵是否连通"""
        n = len(matrix)

        if n == 0:
            return True

        # 如果没有边，返回False
        has_edge = False
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] != 0:
                    has_edge = True
                    break
            if has_edge:
                break

        if not has_edge and n > 1:
            return False

        # 找到第一个有边的节点作为起点
        start_node = 0
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] != 0:
                    start_node = i
                    break
            if start_node != 0:
                break

        # BFS遍历
        visited = [False] * n
        queue = [start_node]
        visited[start_node] = True

        while queue:
            current = queue.pop(0)
            for neighbor in range(n):
                if matrix[current][neighbor] != 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        # 检查所有节点是否都被访问
        for i in range(n):
            if not visited[i]:
                return False
        return True

    def convert_to_directed(self, undirected_matrix):
        """3. 如果有向图，根据无向图的矩阵修改"""
        n = len(undirected_matrix)
        directed_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if undirected_matrix[i][j] != 0:
                    # 随机选择方向：单向或双向
                    rand = random.random()
                    if rand < 0.3:  # 30%概率双向
                        directed_matrix[i][j] = undirected_matrix[i][j]
                        directed_matrix[j][i] = undirected_matrix[i][j]
                    elif rand < 0.65:  # 35%概率 i->j
                        directed_matrix[i][j] = undirected_matrix[i][j]
                    else:  # 35%概率 j->i
                        directed_matrix[j][i] = undirected_matrix[i][j]

        return directed_matrix

    def convert_to_unweighted(self, weighted_matrix):
        """4. 如果不带权图，将所有权重修改为1"""
        n = len(weighted_matrix)
        unweighted_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if weighted_matrix[i][j] != 0:
                    unweighted_matrix[i][j] = 1

        return unweighted_matrix

    def generate_random_graph(self):
        """主函数：按照4个步骤生成随机图"""
        try:
            n = self.node_count
            max_attempts = 1000
            density = 0.5  # 使用固定的较高密度，确保容易生成连通图

            connected_matrix = None

            # 步骤1和2：生成无向图并检查连通性，不连通则重试
            for attempt in range(max_attempts):
                # 步骤1：生成随机无向图邻接矩阵
                temp_matrix = self.generate_random_adj_matrix(n, density)

                # 步骤2：检查连通性
                if self.is_graph_connected(temp_matrix):
                    connected_matrix = temp_matrix
                    print(f"成功生成连通图，尝试次数: {attempt + 1}")
                    break

            if connected_matrix is None:
                # 如果尝试max_attempts次后仍未生成连通图，提高密度再试一次
                messagebox.showinfo("提示", f"尝试{max_attempts}次未生成连通图，提高密度重试")
                density = 0.8
                for attempt in range(max_attempts):
                    temp_matrix = self.generate_random_adj_matrix(n, density)
                    if self.is_graph_connected(temp_matrix):
                        connected_matrix = temp_matrix
                        print(f"提高密度后成功生成连通图，尝试次数: {attempt + 1}")
                        break

            if connected_matrix is None:
                messagebox.showerror("错误", "无法生成连通图，请减少节点数或重试")
                return

            # 步骤3：如果是有向图，根据无向图的矩阵修改
            if self.graph_type.get() == "directed":
                connected_matrix = self.convert_to_directed(connected_matrix)

            # 步骤4：如果是不带权图，将所有权重修改为1
            if not self.weighted_graph.get():
                connected_matrix = self.convert_to_unweighted(connected_matrix)

            self.adj_matrix = connected_matrix
            self.update_matrix_display()

            # 更新起始节点下拉框
            if hasattr(self, 'start_node_combo'):
                self.start_node_combo.config(values=self.node_labels[:self.node_count])
                self.start_node_var.set("A")

            self.draw_graph()
            messagebox.showinfo("成功", f"随机{'带权' if self.weighted_graph.get() else '无权'}图生成成功！")

        except Exception as e:
            messagebox.showerror("错误", f"生成随机图失败: {str(e)}")

    def run_algorithm(self):
        """运行选中的算法"""
        try:
            algorithm_name = self.algorithm_var.get()

            # 将用户选择的字母（如'A'）映射为数字索引（如0）
            selected_letter = self.start_node_var.get()
            if selected_letter in self.node_labels:
                start_node = self.node_labels.index(selected_letter)
                # 确保索引不超过当前图的节点数量
                if start_node >= self.node_count:
                    messagebox.showwarning("警告", f"起始节点 {selected_letter} 超出当前图的节点范围，将使用节点A。")
                    start_node = 0
            else:
                messagebox.showwarning("警告", f"起始节点 {selected_letter} 无效，将使用节点A。")
                start_node = 0

            result = None

            # 调用对应的算法函数
            if algorithm_name == "Kruskal算法":
                result = kruskal(self.adj_matrix)
            elif algorithm_name == "Prim算法":
                result = prim(self.adj_matrix, start_node)
            elif algorithm_name == "破圈法":
                result = break_cycle(self.adj_matrix)
            elif algorithm_name == "Dijkstra算法":
                result = dijkstra(self.adj_matrix, start_node)
            elif algorithm_name == "Floyd算法":
                result = floyd(self.adj_matrix)
            elif algorithm_name == "Floyd-Warshall算法":
                result = floyd_warshall(self.adj_matrix)
            elif algorithm_name == "匈牙利算法":
                result = hungarian(self.adj_matrix)
            # elif algorithm_name == "Kuhn-Munkres算法":
            #     result = kuhn_munkres(self.adj_matrix)
            else:
                messagebox.showerror("错误", "请选择一个算法！")
                return

            # 显示结果
            self.display_result(result)

            # 可视化结果
            self.visualize_algorithm_result(result)

        except Exception as e:
            messagebox.showerror("错误", f"算法执行失败: {str(e)}")

    def display_result(self, result):
        """在结果框中显示算法结果"""
        self.result_text.delete(1.0, tk.END)

        if not result:
            return

        self.result_text.insert(tk.END, f"算法: {result.get('algorithm', '未知')}\n")
        self.result_text.insert(tk.END, "=" * 40 + "\n\n")

        # 根据算法类型显示不同信息
        if 'total_weight' in result:
            self.result_text.insert(tk.END, f"总权重: {result['total_weight']:.2f}\n\n")

        if 'error' in result and result['error']:
            self.result_text.insert(tk.END, "错误:\n")
            self.result_text.insert(tk.END, result['error'])
        elif 'edges' in result and result['edges']:
            self.result_text.insert(tk.END, "最小生成树边:\n")
            for u, v, w in result['edges']:
                self.result_text.insert(tk.END, f"  {self.node_labels[u]} - {self.node_labels[v]}: {w}\n")

        elif 'distances' in result:
            self.result_text.insert(tk.END, "最短路径距离:\n")
            for i, dist in enumerate(result['distances']):
                if dist == float('inf'):
                    self.result_text.insert(tk.END, f"  到节点{self.node_labels[i]}: 不可达\n")
                else:
                    self.result_text.insert(tk.END, f"  到节点{self.node_labels[i]}: {dist:.2f}\n")
        elif 'distance_matrix' in result:
            self.result_text.insert(tk.END, "所有节点对最短距离矩阵:\n")
            dist_matrix = result['distance_matrix']
            n = len(dist_matrix)
            max_display = min(10, n)
            # 紧凑的表头
            self.result_text.insert(tk.END, "    ")
            for j in range(max_display):
                self.result_text.insert(tk.END, f"{self.node_labels[j]:>5} ")
            self.result_text.insert(tk.END, "\n")
            # 紧凑的数据
            for i in range(max_display):
                self.result_text.insert(tk.END, f"{self.node_labels[i]:>2}: ")
                for j in range(max_display):
                    if dist_matrix[i][j] == float('inf') or i == j:
                        self.result_text.insert(tk.END, "   ∞  ")
                    else:
                        self.result_text.insert(tk.END, f"{dist_matrix[i][j]:5.1f} ")
                self.result_text.insert(tk.END, "\n")
            if n > 10:
                self.result_text.insert(tk.END, f"... (显示前{max_display}行{max_display}列)\n")

        elif 'matches' in result and result['matches']:
            self.result_text.insert(tk.END, "匹配结果:\n")
            for u, v in result['matches']:  # 只取两个元素 (u, v)，不再需要 w
                self.result_text.insert(tk.END, f"  {self.node_labels[u]} ←→ {self.node_labels[v]}\n")

    def visualize_algorithm_result(self, result):
        """可视化算法结果"""
        self.draw_graph(highlight_result=result)

    def draw_graph(self, highlight_result=None):
        """绘制图"""
        try:
            self.ax.clear()

            # 创建NetworkX图
            if self.graph_type.get() == "undirected":
                self.G = nx.Graph()
            else:
                self.G = nx.DiGraph()

            # 添加节点
            for i in range(self.node_count):
                self.G.add_node(i, label=self.node_labels[i])

            # 添加边
            edge_labels = {}
            for i in range(self.node_count):
                for j in range(self.node_count):
                    weight = self.adj_matrix[i][j]
                    if weight != 0 and weight != float('inf'):
                        self.G.add_edge(i, j, weight=weight)
                        edge_labels[(i, j)] = f"{weight:.1f}"

            # 计算节点位置
            if len(self.G.nodes()) > 0:
                self.pos = nx.spring_layout(self.G, seed=42)

                # 设置中文字体参数
                font_properties = {
                    'font_size': 12,
                    'font_weight': 'bold',
                }

                # 检查是否设置了中文字体
                import matplotlib.pyplot as plt
                if plt.rcParams['font.sans-serif']:
                    font_family = plt.rcParams['font.sans-serif'][0]
                    font_properties['font_family'] = font_family

                # 绘制图
                node_colors = ['lightblue' for _ in self.G.nodes()]
                edge_colors = ['gray' for _ in self.G.edges()]
                edge_widths = [1 for _ in self.G.edges()]

                # 高亮算法结果
                if highlight_result:
                    edge_colors, edge_widths = self.highlight_edges(highlight_result, edge_colors, edge_widths)

                nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_colors,
                                       node_size=500, alpha=0.8)
                nx.draw_networkx_labels(self.G, self.pos, ax=self.ax,
                                        labels={i: self.node_labels[i] for i in self.G.nodes()},
                                        **font_properties)
                nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color=edge_colors,
                                       width=edge_widths, alpha=0.7, arrows=self.graph_type.get() == "directed")

                # 绘制边权重
                nx.draw_networkx_edge_labels(self.G, self.pos, ax=self.ax,
                                             edge_labels=edge_labels, font_size=9)

                # 设置标题
                title_font = {'fontsize': 14, 'fontweight': 'bold'}
                if 'font_family' in font_properties:
                    title_font['fontname'] = font_properties['font_family']

                graph_type_text = "无向图" if self.graph_type.get() == "undirected" else "有向图"
                weight_text = "带权" if self.weighted_graph.get() else "无权"
                self.ax.set_title(f"{graph_type_text}可视化 ({self.node_count}个节点, {weight_text})", **title_font)
                self.ax.axis('off')

                self.canvas.draw()
        except Exception as e:
            print(f"绘制图失败: {e}")
            import traceback
            traceback.print_exc()

    def highlight_edges(self, result, edge_colors, edge_widths):
        """高亮算法结果中的边"""
        edge_list = list(self.G.edges())

        if 'edges' in result:  # 最小生成树算法
            for u, v, w in result['edges']:
                try:
                    idx = edge_list.index((u, v))
                    edge_colors[idx] = 'red'
                    edge_widths[idx] = 3
                except ValueError:
                    try:
                        idx = edge_list.index((v, u))
                        edge_colors[idx] = 'red'
                        edge_widths[idx] = 3
                    except ValueError:
                        pass

        elif 'paths' in result:  # 最短路径算法
            paths = result['paths']

            # 处理Dijkstra算法返回的路径结构
            if isinstance(paths, dict):
                for target_node, path_info in paths.items():
                    if 'path' in path_info:
                        path = path_info['path']
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            self.highlight_single_edge(u, v, edge_list, edge_colors, edge_widths, 'green', 2.5)

        return edge_colors, edge_widths

    def highlight_single_edge(self, u, v, edge_list, edge_colors, edge_widths, color, width):
        """高亮单条边"""
        try:
            idx = edge_list.index((u, v))
            edge_colors[idx] = color
            edge_widths[idx] = width
        except ValueError:
            try:
                idx = edge_list.index((v, u))
                edge_colors[idx] = color
                edge_widths[idx] = width
            except ValueError:
                pass

    def save_image(self):
        """保存图片"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图片已保存到: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图片失败: {str(e)}")

    def reset_graph(self):
        """重置图"""
        self.initialize_graph()
        self.result_text.delete(1.0, tk.END)
        messagebox.showinfo("重置", "图已重置！")
