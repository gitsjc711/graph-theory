import numpy as np


# ==================== 最小生成树算法 ====================

def kruskal(adj_matrix):
    """
    Kruskal算法实现最小生成树
    参数: adj_matrix - 邻接矩阵
    返回: dict - 包含MST边和总权重的字典
    1.将邻接矩阵中的所有边按权重从小到大排序
    2.依次选择边，如果这条边不会形成环（通过是否一队判断是否会成环），就加入最小生成树
    3.直到选择了n-1条边（n为节点数）
    """
    n = len(adj_matrix)

    # 收集所有边
    edges = []
    for i in range(n):
        for j in range(i + 1, n):  # 无向图，只考虑上三角
            weight = adj_matrix[i][j]
            if weight != 0 and weight != float('inf'):
                edges.append((weight, i, j))

    # 按权重排序
    edges.sort(key=lambda x: x[0])

    # 并查集实现
    # 初始化每个节点的父节点为自身
    parent = list(range(n))

    def find(x):
        """查找根节点（路径压缩）"""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        """合并两个集合"""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x
            return True
        return False

    # Kruskal主算法
    mst_edges = []
    total_weight = 0
    edges_selected = 0

    for weight, u, v in edges:
        if union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            edges_selected += 1
            if edges_selected == n - 1:
                break

    return {
        'algorithm': 'Kruskal',
        'edges': mst_edges,
        'total_weight': total_weight,
        'node_count': n,
        'edges_count': len(mst_edges)
    }


def prim(adj_matrix, start_node=0):
    """
    Prim算法实现最小生成树
    参数:
        adj_matrix - 邻接矩阵
        start_node - 起始节点索引（默认0）
    返回: dict - 包含MST边和总权重的字典
    1.初始化，从指定的节点开始，设置所有的节点未被访问
    2.寻找下一个顶点：在未访问的顶点中，找到 min_edge值最小的顶点 u。这个顶点是当前“距离”已生成部分树最近的顶点。
    3.收录顶点：将顶点 u标记为已访问 (visited[u] = True)。将 u对应的 min_edge值累加到总权重 total_weight中。如果 u不是起始节点（即 parent[u] != -1），则将边 (parent[u], u, min_edge[u])加入结果边集 mst_edges。这条边就是连接 u到当前生成树的最小边。
    4.更新邻居信息：遍历 u的所有邻居顶点 v。对于每个未被访问且与 u相连（权重有效且非零）的邻居，检查边 (u, v)的权重是否小于该邻居当前记录的 min_edge[v]。如果是，则更新 min_edge[v]为这个更小的权重，并设置 parent[v] = u，表示当前看来，v最好通过边 (u, v)加入生成树。
    """
    n = len(adj_matrix)

    # 初始化数据结构
    visited = [False] * n
    min_edge = [float('inf')] * n
    parent = [-1] * n

    min_edge[start_node] = 0
    total_weight = 0
    mst_edges = []

    for _ in range(n):
        # 找到未访问节点中具有最小边的节点
        u = -1
        min_val = float('inf')
        for i in range(n):
            if not visited[i] and min_edge[i] < min_val:
                min_val = min_edge[i]
                u = i

        if u == -1:
            break

        visited[u] = True
        total_weight += min_edge[u]

        if parent[u] != -1:
            mst_edges.append((parent[u], u, min_edge[u]))

        # 更新相邻节点的最小边
        for v in range(n):
            weight = adj_matrix[u][v]
            if (not visited[v] and weight != 0 and weight != float('inf')
                    and weight < min_edge[v]):
                min_edge[v] = weight
                parent[v] = u

    return {
        'algorithm': 'Prim',
        'edges': mst_edges,
        'total_weight': total_weight,
        'start_node': start_node,
        'node_count': n
    }


def break_cycle(adj_matrix):
    """
    破圈法实现最小生成树
    参数: adj_matrix - 邻接矩阵
    返回: dict - 包含MST边和总权重的字典
    1.在图中寻找环，并将访问过的环顶点标记为已访问，若已经访问则不进入递归。
    2.寻找过程中记录当前环中的权重最大边，及其对应信息。
    3.寻找环的过程闭合之后，去除最大边。
    4.重复步骤1-3，直到图中不存在环为止。
    """
    n = len(adj_matrix)

    # 收集所有边并降序排序（先移除权重大的边）
    edges = []
    for i in range(n):
        for j in range(i + 1, n):  # 无向图
            weight = adj_matrix[i][j]
            if weight != 0 and weight != float('inf'):
                edges.append((weight, i, j))

    edges.sort(reverse=True, key=lambda x: x[0])

    # 使用并查集构建MST
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x
            return True
        return False

    # 构建MST边列表
    mst_edges = []
    total_weight = 0

    # 按权重从大到小遍历边
    for weight, u, v in edges:
        # 尝试连接u和v
        if union(u, v):
            # 如果连接成功，说明不会形成环，加入MST
            mst_edges.append((u, v, weight))
            total_weight += weight

    # 破圈法需要保留n-1条边，所以取前n-1条边
    if len(mst_edges) > n - 1:
        mst_edges = mst_edges[:n - 1]
        # 重新计算总权重
        total_weight = sum(edge[2] for edge in mst_edges)

    return {
        'algorithm': '破圈法',
        'edges': mst_edges,
        'total_weight': total_weight,
        'node_count': n
    }


# ==================== 最短路径算法 ====================

def dijkstra(adj_matrix, start_node=0):
    """
    Dijkstra算法实现单源最短路径
    参数:
        adj_matrix - 邻接矩阵
        start_node - 起始节点索引（默认0）
    返回: dict - 包含最短路径和距离的字典
    1.初始状态：只知道起点到自己的距离是0，到其他点的距离未知（视为无穷大）。
    2.逐步扩张：每一轮都从尚未确定最短路径的顶点中，选择一个 距离起点最近的顶点。
    3.松弛操作：将这个新确定的顶点作为“跳板”，检查是否能通过它来缩短起点到其邻居顶点的距离。如果能，就更新这个距离。
    4.重复：重复步骤2和3，直到所有顶点的最短路径都被确定。
    """
    n = len(adj_matrix)

    # 初始化
    distances = [float('inf')] * n
    predecessors = [-1] * n
    visited = [False] * n

    distances[start_node] = 0

    for _ in range(n):
        # 找到未访问节点中距离最小的节点
        u = -1
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i

        if u == -1 or distances[u] == float('inf'):
            break

        visited[u] = True

        # 更新相邻节点的距离
        for v in range(n):
            weight = adj_matrix[u][v]
            if (not visited[v] and weight != 0 and weight != float('inf')):
                new_distance = distances[u] + weight
                if new_distance < distances[v]:
                    distances[v] = new_distance
                    predecessors[v] = u

    # 构建路径信息
    paths = {}
    for i in range(n):
        if distances[i] == float('inf'):
            paths[i] = {'distance': float('inf'), 'path': []}
        else:
            # 回溯构建路径
            path = []
            current = i
            while current != -1:
                path.insert(0, current)
                current = predecessors[current]
            paths[i] = {'distance': distances[i], 'path': path}

    return {
        'algorithm': 'Dijkstra',
        'start_node': start_node,
        'distances': distances,
        'predecessors': predecessors,
        'paths': paths,
        'node_count': n
    }


def floyd(adj_matrix):
    """
    Floyd算法实现所有节点对最短路径
    参数: adj_matrix - 邻接矩阵
    返回: dict - 包含所有迭代阶段距离矩阵的字典
    1. 定义D^(k)为经过前k个中间节点的最短距离矩阵
    """
    n = len(adj_matrix)

    # 初始化D^(0)：边权矩阵
    D = [np.full((n, n), float('inf')) for _ in range(n + 1)]  # D[0]到D[n]

    # 初始化D^(0) = 原始邻接矩阵
    for i in range(n):
        for j in range(n):
            if i == j:
                D[0][i][j] = float('inf')
            else:
                weight = adj_matrix[i][j]
                if weight == 0:
                    D[0][i][j] = float('inf')  # 无边相连
                else:
                    D[0][i][j] = weight
    # 动态规划递推：计算D^(1)到D^(n)
    for k in range(1, n + 1):  # k表示中间节点编号（1-based）
        for w in range(n):
            for i in range(n):
                for j in range(n):
                    # 递推公式：d_ij^(k) = min(d_ij^(k-1), d_i(k-1)^(k-1) + d_(k-1)j^(k-1))
                        through_k = D[k - 1][i][w] + D[0][w][j]
                        original = D[k][i][j]
                        D[k][i][j] = min(original, through_k)
    result=np.full((n, n), float('inf'))
    for k in range(n+1):
        for i in range(n):
            for j in range(n):
                if D[k][i][j]<result[i][j]:
                    result[i][j]=D[k][i][j]



    # 最终结果：D^(n)包含所有节点对的最短距离
    return {
        'algorithm': 'Floyd',
        'distance_matrix': result,  # D^(n)是最终结果
        'node_count': n
    }


def floyd_warshall(adj_matrix):
    """
    Floyd-Warshall算法实现所有节点对最短路径
    参数: adj_matrix - 邻接矩阵
    返回: dict - 包含最短路径距离矩阵和路径矩阵的字典
    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    """
    n = len(adj_matrix)

    # 初始化距离矩阵和路径记录矩阵
    dist = adj_matrix.copy().astype(float)
    next_node = np.full((n, n), -1, dtype=int)

    # 初始化
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
                next_node[i][j] = i
            elif adj_matrix[i][j] != 0 and adj_matrix[i][j] != float('inf'):
                dist[i][j] = adj_matrix[i][j]
                next_node[i][j] = j
            else:
                dist[i][j] = float('inf')

    # Floyd-Warshall算法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    # 构建路径函数
    def get_path(i, j):
        if dist[i][j] == float('inf'):
            return []
        path = [i]
        while i != j:
            i = next_node[i][j]
            path.append(i)
        return path

    # 为所有节点对生成路径
    paths = {}
    for i in range(n):
        paths[i] = {}
        for j in range(n):
            paths[i][j] = get_path(i, j)

    return {
        'algorithm': 'Floyd-Warshall',
        'distance_matrix': dist,
        'next_matrix': next_node,
        'paths': paths,
        'node_count': n
    }


# ==================== 匹配算法 ====================

def hungarian(cost_matrix):
    """
    匈牙利算法解决最大匹配问题（二分图最大权重匹配）
    修正后的实现
    """
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0

    if n == 0 or m == 0:
        return {
            'algorithm': '匈牙利算法',
            'matching': {},
            'matches': [],
            'total_cost': 0,
            'row_count': n,
            'col_count': m
        }

    # 确保矩阵是方阵
    size = max(n, m)
    cost = np.full((size, size), 0, dtype=float)
    cost[:n, :m] = cost_matrix

    # 转换为最小化问题（将求最大匹配转换为求最小匹配）
    max_val = np.max(cost)
    cost = max_val - cost[:size, :size]

    # 初始化
    u = np.zeros(size + 1, dtype=float)  # 左侧顶点的势
    v = np.zeros(size + 1, dtype=float)  # 右侧顶点的势
    p = np.zeros(size + 1, dtype=int)  # 右侧顶点匹配的左侧顶点
    way = np.zeros(size + 1, dtype=int)  # 路径

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, float('inf'))
        used = np.zeros(size + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0

            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # 增广
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 提取匹配结果
    matching = {}
    matches = []
    total_cost = 0.0

    for j in range(1, size + 1):
        if p[j] != 0:
            i = p[j] - 1
            j_idx = j - 1
            if i < n and j_idx < m:
                matching[i] = j_idx
                weight = cost_matrix[i][j_idx]
                matches.append((i, j_idx, weight))
                total_cost += weight

    return {
        'algorithm': '匈牙利算法',
        'matching': matching,
        'matches': matches,
        'total_cost': total_cost,
        'row_count': n,
        'col_count': m
    }


def kuhn_munkres(cost_matrix, is_max=True):
    """
    修正的Kuhn-Munkres算法
    """
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0

    if n == 0 or m == 0:
        return {
            'algorithm': 'Kuhn-Munkres算法',
            'matches': [],
            'total_weight': 0,
            'is_maximization': is_max,
            'row_count': n,
            'col_count': m
        }

    # 确保矩阵是方阵
    size = max(n, m)
    cost = np.full((size, size), 0, dtype=float)
    cost[:n, :m] = cost_matrix

    # 如果求最大匹配，转换为最小化问题
    if is_max:
        max_val = np.max(cost)
        cost = max_val - cost

    # 初始化
    u = np.zeros(size, dtype=float)
    v = np.zeros(size, dtype=float)
    p = np.zeros(size, dtype=int) - 1
    way = np.zeros(size, dtype=int) - 1

    for i in range(size):
        p[0] = i
        j0 = 0
        minv = np.full(size, float('inf'))
        used = np.zeros(size, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0

            for j in range(1, size):
                if not used[j]:
                    cur = cost[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(size):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == -1:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 提取结果
    matches = []
    total_weight = 0.0

    for j in range(size):
        if p[j] != -1:
            i = p[j]
            if i < n and j < m:
                weight = cost_matrix[i][j]
                matches.append((i, j, weight))
                total_weight += weight

    return {
        'algorithm': 'Kuhn-Munkres算法',
        'matches': matches,
        'total_weight': total_weight,
        'is_maximization': is_max,
        'row_count': n,
        'col_count': m
    }


# ==================== 工具函数 ====================

def validate_graph_matrix(matrix):
    """
    验证邻接矩阵是否有效
    参数: matrix - 输入的邻接矩阵
    返回: (bool, str) - 是否有效和错误信息
    """
    try:
        matrix = np.array(matrix)
        n = len(matrix)

        # 检查矩阵是否为方阵
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            return False, "邻接矩阵必须是方阵"

        # 检查节点数量
        if n > 26:
            return False, "节点数量不能超过26个"

        # 检查对角线元素（自环）
        for i in range(n):
            if matrix[i][i] != 0:
                return False, f"节点 {i} 存在自环（对角线元素应为0）"

        return True, "邻接矩阵有效"

    except Exception as e:
        return False, f"矩阵验证失败: {str(e)}"


def matrix_to_adjacency_list(matrix):
    """
    将邻接矩阵转换为邻接表
    参数: matrix - 邻接矩阵
    返回: dict - 邻接表
    """
    n = len(matrix)
    adj_list = {}

    for i in range(n):
        neighbors = []
        for j in range(n):
            if i != j and matrix[i][j] != 0 and matrix[i][j] != float('inf'):
                neighbors.append((j, matrix[i][j]))
        adj_list[i] = neighbors

    return adj_list


def generate_random_matrix(n, max_weight=10, directed=False, density=0.5):
    """
    生成随机邻接矩阵
    参数:
        n - 节点数量
        max_weight - 最大权重
        directed - 是否是有向图
        density - 边的密度（0-1）
    返回: np.ndarray - 随机邻接矩阵
    """
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            elif np.random.random() < density:
                weight = np.random.randint(1, max_weight + 1)
                matrix[i][j] = weight
                if not directed:
                    matrix[j][i] = weight
            else:
                matrix[i][j] = 0 if directed else float('inf')

    return matrix

