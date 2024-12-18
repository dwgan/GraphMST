@[toc]
# 前言
最近在研究将序列模型（例如Mamba）应用到图像处理的深度学习任务中，发现将图像数据转换成序列数据总会存在空间位置信息的损失。由于这种类RNN的模型固有的遗忘问题，在处理图象时无论是逐行逐列扫描还是多个方向混合扫描，都不能够自适应的按照图像像素的相关性进行先后排序，这将导致图像信息处理的瓶颈。

而图结构能够提供一个很强大的表征方式，如果将像素点当成图的顶点，像素关系当成图的边权重，然后通过构建最小生成树，则可以建立一种自适应的图像信息和序列信息的转换桥梁，因此希望研究一下如何将图结构进行进一步的应用。

为了深刻的理解图结构，这里从一个最简单的用最小生成树表示无向图的例子进行切入。
# 什么是无向图

无向图（Undirected Graph）是一种图结构，在这种图中，边（Edge）没有方向性。换句话说，如果存在一条边连接两个顶点（Vertex），则可以从任意一个顶点到达另一个顶点。无向图通常表示为 \( G = (V, E) \)，其中：

- \( V \) 是顶点集合。
- \( E \) 是边集合，由顶点对组成的集合。

在无向图中，边 \( (u, v) \) 表示顶点 \( u \) 和顶点 \( v \) 之间存在一条边，且 \( (u, v) \) 和 \( (v, u) \) 是等价的。

**示例**

考虑一个无向图 \( G \) ，其顶点集合 \( V = \{1, 2, 3, 4\} \) ，边集合 \( E = \{(1, 2), (2, 3), (3, 4), (4, 1)\} \)。这个图可以表示为：

```bash
    1 -- 2
    |    |
    4 -- 3
```

**无向图的特性**

1. **对称性**：如果边 \( (u, v) \) 存在，则边 \( (v, u) \) 也存在。
2. **度（Degree）**：一个顶点的度是连接到该顶点的边的数量。
3. **连通性**：如果从图中的任意一个顶点都能到达其他任意一个顶点，则这个图是连通的。

无向图在很多领域都有应用，例如网络拓扑、社交网络分析、生物网络等。
# 什么是最小生成树

最小生成树（Minimum Spanning Tree，简称 MST）是无向加权图中的一个子图，它包含图中的所有顶点，并且使得所有边的权重之和最小。换句话说，最小生成树是连接图中所有顶点的权重最小的无环子图。

**特性**

1. **唯一性**：一个无向图可能有多个最小生成树，但对于给定的边权重集合，最小生成树的总权重是唯一的。
2. **连通性**：最小生成树必须包含图中的所有顶点，且是连通的。
3. **无环性**：最小生成树是无环的，否则可以通过去掉环中的某条边来减少总权重。

**应用**

最小生成树在许多实际问题中有重要应用，例如：

- **网络设计**：设计最小成本的网络连接方案，如电网、通信网络等。
- **集群分析**：在数据聚类中，用最小生成树来找到数据点之间的最短连接路径。
- **图像处理**：在图像分割中，使用最小生成树来连接像素点。

**常见算法**

1. **Kruskal算法**：通过排序所有边并逐步选择不形成环的最小边来构建最小生成树。
2. **Prim算法**：从一个顶点开始，逐步选择最小边扩展生成树，直到包括所有顶点。

# 算法思路

对于一个具有n个顶点的无向图，最多有n(n−1)2条边\frac{n(n-1)}{2}条边

举一个具有6个顶点的无向图为例，其中(0),(1),(2),(3),(4),(5)(0), (1), (2), (3), (4), (5)表示6个顶点，顶点之间的连线上的数字表示边的权重

这里将采用Kruskal算法构建最小生成树


```bash
 (0)-1-(1)
  | \   |  \
  5  2  3   3
  |   \ |    \
 (2)   (3)-1-(5)
   \    |    /
    3   2   4
     \  |  /
       (4)
```
这里设置了9条边,将它表示为列表，并且按照边的权重从小到大排序

```bash
[
# (顶点1,顶点2,边权重)
(0, 1, 1),
(3, 5, 1),
(0, 3, 2),
(3, 4, 2),
(1, 5, 3),
(1, 3, 3),
(2, 4, 3),
(4, 5, 4),
(0, 2, 5),
]
```
利用迭代来找出最小的连通图。

**第一轮迭代**

**选择最小边**

```bash
对于连通分量0，最小的边是(0, 1, 1)
对于连通分量1，最小的边是(0, 1, 1)
对于连通分量2，最小的边是(2, 4, 3)
对于连通分量3，最小的边是(3, 5, 1)
对于连通分量4，最小的边是(3, 4, 2)
对于连通分量5，最小的边是(3, 5, 1)
```
**合并连通分量**


选择(0, 1, 1)合并连通分量0和1。
```bash
 (0)-1-(1)
            
            
  
 (2)   (3)  (5)
 
 
 
       (4)
```
选择(3, 5, 1)合并连通分量3和5。
```bash
 (0)-1-(1)
            
            
  
 (2)   (3)-1-(5)
 
 
 
       (4)
```

选择(3, 4, 2)合并连通分量3和4。
```bash
 (0)-1-(1)
            
            
  
 (2)   (3)-1-(5)
        |
        2
        |
       (4)
```

选择(2, 4, 3)合并连通分量2和4。
```bash
 (0)-1-(1)
            
            
  
 (2)   (3)-1-(5)
   \    |
    3   2
     \  |
       (4)
```
更新并查集

**第二轮迭代**

**选择最小边**

```bash
对于连通分量0-1，最小的边是(0, 3, 2)
对于连通分量2-3-4-5，最小的边是(0, 3, 2)
```
选择(0, 3, 2)合并连通分量0和3。
```bash
 (0)-1-(1)
    \
     2
      \ 
 (2)   (3)-1-(5)
   \    |
    3   2
     \  |
       (4)
```
更新并查集，发现所有的顶点都已连通，算法结束



# 算法实现

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)

        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def contractive_boruvka(edges, n):
    uf = UnionFind(n)
    mst_edges = []
    mst_weight = 0
    iteration = 0

    while len(mst_edges) < n - 1:
        iteration += 1
        print(f"Iteration {iteration}:")

        # 初始化每个组件的最小边
        min_edge = [-1] * n

        for index, (u, v, weight) in enumerate(edges):
            root_u = uf.find(u)
            root_v = uf.find(v)
            if root_u != root_v:
                if min_edge[root_u] == -1 or edges[min_edge[root_u]][2] > weight:
                    min_edge[root_u] = index
                if min_edge[root_v] == -1 or edges[min_edge[root_v]][2] > weight:
                    min_edge[root_v] = index

        # 收缩每个组件的最小边
        for edge_index in min_edge:
            if edge_index != -1:
                u, v, weight = edges[edge_index]
                root_u = uf.find(u)
                root_v = uf.find(v)
                if root_u != root_v:
                    uf.union(u, v)
                    mst_edges.append((u, v, weight))
                    mst_weight += weight
                    print(f"  Connecting {u} - {v} with weight {weight}")

        print(f"  MST so far: {mst_edges}")
        print()

    return mst_edges, mst_weight

# 示例使用
edges = [
	# (顶点1,顶点2,边权重)
	(0, 1, 1),
	(3, 5, 1),
	(0, 3, 2),
	(3, 4, 2),
	(1, 5, 3),
	(1, 3, 3),
	(2, 4, 3),
	(4, 5, 4),
	(0, 2, 5),
]
n = 6

mst_edges, mst_weight = contractive_boruvka(edges, n)
print("Final MST edges:", mst_edges)
print("Total weight of MST:", mst_weight)
```
**输出**
```python
Iteration 1:
  Connecting 0 - 1 with weight 1
  Connecting 2 - 4 with weight 3
  Connecting 3 - 5 with weight 1
  Connecting 3 - 4 with weight 2
  MST so far: [(0, 1, 1), (2, 4, 3), (3, 5, 1), (3, 4, 2)]

Iteration 2:
  Connecting 0 - 3 with weight 2
  MST so far: [(0, 1, 1), (2, 4, 3), (3, 5, 1), (3, 4, 2), (0, 3, 2)]

Final MST edges: [(0, 1, 1), (2, 4, 3), (3, 5, 1), (3, 4, 2), (0, 3, 2)]
Total weight of MST: 9
```
>[GrootVL: Tree Topology is All You Need in State Space Model](https://arxiv.org/abs/2406.02395)
