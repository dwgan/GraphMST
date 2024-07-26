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
