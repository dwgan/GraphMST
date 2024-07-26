import triton
import triton.language as tl


@triton.jit
def find_kernel(parent, u, ret_ptr, BLOCK_SIZE: tl.constexpr):
    pu = tl.load(parent + u)
    while pu != u:
        u = pu
        pu = tl.load(parent + u)
    ret_ptr[u % BLOCK_SIZE] = pu


@triton.jit
def union_kernel(parent, rank, u, v, BLOCK_SIZE: tl.constexpr):
    # Create local arrays to store roots and rank updates
    root_u = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    root_v = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    # Finding the roots of u and v
    find_kernel(parent, u, root_u, BLOCK_SIZE=BLOCK_SIZE)
    find_kernel(parent, v, root_v, BLOCK_SIZE=BLOCK_SIZE)
    # Synchronize to make sure all threads have found their roots
    tl.syncwarp()
    root_u = root_u[0]
    root_v = root_v[0]

    if root_u != root_v:
        ru_rank = tl.load(rank + root_u)
        rv_rank = tl.load(rank + root_v)
        if ru_rank > rv_rank:
            tl.store(parent, root_v, root_u)
        elif ru_rank < rv_rank:
            tl.store(parent, root_u, root_v)
        else:
            tl.store(parent, root_v, root_u)
            tl.atomic_add(rank, root_u, 1)


# 主函数中使用 Triton 内核
def kruskal_triton(edges, n):
    edges = sorted(edges, key=lambda x: x[2])  # Sort by weight
    parent = triton.testing.randn((n,), dtype=triton.int32, generator=triton.manual_seed(0))
    rank = triton.testing.zeros((n,), dtype=triton.int32)

    for i in range(n):
        parent[i] = i

    mst = []
    for u, v, weight in edges:
        if len(mst) == n - 1:
            break
        if find_kernel(parent, u)[0] != find_kernel(parent, v)[0]:
            union_kernel(parent, rank, u, v, BLOCK_SIZE=32)
            mst.append((u, v, weight))

    return mst


# 使用示例
edges = [
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

mst = kruskal_triton(edges, n)
print("Final MST edges:", mst)
