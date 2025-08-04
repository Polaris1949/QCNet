import torch

# Dataset-specific loss-related parameters.
lr = 0.005
w_loss1 = 2.0
w_loss2 = 0.001
margin1 = 0.8
margin2 = 0.2
my_margin_2 = margin1 + margin2

# Algorithm parameters.
num_neg_samples = 10

def slide_sequence(seq, num, step):
    """
    将序列按每num个元素以step步进打包，返回迭代器。

    Args:
        seq: 输入序列
        num: 每个分组的元素数量
        step: 每次移动的步长

    Yields:
        tuple: 包含num个元素的元组
    """
    for i in range(0, len(seq) - num + 1, step):
        yield i, seq[i:i + num]


def filter_edges_unified(edge_index, num_steps, num_agents, limit):
    """
    筛选边集张量中顶点编号在所有 range(i, i*num_agents+limit) 并集内的边，
    并将顶点编号从 i*num_agents+j 映射为 i*limit+j，其中 j 在 range(limit)。

    Args:
        edge_index (torch.Tensor): 边集张量，形状 [2, num_edges]，每列为 [source_node, target_node]
        num_steps (int): 步数
        num_agents (int): 每个步长的代理数
        limit (int): 顶点编号的范围限制

    Returns:
        torch.Tensor: 筛选并重新映射顶点编号后的边集张量，形状 [2, num_filtered_edges]
    """
    # 生成所有 node_range 的并集
    node_set = set()
    for i in range(num_steps):
        node_set.update(range(i, i * num_agents + limit))

    # 转换为布尔掩码
    node_mask = torch.tensor([n in node_set for n in range(edge_index.max() + 1)],
                             dtype=torch.bool, device=edge_index.device)

    # 筛选边：起点和终点都在 node_set 内
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    filtered_edge = edge_index[:, edge_mask]

    # 创建顶点编号映射：从 i*num_agents+j 映射到 i*limit+j
    mapping = {}
    for i in range(num_steps):
        for j in range(limit):
            old_node = i * num_agents + j
            if old_node in node_set:  # 仅映射在 node_set 内的顶点
                new_node = i * limit + j
                mapping[old_node] = new_node

    # 应用映射到筛选后的边集
    mapped_edge = filtered_edge
    for old_node, new_node in mapping.items():
        mapped_edge[mapped_edge == old_node] = new_node

    return mapped_edge

def normalize_graph(A: torch.Tensor) -> torch.Tensor:
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bs1 = x.size()[0]
    bs2 = y.size()[0]
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2) + 0.00000001, 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2) + 0.00000001, 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine
