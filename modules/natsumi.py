import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.grlc import GRLC, GRLC_DEVICE
import numpy as np

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

class Natsumi(nn.Module):
    def __init__(self, num_steps: int, num_agents: int, num_features: int, dim: int, dim_x: int, dropout: float) -> None:
        super(Natsumi, self).__init__()
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.num_nodes = num_steps * num_agents  # Total number of nodes
        self.num_features = num_features
        self.num_neg = num_neg_samples  # Number of negative samples
        self.hidden_dim = dim
        self.model = GRLC(
            n_nb=self.num_nodes,
            n_in=num_features,
            n_h=dim,     # Set the same as qcnet.hidden_dim
            dim_x=dim_x, # Hidden dim multiplier
            dropout=dropout,
            useact=True,
            liner=True,
            useA=True,   # Assumes True
        )
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        self.neg_matrices = []  # List to store negative sample matrices, generated randomly
        self.margin_loss = nn.MarginRankingLoss(margin=margin1, reduce=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 1. Gather input
        # Shape of x: (num_historical_steps, num_agents, hidden_dim)
        self.prepare_data(x, edge_index)
        # 3. Train
        self.do_train()
        # 4. Embed
        self.embed()
        # 5. Convert to edge_index
        return self.output_edge_index()

    def prepare_data(self, x: torch.Tensor, edge_index: torch.Tensor) -> None:
        # Prepare data for training
        self.real_num_agents = x.size(1)
        # assert self.num_features == x.size(1)
        self.num_edges = edge_index.size(1)
        self.edge_index = edge_index

        # Pad x to num_nodes and reshape
        # TODO: Set parameters in self
        x = F.pad(x, (0, 0, 0, self.num_agents - self.real_num_agents, 0, 0)).reshape(-1, self.hidden_dim).to(GRLC_DEVICE)
        edge_index = edge_index.to(GRLC_DEVICE)
        # print(f'prepare_data: {self.real_num_nodes=}, {x.shape=}, {edge_index.shape=}')

        # x = torch.FloatTensor(x)
        eps = 2.2204e-16
        norm = x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        x = x.div(norm.expand_as(x))
        self.x = x

        i = edge_index.long()
        v = torch.FloatTensor(torch.ones([self.num_edges])).to(GRLC_DEVICE)

        A_sp = torch.sparse.FloatTensor(i, v, torch.Size([self.num_nodes, self.num_nodes])).to(GRLC_DEVICE)
        A = A_sp.to_dense()
        I = torch.eye(A.shape[1]).to(GRLC_DEVICE)
        self.A_I = A + I
        self.A_I_nomal = normalize_graph(self.A_I)
        self.I_input = torch.eye(self.A_I_nomal.shape[1]).to(GRLC_DEVICE)

        self.gen_negative_samples()

    def gen_negative_samples(self) -> None:
        x = self.x
        feature_n = []
        nb_nodes = self.num_nodes
        for i in range(self.num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            feature_temp = x[idx_0].to(GRLC_DEVICE)
            feature_n.append(feature_temp)
        self.neg_matrices = feature_n

    def do_train(self) -> None:
        # print('do_train')
        model = self.model
        model.train()
        # FIXED: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # Move loss calculation and optimiser step to QCNet.training_step()
        # torch.set_grad_enabled(True)
        # self.optimiser.zero_grad()
        nb_nodes = self.num_nodes

        feature_X = self.x
        feature_a = feature_X
        feature_p = feature_X
        feature_n = self.neg_matrices
        lbl_z = torch.tensor([0.]).to(GRLC_DEVICE)

        h_a, h_p, h_n_lsit, h_a_0, h_p_0, h_n_0_list = model(feature_a, feature_p, feature_n, self.A_I_nomal, I=self.I_input)
        s_p = F.pairwise_distance(h_a, h_p)
        cos_0_list = []
        for h_n_0 in h_n_0_list:
            cos_0 = F.pairwise_distance(h_a_0, h_n_0)
            cos_0_list.append(cos_0)
        cos_0_stack = torch.stack(cos_0_list).detach()
        cos_0_min = cos_0_stack.min(dim=0)[0]
        cos_0_max = cos_0_stack.max(dim=0)[0]
        gap = cos_0_max - cos_0_min
        weight_list = []
        for i in range(cos_0_stack.size()[0]):
            weight_list.append((cos_0_stack[i] - cos_0_min) / gap)
        s_n_list = []
        s_n_cosin_list = []
        mask_I = (torch.eye(nb_nodes).to(GRLC_DEVICE) == 1)
        for h_n in h_n_lsit:
            # NOTE: Following code runs under certain condition: args.dataset_name in ['Cora', 'CiteSeer']
            s_n_cosin_list.append(cosine_dist(h_a, h_n)[mask_I].detach())
            # END NOTE
            s_n = F.pairwise_distance(h_a, h_n)
            s_n_list.append(s_n)
        self.s_n_cosin_list = s_n_cosin_list
        margin_label = -1 * torch.ones_like(s_p)
        loss_mar = 0
        mask_margin_N = 0
        i = 0
        for s_n in s_n_list:
            loss_mar += (self.margin_loss(s_p, s_n, margin_label) * weight_list[i]).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
            i += 1
        mask_margin_N = mask_margin_N / self.num_neg
        string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||".format(loss_mar.item(), mask_margin_N.item())
        self.loss = loss_mar * w_loss1 + mask_margin_N * w_loss2 / nb_nodes
        # NOTE: Following code runs under certain condition: args.dataset_name in ['Cora']
        # loss = loss_mar * w_loss1 + mask_margin_N * w_loss2
        # END NOTE

        # DONE: Integrate loss
        # self.loss.backward()
        # self.optimiser.step()

    def embed(self) -> None:
        # print('embed')
        model = self.model
        model.eval()
        # FIXED: RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # torch.set_grad_enabled(False)
        feature_a = feature_p = self.x
        feature_n = self.neg_matrices
        self.h_a, self.h_p = model.embed(feature_a, feature_p, feature_n, self.A_I_nomal, I=self.I_input)

    def output_edge_index(self) -> torch.Tensor:
        # NOTE: Following code runs under certain condition: args.dataset_name in ['Cora', 'CiteSeer']
        s_a = cosine_dist(self.h_a, self.h_a).detach()
        S = (torch.stack(self.s_n_cosin_list).mean(dim=0).expand_as(self.A_I) - s_a).detach()
        zero_vec = -9e15 * torch.ones_like(S)
        one_vec = torch.ones_like(S)
        s_a = torch.where(self.A_I_nomal > 0, one_vec, zero_vec)
        attention = torch.where(S < 0, s_a, zero_vec)
        attention_N = normalize_graph(attention)
        I = (torch.eye(self.num_nodes).to(GRLC_DEVICE) == 1)
        attention[I] = 0
        self.model.A = attention_N
        # END NOTE

        # Convert adjacency matrix attention_N to edge index
        edge_index = attention_N.nonzero().t().long()
        # FIXME: Algorithm triggers CUDA IndexError, filter range directly as a workaround
        # edge_index = filter_edges_unified(edge_index, self.num_steps, self.num_agents, self.real_num_agents)
        # Theoretic-incorrect code
        # The graph is undirected, so we only keep one direction of edges.
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        # Remove padding nodes from edge_index
        edge_index = edge_index[:, edge_index[1] < self.num_steps * self.real_num_agents]
        return edge_index
