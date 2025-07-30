import torch

def filter_specific_edges(edge_index, A, T, x):
    # edge_index: (2, num_edges) tensor
    # A, T, x: parameters defining specific values i*A + x for i in range(T)

    # Generate specific values for i in range(T)
    specific_values = torch.tensor([i * A + x for i in range(T)], dtype=torch.long, device=edge_index.device)
    #print(f'{specific_values=}')

    # Get source and target node indices
    src = edge_index[0]
    dst = edge_index[1]

    # Create mask for edges where either source or target is in specific_values
    src_mask = torch.isin(src, specific_values)
    dst_mask = torch.isin(dst, specific_values)
    valid_edges = src_mask | dst_mask

    # Filter edge_index using the valid_edges mask
    filtered_edge_index = edge_index[:, valid_edges]

    # Collect non-specific value vertices
    non_specific_vertices = []

    # Iterate over filtered edges
    for edge_idx in range(filtered_edge_index.shape[1]):
        src_node = filtered_edge_index[0, edge_idx]
        dst_node = filtered_edge_index[1, edge_idx]

        # If source is specific, collect destination if it's not specific
        if src_mask[edge_idx] and not torch.isin(dst_node, specific_values):
            non_specific_vertices.append(dst_node)
        # If destination is specific, collect source if it's not specific
        if dst_mask[edge_idx] and not torch.isin(src_node, specific_values):
            non_specific_vertices.append(src_node)

    # Convert to tensor and remove duplicates
    if non_specific_vertices:
        non_specific_vertices = torch.unique(
            torch.tensor(non_specific_vertices, dtype=torch.long, device=edge_index.device))
    else:
        non_specific_vertices = torch.tensor([], dtype=torch.long, device=edge_index.device)

    # Convert non-specific vertices to (i, j) form
    non_specific_ij = []
    for v in non_specific_vertices:
        i = v // A
        j = v % A
        if 0 <= i < T:  # Ensure valid i
            non_specific_ij.append([i, j])

    # Convert to tensor of shape [2, K]
    if non_specific_ij:
        non_specific_ij_tensor = torch.tensor(non_specific_ij, dtype=torch.long, device=edge_index.device).t()
    else:
        non_specific_ij_tensor = torch.tensor([[], []], dtype=torch.long, device=edge_index.device)

    return filtered_edge_index, non_specific_vertices, non_specific_ij_tensor


def check_nan(tensor, message="Tensor contains NaN values"):
    """
    检查张量中是否存在 NaN 值，若存在则打印消息。

    Args:
        tensor (torch.Tensor): 输入张量
        message (str): 发现 NaN 值时打印的消息，默认为 "Tensor contains NaN values"

    Returns:
        bool: True 如果张量包含 NaN，False 否则
    """
    has_nan = torch.isnan(tensor).any()
    if has_nan:
        print(message)
    return has_nan
