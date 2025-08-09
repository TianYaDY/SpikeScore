import torch
import torch.nn.functional as F

def deterministic_neural_sort(s, tau=1.0, mask=None):
    # s: [B, n, 1]
    # 输出: [B, n, n]
    B, n, _ = s.shape
    device = s.device
    one = torch.ones((n, 1), device=device)
    s = s.squeeze(-1)
    A_s = torch.abs(s.unsqueeze(-1) - s.unsqueeze(-2))
    B_mat = torch.matmul(A_s, torch.arange(1, n+1, device=device, dtype=s.dtype).view(1, -1))
    scaling = n + 1 - 2 * torch.arange(1, n+1, device=device, dtype=s.dtype)
    C = torch.matmul(s, scaling.view(-1, 1))
    P_max = (C - B_mat).transpose(-2, -1)
    if mask is not None:
        P_max = P_max.masked_fill(mask.unsqueeze(-1), float('-inf'))
    P_hat = F.softmax(P_max / tau, dim=-1)
    return P_hat

def sinkhorn_scaling(P, mask, tol=1e-6, max_iter=10):
    # P: [B, n, n]
    # mask: [B, n]
    for _ in range(max_iter):
        # Row normalization
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-12)
        # Column normalization
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-12)
        if mask is not None:
            P = P.masked_fill(mask.unsqueeze(-1), 0.0)
            P = P.masked_fill(mask.unsqueeze(1), 0.0)
    return P

def dcg(y_true, y_pred, ats=[None], gain_function=lambda x: torch.pow(2., x) - 1):
    # y_true, y_pred: [B, n]
    # ats: list of cutoff points (k)
    results = []
    gain = gain_function(y_true)
    sorted_gain, _ = torch.sort(gain, descending=True, dim=-1)
    for k in ats:
        if k is None:
            k = y_true.size(-1)
        discounts = 1.0 / torch.log2(torch.arange(k, device=y_true.device, dtype=y_true.dtype) + 2.0)
        dcg_k = (sorted_gain[..., :k] * discounts).sum(dim=-1, keepdim=True)
        results.append(dcg_k)
    return torch.cat(results, dim=-1)





def neuralNDCGLoss(y_pred, y_true, padded_value_indicator=-1, temperature=1., powered_relevancies=True, k=None):
    """
    Minimal NeuralNDCGLoss with deterministic_neural_sort and sinkhorn_scaling.
    y_pred, y_true: [batch_size, slate_length]
    """
    dev = y_true.device
    B, n = y_pred.shape
    if k is None:
        k = n
    mask = (y_true == padded_value_indicator)
    # 1. 排序概率矩阵
    P_hat = deterministic_neural_sort(y_pred.unsqueeze(-1), tau=temperature, mask=mask).unsqueeze(0)
    # 2. Sinkhorn scaling
    P_hat = sinkhorn_scaling(P_hat.view(B, n, n), mask, tol=1e-6, max_iter=10)
    # 3. 应用 mask 和 true labels
    P_hat = P_hat.masked_fill(mask.unsqueeze(-1) | mask.unsqueeze(1), 0.)
    y_true_masked = y_true.masked_fill(mask, 0.).unsqueeze(-1)
    if powered_relevancies:
        y_true_masked = torch.pow(2., y_true_masked) - 1.
    ground_truth = torch.matmul(P_hat, y_true_masked).squeeze(-1)
    discounts = (1.0 / torch.log2(torch.arange(n, dtype=torch.float, device=dev) + 2.0))
    discounted_gains = ground_truth * discounts
    # 4. 理想DCG（IDCG）
    if powered_relevancies:
        idcg = dcg(y_true, y_true, ats=[k]).permute(1, 0)
    else:
        idcg = dcg(y_true, y_true, ats=[k], gain_function=lambda x: x).permute(1, 0)
    discounted_gains = discounted_gains[:, :k]
    ndcg = discounted_gains.sum(dim=-1, keepdim=True) / (idcg + 1e-10)
    idcg_mask = idcg == 0.
    ndcg = ndcg.masked_fill(idcg_mask, 0.)
    if idcg_mask.all():
        return torch.tensor(0., device=dev)
    mean_ndcg = ndcg.sum() / ((~idcg_mask).sum())
    return -1. * mean_ndcg
