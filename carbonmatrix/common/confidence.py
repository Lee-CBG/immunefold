import torch
from torch.nn import functional as F
from einops import rearrange

def compute_plddt(logits):
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bin_centers = torch.arange(start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device)
    probs = F.softmax(logits, dim=-1)
    predicted_lddt_ca = torch.sum(probs * bin_centers[None, None, :], axis=-1)

    return predicted_lddt_ca * 100

def compute_ptm(logits, breaks, mask, chain_id=None, interface=True):
    bin_centers = _calculate_bin_centers(breaks) # (bins,)

    num_res = torch.sum(mask, dim=-1, keepdims=True) #(bs, 1)
    clipped_num_res = torch.clip(num_res, 19, 100000)

    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    probs = F.softmax(logits, dim=-1) #(bs, n, n, d)
    
    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2)) # (bs, bins)
    tm_per_bin = rearrange(tm_per_bin, 'b d -> b () () d') # (bs, 1, 1, bins)
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1) # (bs, n, n)

    pair_mask = rearrange(mask, 'b n -> b n ()' ) * rearrange(mask, 'b n -> b () n')
    if interface and (chain_id is not None):
        pair_mask *= (chain_id[..., None] != chain_id[..., None, :])
    # (bs, n,) / (bs, 1, ) = (bs, n)
    per_alignment = torch.sum(predicted_tm_term * pair_mask, dim=-1) / (1e-8 + num_res)

    ptm, _ = torch.max(per_alignment, dim=-1)

    return ptm

def compute_pair_iptm(logits, breaks, mask, chain_id):
    bin_centers = _calculate_bin_centers(breaks) # (bins,)

    num_res = torch.sum(mask, dim=-1, keepdims=True) #(bs, 1)
    clipped_num_res = torch.clip(num_res, 19, 100000)

    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    probs = F.softmax(logits, dim=-1) #(bs, n, n, d)
    
    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2)) # (bs, bins)
    tm_per_bin = rearrange(tm_per_bin, 'b d -> b () () d') # (bs, 1, 1, bins)
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1) # (bs, n, n)

    chain_pair_iptm = {}
    chain_id_list = torch.unique(chain_id).tolist()
    for idx1 in chain_id_list:
        chain_iptm = {}
        for idx2 in chain_id_list:
            mask_pair_chain = (
                (chain_id[:, None, :] == idx1)
                * (chain_id[:, :, None] == idx2)
                * mask[:, None, :]
                * mask[:, :, None]
            )

            chain_iptm[idx2] = torch.max(
                torch.sum(predicted_tm_term * mask_pair_chain, dim=-1)
                / (torch.sum(mask_pair_chain, dim=-1) + 1e-5),
                dim=1,
            ).values
        chain_pair_iptm[idx1] = chain_iptm

    return chain_pair_iptm

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

