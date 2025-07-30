import torch

def calculate_l1_snr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_invariant: bool = False,
    take_log: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculates the L1 Signal-to-Noise Ratio (SNR) between prediction and target.

    Args:
        prediction (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        scale_invariant (bool): If True, apply scale-invariant projection.
        take_log (bool): If True, apply 10 * log10 to the ratio.
        eps (float): Small epsilon for numerical stability in division.

    Returns:
        torch.Tensor: The calculated L1 SNR.
    """
    # Flatten the tensors for calculation
    est_target_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    # Apply scale-invariant projection if enabled
    if scale_invariant:
        dot = torch.sum(est_target_flat * target_flat, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target_flat * target_flat, dim=-1, keepdim=True)
        target_scaler = (dot + eps) / (s_target_energy + eps)
        target_flat = target_flat * target_scaler
 
    # Calculate error and target energy based on L1 norm
    e_error = torch.abs(est_target_flat - target_flat).mean(dim=-1)
    e_target = torch.abs(target_flat).mean(dim=-1)

    # Calculate loss ratio
    if take_log:
        loss = 10 * (torch.log10(e_error + eps) - torch.log10(e_target + eps))
    else:
        loss = (e_error + eps) / (e_target + eps)

    return torch.mean(loss)