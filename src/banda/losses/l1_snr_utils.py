import torch
import structlog

logger = structlog.get_logger(__name__)

def calculate_l1_snr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_invariant: bool = False,
    take_log: bool = True,
    eps: float = 1e-3,
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
    if torch.isnan(est_target_flat).any():
        logger.error("L1_SNR: NaN detected in est_target_flat", mean_val=est_target_flat.mean().item())
        raise ValueError("NaN in est_target_flat")
    if torch.isnan(target_flat).any():
        logger.error("L1_SNR: NaN detected in target_flat", mean_val=target_flat.mean().item())
        raise ValueError("NaN in target_flat")
    logger.debug("L1_SNR: est_target_flat and target_flat stats", est_target_flat_mean=est_target_flat.mean().item(), target_flat_mean=target_flat.mean().item())

    # Apply scale-invariant projection if enabled
    if scale_invariant:
        dot = torch.sum(est_target_flat * target_flat, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target_flat * target_flat, dim=-1, keepdim=True)
        logger.debug("L1_SNR: Before target_scaler calculation", dot=dot.mean().item(), s_target_energy=s_target_energy.mean().item(), eps=eps)
        target_scaler = (dot + eps) / (s_target_energy + eps)
        if torch.isnan(target_scaler).any():
            logger.error("L1_SNR: NaN detected in target_scaler", dot=dot.mean().item(), s_target_energy=s_target_energy.mean().item(), eps=eps)
            raise ValueError("NaN in target_scaler")
        target_flat = target_flat * target_scaler
 
    # Calculate error and target energy based on L1 norm
    e_error = torch.abs(est_target_flat - target_flat).mean(dim=-1)
    e_target = torch.abs(target_flat).mean(dim=-1)

    # Calculate loss ratio
    if take_log:
        logger.debug("L1_SNR: Before log10 calculation", e_error=e_error.mean().item(), e_target=e_target.mean().item(), eps=eps)
        loss = 10 * (torch.log10(e_error + eps) - torch.log10(e_target + eps))
        if torch.isnan(loss).any():
            logger.error("L1_SNR: NaN detected in log10 loss", e_error=e_error.mean().item(), e_target=e_target.mean().item(), eps=eps)
            raise ValueError("NaN in log10 loss")
    else:
        logger.debug("L1_SNR: Before division loss calculation", e_error=e_error.mean().item(), e_target=e_target.mean().item(), eps=eps)
        loss = (e_error + eps) / (e_target + eps)
        if torch.isnan(loss).any():
            logger.error("L1_SNR: NaN detected in division loss", e_error=e_error.mean().item(), e_target=e_target.mean().item(), eps=eps)
            raise ValueError("NaN in division loss")

    return torch.mean(loss)