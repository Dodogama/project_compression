import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis

    
def get_model_stats(model, input_size) -> dict:
    """
    Calculates the number of floating-point operations (FLOPs) in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_size (tuple): The size of the input tensor (e.g., (3, 224, 224) for a single RGB image of size 224x224).

    Returns:
        dict: A dictionary containing model statistics
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        with torch.no_grad():
            sample = torch.randn(*input_size).to(device)
            flops = FlopCountAnalysis(model, sample)
            summary_info = summary(model, input_size=input_size, device=device)
            stats = {
                'flops': flops.total(),
                'params': summary_info.total_params,
            }
        return stats
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None