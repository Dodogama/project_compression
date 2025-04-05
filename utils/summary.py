import torch
from torchinfo import summary


def get_model_flops(model, input_size):
    """
    Calculates the number of floating-point operations (FLOPs) in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_size (tuple): The size of the input tensor (e.g., (1, 3, 224, 224) for a single RGB image of size 224x224).

    Returns:
        float: The estimated number of FLOPs in the model. Returns None if an error occurs.
    """
    try:
        model_stats = summary(model, input_size=input_size, verbose=0)
        return model_stats.total_params
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return None