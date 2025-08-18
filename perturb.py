import torch

def apply_perturbation(waveform: torch.Tensor, strength: float) -> torch.Tensor:
    """
    对输入的音频波形应用扰动。

    :param waveform: 输入的音频波形张量，形状为 (通道数, 采样点数)
    :param strength: 扰动强度
    :return: 经过扰动后的音频波形张量
    """
    
    # --- TODO: 在这里实现您自己设计的“不可学习”算法 ---
    # 下面是一个简单的高斯噪声示例，请替换为您自己的逻辑

    if strength > 0:
        noise = torch.randn_like(waveform) * strength
        perturbed_waveform = waveform + noise
        return perturbed_waveform
    else:
        return waveform
    # ---------------------------------------------------------