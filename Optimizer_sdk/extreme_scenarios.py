import torch
import numpy as np

def generate_extreme_trace(mode="vanishing_gradient", length=100):
    if mode == "vanishing_gradient":
        return np.exp(-np.linspace(0, 5, length)) + np.random.normal(0, 0.01, size=length)

    elif mode == "nan_divergence":
        base = np.linspace(1, 20, length)
        base[length//2:] += np.linspace(0, 10, length//2) ** 2
        return base + np.random.normal(0, 0.5, size=length)

    elif mode == "chaotic_gradient":
        t = np.linspace(0, 4*np.pi, length)
        return np.sin(t) * np.cos(5*t) + np.random.normal(0, 0.1, size=length)

    elif mode == "adversarial_spike":
        stable = np.exp(-np.linspace(0, 2, length//2))
        spike = np.exp(np.linspace(0, 4, length//2))
        return np.concatenate([stable, spike]) + np.random.normal(0, 0.1, size=length)

    elif mode == "staircase_explosion":
        return np.concatenate([
            np.linspace(1.0, 0.7, length//4),
            np.ones(length//4) * 0.7,
            np.linspace(0.7, 2.0, length//2)
        ]) + np.random.normal(0, 0.05, size=length)

    elif mode == "multi_modal_noise":
        t = np.linspace(0, 8*np.pi, length)
        return 0.5*np.sin(t) + 0.3*np.sin(3*t + 1.5) + 0.2*np.random.normal(0, 0.2, size=length)

    # 🔥 新增模式：plateau_burst
    elif mode == "plateau_burst":
        plateau = np.ones(length // 2) * 0.5
        burst = np.exp(np.linspace(0, 3, length // 2)) + np.random.normal(0, 0.2, length // 2)
        return np.concatenate([plateau, burst]) + np.random.normal(0, 0.05, size=length)

    # 🔥 新增模式：entropy_pulse
    elif mode == "entropy_pulse":
        base = np.exp(-np.linspace(0, 4, length))
        pulse_positions = np.random.choice(length, size=5, replace=False)
        base[pulse_positions] += np.random.normal(5, 2, size=5)
        return base + np.random.normal(0, 0.05, size=length)

    else:
        raise ValueError("Unsupported trace mode: " + mode)

def generate_batch_traces(mode, batch=16, length=100):
    traces = [generate_extreme_trace(mode, length) for _ in range(batch)]
    return torch.tensor(np.array(traces), dtype=torch.float32)
