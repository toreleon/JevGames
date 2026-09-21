"""PyTorch device selection shared by model adapters."""

from __future__ import annotations


def resolve_torch_device(requested: str | None = "auto"):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("PyTorch is required by the selected model adapter") from exc

    choice = (requested or "auto").lower()
    if choice == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            choice = "mps"
        elif torch.cuda.is_available():
            choice = "cuda"
        else:
            choice = "cpu"
    if choice == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    elif choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    elif choice not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")
    return torch.device(choice)
