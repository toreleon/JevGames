"""Hardware-aware device resolution for the PyTorch Laya implementation."""

from __future__ import annotations


def resolve_device(requested: str | None = "auto"):
    """Return a usable PyTorch device, preferring Apple Metal on this project.

    Laya is implemented with PyTorch/Transformers, so Apple Silicon should use
    PyTorch's ``mps`` backend—not CUDA and not an unrelated MLX rewrite.  CUDA
    remains supported for portability when a caller explicitly asks for it.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("PyTorch is required. Install the optional dependencies: pip install -e '.[laya]'") from exc

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
            raise RuntimeError(
                "MPS is unavailable. Install a current macOS PyTorch build and check "
                "torch.backends.mps.is_available()."
            )
    elif choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    elif choice != "cpu" and choice not in {"mps", "cuda"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")
    return torch.device(choice)
