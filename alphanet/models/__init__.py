__all__ = ["AlphaNet"]


def __getattr__(name):
    if name == "AlphaNet":
        from .alphanet import AlphaNet

        return AlphaNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
