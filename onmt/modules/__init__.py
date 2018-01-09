from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.ImageEncoder import ImageEncoder
from onmt.modules.MultiModules import *

# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, MultiWordEmbedding, MultiLinear, MultiCloneModule, MultiModule]
