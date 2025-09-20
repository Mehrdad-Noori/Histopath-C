
import torch
import torch.nn as nn
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize as conch_tokenize
from conch.open_clip_custom import get_tokenizer as conch_get_tokenizer

# Wrapper to mimic timm Attention but use torch.nn.MultiheadAttention inside
class TorchMHAWrapper(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         bias=qkv_bias, batch_first=True)
        self.out_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # timm passes [B, N, C], torch MHA expects [B, N, C] if batch_first=True
        out, _ = self.mha(x, x, x, need_weights=False)
        return self.out_proj_drop(out)

def replace_timm_attn_with_torch(block):
    """Convert a single timm Attention block to torch.nn.MultiheadAttention."""
    dim = block.qkv.in_features
    num_heads = block.num_heads
    qkv_bias = block.qkv.bias is not None

    new_attn = TorchMHAWrapper(dim, num_heads, qkv_bias,
                               attn_drop=block.attn_drop.p,
                               proj_drop=block.proj_drop.p)

    # Copy weights
    with torch.no_grad():
        # qkv -> in_proj
        new_attn.mha.in_proj_weight.copy_(block.qkv.weight)
        new_attn.mha.in_proj_bias.copy_(block.qkv.bias)
        # proj -> out_proj
        new_attn.mha.out_proj.weight.copy_(block.proj.weight)
        new_attn.mha.out_proj.bias.copy_(block.proj.bias)

    return new_attn

def convert_visual_to_torch_attn(visual_model):
    """Replace all timm Attention layers in a VisionTransformer with torch MHA."""
    for i, blk in enumerate(visual_model.trunk.blocks):
        blk.attn = replace_timm_attn_with_torch(blk.attn)
    return visual_model

def test_attention_swap(model, image_size=224, batch_size=2, tol=1e-5):
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    # Forward before replacement
    with torch.no_grad():
        out_before = model.visual(dummy_input)
    # If tuple, take the first element
    if isinstance(out_before, tuple):
        out_before = out_before[0]

    # Replace attention
    model.visual = convert_visual_to_torch_attn(model.visual)

    # Forward after replacement
    with torch.no_grad():
        out_after = model.visual(dummy_input)
    if isinstance(out_after, tuple):
        out_after = out_after[0]

    # Compare
    max_diff = (out_before - out_after).abs().max().item()
    print(f"Output before shape: {out_before.shape}")
    print(f"Output after shape : {out_after.shape}")
    print(f"Max difference     : {max_diff:.6f}")

    if max_diff < tol:
        print("✅ Conversion successful: outputs match within tolerance.")
    else:
        print("⚠️ Warning: outputs differ more than tolerance!")

    return out_before, out_after

