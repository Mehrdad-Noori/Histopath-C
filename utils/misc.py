import os
import sys
import yaml
import torch
import torch.nn as nn
import random
import numpy as np

import open_clip

from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize as conch_tokenize
from conch.open_clip_custom import get_tokenizer as conch_get_tokenizer


def load_clip_model(clip_type, backbone, device):
    if clip_type == 'open_clip':  
        if backbone=="hf-hub:wisdomik/QuiltNet-B-32":
            model, _, original_transforms = open_clip.create_model_and_transforms(backbone)
            tokenizer = open_clip.get_tokenizer(backbone)
            model.vit_backbone = 'ViT-B-32'  # to keep track of the model type in the results file
            print("Using QuiltNet-B-32 model from HuggingFace Hub")
        
        elif "pathgenclip.pt" in backbone:
            # check if the args.backbone exists as a file
            if not os.path.isfile(backbone):
                raise ValueError(f"PathGen-CLIP model not found at {backbone}. Please provide a valid path.")
            model, _, original_transforms = open_clip.create_model_and_transforms('ViT-B-16', pretrained=backbone)
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.vit_backbone = 'ViT-B-16'  # to keep track of the model type in the results file
            print("Using PathGen-CLIP model")

        else:
            raise ValueError(f"Not supported backbone: {backbone} => only 'hf-hub:wisdomik/QuiltNet-B-32' and 'pathgen' are supported for now")

        model.clip_type = clip_type  # to keep track of the CLIP type in the results file

    elif clip_type == 'conch': 
        if "conch.bin" in backbone:     
                model_cfg = 'conch_ViT-B-16'
                model, original_transforms = create_model_from_pretrained(model_cfg, backbone, device=device)
                tokenizer = conch_get_tokenizer()
                model.vit_backbone = model_cfg
                model.tokenize = conch_tokenize #TODO: it is a quick fix, better to change in future
                model.text_decoder = nn.Identity() # to avoid using the text decoder

                # Replace attention (make it much more efficient)
                model.visual = convert_visual_timm_to_torch_attn(model.visual)
                print("Conch loaded and attention layers replaced ✅")

                print("\n+++Using Conch model")
        else:
            raise ValueError(f"Not supported backbone: {backbone} => only '*conch.bin' is supported for now")

        model.clip_type = 'conch'  

    else: 
        raise ValueError(f"Not supported CLIP type: {clip_type} => for 'import clip' it should be very easy to implment")
    

    model.to(device)

    return model, tokenizer, original_transforms
    



def print_clip_parameters(model):
    """
    Print the total and learnable parameters (requires_grad=True) for each module in a CLIP model
    and the overall summary.

    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    # Define the modules to analyze
    if model.clip_type == 'open_clip':
        modules = {
            "model.visual": model.visual,
            "model.transformer": model.transformer,
            "model.ln_final": model.ln_final,
            "model.token_embedding": model.token_embedding
        }
    elif model.clip_type == 'conch':
        modules = {
            "model.visual": model.visual,
            "model.text": model.text,
            "model.text_decoder": model.text_decoder,
        }
    
    print("\nModel Parameters Summary +++++++++++++++++++")
    
    total_params = 0
    learnable_params = 0

    # Print parameters for each module
    for name, module in modules.items():
        module_total = sum(p.numel() for p in module.parameters())
        module_learnable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += module_total
        learnable_params += module_learnable
        print(f"{name:25}: Total = {module_total:,}, Learnable = {module_learnable:,}")

    # Print overall summary
    print("----------------------------------------")
    print(f"Total Parameters      : {total_params:,}")
    print(f"Learnable Parameters  : {learnable_params:,}")
    print("----------------------------------------")


def print_optimizer_parameters(optimizer, model):
    """
    Print the total and learnable parameters passed to the optimizer,
    grouped by each module of the CLIP model.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        model (torch.nn.Module): The PyTorch model.
    """
    # Define the modules to analyze
    if model.clip_type == 'open_clip':
        modules = {
            "model.visual": model.visual,
            "model.transformer": model.transformer,
            "model.ln_final": model.ln_final,
            "model.token_embedding": model.token_embedding
        }
    elif model.clip_type == 'conch':
        modules = {
            "model.visual": model.visual,
            "model.text": model.text,
            "model.text_decoder": model.text_decoder,
        }
        
    # Collect parameters in the optimizer
    optimizer_params = {id(p): p for group in optimizer.param_groups for p in group['params']}
    
    print("\nOptimizer Parameters by Module +++++++++++++")
    total_optimizer_params = 0

    # Count parameters for each module
    for name, module in modules.items():
        module_params = sum(
            p.numel() for p in module.parameters() if id(p) in optimizer_params
        )
        total_optimizer_params += module_params
        print(f"{name:25}: Parameters in Optimizer = {module_params:,}")

    # Print the total
    print("---------------------------------------------")
    print(f"Total Parameters in Optimizer : {total_optimizer_params:,}")


def set_global_seeds(seed_value=42):
    """Set random seeds for reproducibility across various libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_configuration(args):
    """Save configuration parameters to a file."""
    os.makedirs(args.save_dir, exist_ok=True)
    config_filepath = os.path.join(args.save_dir, 'configurations.txt')
    print("---"*10)
    print("configurations:")
    with open(config_filepath, 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
            print(f"       {arg}: {getattr(args, arg)}")

    print("---"*10)


def load_templates_from_yaml(file_path='templates.yaml'):
    """Load text templates from a YAML file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['templates'] 


def save_checkpoint(state, is_best, args):
    torch.save(state, args.save + args.dataset + '_' + args.model + '.pth')
    if is_best:
            torch.save(state, args.save + args.dataset + '_' + args.model + '_torch_best.pth')


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# Wrapper to mimic timm Attention but use torch.nn.MultiheadAttention inside
class TorchMHAWrapper(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                         bias=qkv_bias, batch_first=True)
        self.out_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
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


def convert_visual_timm_to_torch_attn(visual_model):
    """Replace all timm Attention layers in a VisionTransformer with torch MHA."""
    for i, blk in enumerate(visual_model.trunk.blocks):
        blk.attn = replace_timm_attn_with_torch(blk.attn)
    return visual_model