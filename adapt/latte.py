import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.misc import load_templates_from_yaml, print_clip_parameters, print_optimizer_parameters
from utils.loralib import utils
from utils.loralib.layers import LoRALayer

REFERENCE_TEMPLATE = 'a histopathology slide showing {}'


class LATTE:
    """ 
    LATTE (Low-rank Adaptation with Transductive Template Ensembling) is a 
    test-time adaptation method designed for histopathology vision–language models (VLMs). 
    It combines lightweight low-rank adaptation (via LoRA or LayerNorm tuning) with 
    template-based ensembling to mitigate sensitivity to text prompts and improve 
    robustness under realistic distribution shifts (e.g., staining variation, 
    blur, contamination, noise, illumination). 

    LATTE is introduced in the paper:
    "Histopath-C: Towards Realistic Domain Shifts for Histopathology 
    Vision-Language Adaptation" (WACV 2026).

    This implementation is based on and extends ideas from the WATT repository:
    https://github.com/Mehrdad-Noori/WATT
    """

    def __init__(self, model, tokenizer, lr, classes, batch_size, steps=10, lora=False, lora_ln=False,
                 temp_dir='templates.yaml', use_laplacian=False,
                 # lora parameters 
                 lora_params=['q', 'k', 'v'], lora_rank=2, lora_alpha=1, lora_dropout=0.25, lora_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 avg_type = 'text_avg'
                 ):
        """
        Initialize the LATTE adaptation module.

        Args:
            model (torch.nn.Module): Pretrained vision language model (e.g., Quilt, CONCH).
            tokenizer: Tokenizer associated with the text encoder.
            lr (float): Learning rate for adaptation optimizer.
            classes (list[str]): List of class names for the downstream task.
            batch_size (int): Number of samples in each adaptation batch.
            steps (int, optional): Number of gradient steps during adaptation. Default: 10.
            lora (bool, optional): If True, apply LoRA adaptation on vision backbone. Default: False.
            lora_ln (bool, optional): If True, adapt both LoRA and LayerNorm parameters. Default: False.
            temp_dir (str, optional): Path to YAML file containing text templates. Default: 'templates.yaml'.
            use_laplacian (bool, optional): Whether to add global Laplacian regularization. Default: False.
            lora_params (list[str]): Parameters to apply LoRA on (e.g., ['q','k','v']). Default: ['q','k','v'].
            lora_rank (int): Rank for LoRA decomposition. Default: 2.
            lora_alpha (int): Scaling factor for LoRA. Default: 1.
            lora_dropout (float): Dropout probability for LoRA layers. Default: 0.25.
            lora_layers (list[int]): Indices of transformer layers to apply LoRA. Default: all layers [0–11].
            avg_type (str): Strategy for combining text embeddings 
                            ('text_avg' = average templates, 
                            'loss_avg' = average losses across templates, 
                            'reference' = single reference template). 
                            Default: 'text_avg'.
        """

        # loading the base model
        self.model = model
        self.tokenizer = tokenizer

        self.lr = lr
        self.type = type
        self.steps = steps
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size
        self.use_laplacian = use_laplacian
        self.avg_type = avg_type

        ## lora parameters
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.dropout_rate = lora_dropout
        self.lora_params = lora_params
        self.lora_layers = lora_layers
        
        # Load the text templates
        if temp_dir:
            self.all_templates = load_templates_from_yaml(temp_dir)
            print(f"The number of templates loaded: {len(self.all_templates)}")
            print("The average will be used during both adaptation and evaluation")
        else:
            self.all_templates = [REFERENCE_TEMPLATE]
            print(f"No templates loaded. Using the default template: {REFERENCE_TEMPLATE}")

        ## lora parameters
        assert not (lora and lora_ln), "LoRA and LoRA-LN cannot be used together"
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.dropout_rate = lora_dropout
        self.lora_params = lora_params
        self.lora_layers = lora_layers

        if lora:
            print("+++ Using LoRA")
            lora_dict = {'model': self.model.vit_backbone, 'position': 'all', 'encoder': 'vision', 'params': self.lora_params,
                         'r': self.rank,
                         'alpha': self.alpha, 'dropout_rate': self.dropout_rate,
                         'layers': self.lora_layers}
            _ = utils.apply_lora(lora_dict, self.model)
            self.model.to(self.device)
            # utils.load_lora(args, list_lora_layers)
            utils.mark_only_lora_as_trainable(self.model)
            params = utils.get_lora_parameters(self.model)
            print_clip_parameters(self.model)

        elif lora_ln:
            print("+++ Using LoRA + LN params")

            # Set the gradients for LayerNorm layers only for visual encoder
            if model.clip_type == "open_clip":
                self.model.transformer.requires_grad_(False)
                self.model.ln_final.requires_grad_(False)
                self.model.token_embedding.requires_grad_(False)

                # Collect the LayerNorm parameters and set the optimizer
                params_LN, _ = self.collect_ln_params(self.model.visual)

            elif model.clip_type == "conch":
                self.model.text.requires_grad_(False)
                self.model.text_decoder.requires_grad_(False)

                # Collect the LayerNorm parameters and set the optimizer
                params_LN, _ = self.collect_ln_params(self.model.visual)


            ## lora params
            lora_dict = {'model': self.model.vit_backbone, 'position': 'all', 'encoder': 'vision', 'params': self.lora_params,
                         'r': self.rank,
                         'alpha': self.alpha, 'dropout_rate': self.dropout_rate,
                         'layers': self.lora_layers}
            _ = utils.apply_lora(lora_dict, self.model)
            self.model.to(self.device)

            params_lora = utils.get_lora_parameters(self.model.visual)

            # set LN and LoRA as tranable params
            self.set_lora_and_ln_trainable(self.model.visual)

            # Set the optimizer
            params = params_LN + params_lora

            print_clip_parameters(self.model)


        else:
            print("+++ Using LN params")
            if model.clip_type == "open_clip":
                # Set the gradients for LayerNorm layers only for visual encoder
                self.model.transformer.requires_grad_(False)
                self.model.ln_final.requires_grad_(False)
                self.model.token_embedding.requires_grad_(False)

                # Set the gradients for LayerNorm layers only for visual encoder
                self.model.visual = self.set_ln_grads(self.model.visual)

                # Collect the LayerNorm parameters and set the optimizer
                params, _ = self.collect_ln_params(self.model.visual)

                # Set the optimizer
                print_clip_parameters(self.model)
        
            elif model.clip_type == "conch":
                self.model.text.requires_grad_(False)
                self.model.text_decoder.requires_grad_(False)

                # Set the gradients for LayerNorm layers only for visual encoder
                self.model.visual = self.set_ln_grads(self.model.visual)

                # Collect the LayerNorm parameters and set the optimizer
                params, _ = self.collect_ln_params(self.model.visual)



        # Set the optimizer
        print_clip_parameters(self.model)
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # Create assignment variables (Gus)
        if self.use_laplacian:
            self.K = len(self.all_templates)
            self.Z = torch.nn.Parameter(torch.randn(self.K).to(self.device), requires_grad=True)
            self.optimizer.add_param_group({"params": self.Z})

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        with torch.no_grad():
            self.extracted_text_features = self.extract_text_embeddings(classes, self.all_templates,
                                                                        average=True).squeeze()  # (class, 512)

    def adapt(self, x):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """

        self.reset()
        loss_report = self.perform_adaptation(x)
        return loss_report

    @torch.no_grad()
    def evaluate(self, x):
        """
        Forward pass without adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        Returns:
            pred: Predicted class labels for the input images.

        """

        # extracting features
        image_features = self.model.encode_image(x)

        # Pick the top most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if 'reference' in self.avg_type:
            text_features = self.extracted_text_features[0]
        else:
            text_features = self.extracted_text_features[-1]

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()

        return pred


    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)
        if self.use_laplacian:
            self.Z.data = torch.randn(self.batch_size, self.K).to(self.device)


    def perform_adaptation(self, x):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """

        loss_report = []
        for _ in range(self.steps):

            ### optimized version
            if 'loss_avg' in self.avg_type: # == 'loss_avg':
                text_x = self.extracted_text_features[:-1] # (T, B, feat) 

            elif 'text_avg' in self.avg_type: # == 'text_avg':
                text_x = self.extracted_text_features[-1:] # (T, B, feat) T is one here

            with torch.no_grad():
                image_features = self.model.encode_image(x) # (B, feat)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (B, feat)

                similarity_all = (100*torch.einsum('bf, tcf -> tbc', image_features, text_x)).softmax(-1) # ( T, B, class) = (B, feat) @ (T, class, feat)

                # get top 1 values and pred (so the resulting shape is (T, B, 1))
                values, pred = similarity_all.topk(1, -1, True, True) # (T, B, 1)


            # extract text_features from text_x using pred so the resulting shape is (T, B, feat)
            text_features_all = text_x.gather(1, pred.squeeze(-1).long().unsqueeze(-1).expand(-1, -1, text_x.size(-1)))

            # Calculating the Loss
            image_features = self.model.encode_image(x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)

            logits_all = self.model.logit_scale.exp() * torch.einsum('bf, tBf -> tbB', image_features, text_features_all) # (T, B, B)

            images_similarity2 = image_features @ image_features.t() # (B, B)
            texts_similarity_all = torch.bmm(text_features_all, text_features_all.transpose(1, 2)) # (T, B, B)

            targets_all = F.softmax(100 * ((images_similarity2 + texts_similarity_all) / 2), dim=-1) # (T, B, B)

            # loss
            loss = self.cross_entropy_all(logits_all, targets_all, reduction='none', dim=-1)

            #overall loss
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # loss_report.append(loss.item())
            loss_report.append(loss.item())

        return loss_report


    def extract_text_embeddings(self, class_names, templates, average=True):
        """
        Extracts text embeddings for given class names and templates.

        Args:
            class_names: List of class names to generate text embeddings for.
            templates: List of text templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.

        Returns:
            text_features: Tensor of text embeddings for the given class names and templates.
        """
        text_features = []
        for class_name in class_names:
            texts = [template.format(class_name) for template in templates]
            if self.model.clip_type == "conch":
                    texts = self.model.tokenize(texts=texts, tokenizer=self.tokenizer).to(self.device)
            else:
                texts = self.tokenizer(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)  # Shape: (#templates, 512)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)  # Shape: (512,)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                # add the averaged embeddings to the original embeddings
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features


    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.

        Args:
            model: The model whose LayerNorm layers' gradients are to be set.

        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model


    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.

        Args:
            model: The model from which to collect LayerNorm parameters.

        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names


    @staticmethod
    def set_lora_and_ln_trainable(model: nn.Module, bias: str = 'none') -> None:
        """
        Set gradient settings for both LayerNorm (LN) layers and LoRA parameters 
        while freezing all other parameters.

        Args:
            model: The model whose parameters are to be modified.
            bias (str): Specifies which bias parameters should be trainable.
                        Options: 'none', 'all', 'lora_only'.

        Returns:
            None
        """
        model.requires_grad_(False)  # Freeze all parameters

        # Enable gradients for LayerNorm layers
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for param in m.parameters():
                    param.requires_grad = True

        # Enable gradients for LoRA parameters
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True

        # Handle bias settings
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError(f"Bias setting '{bias}' is not implemented.")


    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        """
        Calculate the cross-entropy loss between predictions and targets.

        Args:
            preds: Predicted logits.
            targets: Target probabilities.
            reduction: Type of reduction to apply to the output ('none' or 'mean').

        Returns:
            The computed loss.
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


    @staticmethod
    def cross_entropy_all(preds, targets, reduction='none', dim=-1):
        """
        Calculate the cross-entropy loss between predictions and targets.

        Args:
            preds: Predicted logits.
            targets: Target probabilities.
            reduction: Type of reduction to apply to the output ('none' or 'mean').

        Returns:
            The computed loss.
        """
        log_softmax = nn.LogSoftmax(dim)
        loss = (-targets * log_softmax(preds)).sum(dim)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


    @staticmethod
    def weight_average(all_weights):
        """
        Compute the average of the weights from multiple models.

        Args:
            all_weights: List of state dictionaries from different models.

        Returns:
            avg_state_dict: Averaged state dictionary.
        """
        K = len(all_weights)
        avg_state_dict = OrderedDict()
        for param_name, param in all_weights[0].items():
            avg_param = sum(sd[param_name] for sd in all_weights) / K
            avg_state_dict[param_name] = avg_param
        return avg_state_dict

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.

        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.

        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.

        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    def global_laplacian(self, image_features):
        ZZ = self.Z @ self.Z
        W = image_features @ image_features.T

        laplacian = -torch.mean(W * ZZ)

        return laplacian