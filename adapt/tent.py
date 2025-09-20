import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import load_templates_from_yaml, print_clip_parameters, print_optimizer_parameters


REFERENCE_TEMPLATE = 'a histopathology slide showing {}'

class TENT:
    def __init__(self, model, tokenizer, lr, classes, temp_dir=None, steps=10,
                ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.steps = steps
        self.device = next(self.model.parameters()).device

        if model.clip_type == "open_clip":
            # Set the gradients for LayerNorm layers only for visual encoder
            self.model.transformer.requires_grad_(False)
            self.model.ln_final.requires_grad_(False)
            self.model.token_embedding.requires_grad_(False)

            # Set the gradients for LayerNorm layers only for visual encoder
            self.model.visual = self.set_ln_grads(self.model.visual)

            # Collect the LayerNorm parameters and set the optimizer
            params, _ = self.collect_ln_params(self.model.visual)
    
        elif model.clip_type == "conch":
            self.model.text.requires_grad_(False)
            self.model.text_decoder.requires_grad_(False)

            # Set the gradients for LayerNorm layers only for visual encoder
            self.model.visual = self.set_ln_grads(self.model.visual)

            # Collect the LayerNorm parameters and set the optimizer
            params, _ = self.collect_ln_params(self.model.visual)

            # Set the optimizer
            print_clip_parameters(self.model)

        else:
            raise NotImplementedError(f"CLIP type {model.clip_type} not supported for TENT")
        
        # set the optimizer
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)

        # print the parameters passed to the optimizer
        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        # Load the templates
        if temp_dir:
            all_templates = load_templates_from_yaml(temp_dir)
            print(f"The number of templates loaded: {len(all_templates)}")
            print("The average will be used during both adaptation and evaluation")
        else:
            all_templates = [REFERENCE_TEMPLATE]
            print(f"No templates loaded. Using the default template: {REFERENCE_TEMPLATE}")

        # extract text embeddings for the classes [since we freeze the text encoder, we can do this once]
        with torch.no_grad():
            self.extracted_text_features = self.extract_text_embeddings(classes,  all_templates, average=True).squeeze() # (class, 512)


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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Pick the top most similar labels for the image
        # using the pre-extracted text features
        text_features = self.extracted_text_features[-1] # using the average embeddings => if len(templates)==1 then there is no average and the last one represents the only template
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


    def perform_adaptation(self, x):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """
        text_features = self.extracted_text_features[-1] # using the average embeddings => if len(templates)==1 then there is no average and the last one represents the only template

        loss_report = []
        for _ in range(self.steps):
            # extracting features
            image_features = self.model.encode_image(x) 
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = self.model.logit_scale.exp() * image_features @ text_features.T


            # adapt
            loss = self.softmax_entropy(logits).mean(0)
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

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
    def softmax_entropy(x, dim=-1) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


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