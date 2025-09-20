import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.nn.Softmax
# from transformers import AutoProcessor, AutoModel

from utils.misc import print_clip_parameters, print_optimizer_parameters

REFERENCE_TEMPLATE = 'a histopathology slide showing {}'

class CLIPARTT:


    def __init__(self, model, tokenizer, classes, lr, K=3, steps=10,  
                 ):
        # loading the base model
        self.model = model
        self.tokenizer = tokenizer

        self.lr = lr
        self.type = type
        self.steps = steps
        self.device = next(self.model.parameters()).device
        self.K = K
        self.classes = classes


        
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

        else:
            raise NotImplementedError(f"CLIP type '{model.clip_type}' not implemented for CLIPArTT")


        # Set the optimizer
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)
        


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
        text_features = self.extract_text_embeddings(self.classes, [REFERENCE_TEMPLATE], average=True)
        text_features = text_features.T

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

        text_feat = self.extract_text_embeddings(self.classes, [REFERENCE_TEMPLATE], average=False).squeeze()
        loss_report = []
        for _ in range(self.steps):
            with torch.no_grad():
                image_features = self.model.encode_image(x)
            # adapt
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_feat.T).softmax(dim=-1)
            values, pred = similarity.topk(self.K, 1, True, True)
            # pred_inputs = torch.cat([self.tokenizer(self.getprompt(self.K, c, self.classes)) for c in pred]).to(self.device)

            if self.model.clip_type == "conch":
                    pred_inputs = torch.cat([self.model.tokenize(texts=[self.getprompt(self.K, c, self.classes)], tokenizer=self.tokenizer) for c in pred]).to(self.device)
            else:
                    # texts = self.tokenizer(texts).to(self.device)
                    pred_inputs = torch.cat([self.tokenizer(self.getprompt(self.K, c, self.classes)) for c in pred]).to(self.device)


            # Calculating the Loss
            # cosine similarity as logits
            image_features = self.model.encode_image(x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.model.encode_text(pred_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale_exp = self.model.logit_scale.exp()
            
             # cosine similarity as logits
            logits = logit_scale_exp * image_features @ text_features.T

            images_similarity = image_features @ image_features.t()
            texts_similarity = text_features @ text_features.t()
            targets = F.softmax( 
                    ((images_similarity + texts_similarity) / 2) / 0.01, dim=-1
                )
            loss = self.cross_entropy(logits, targets, reduction='mean')

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
        with torch.no_grad():
            text_features = []
            for class_name in class_names:
                texts = [template.format(class_name) for template in templates]
                if self.model.clip_type == "conch":
                    texts = self.model.tokenize(texts=texts, tokenizer=self.tokenizer).to(self.device)
                else:
                    texts = self.tokenizer(texts).to(self.device)
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if average:
                    class_embeddings = class_embeddings.mean(dim=0)
                    class_embeddings /= class_embeddings.norm()
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
    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    @staticmethod
    def getprompt(K, c, classes):
        for k in range(K):
            if k == 0:
                text_prompt = f"a histopathology slide showing " + classes[c[k]]
            else:
                text_prompt = text_prompt + " or " + classes[c[k]]
        return text_prompt


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

