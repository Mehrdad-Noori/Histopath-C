import torch

from utils.misc import load_templates_from_yaml

REFERENCE_TEMPLATE = 'a histopathology slide showing {}'


class LAME:
    def __init__(self, model, tokenizer, classes, affinity, temp_dir=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device
        self.affinity = affinity


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
            self.extracted_text_features = self.extract_text_embeddings(classes, all_templates, average=True).squeeze()  # (class, 512)


    def adapt(self, x):
        """
        Forward pass with adaptation.

        Args:
            x: Input image tensor.
            classes: List of class names.

        """
        logits = self.perform_adaptation(x)
        values, pred = logits.topk(1, 1, True, True)
        pred = pred.t()
        return pred

    def reset(self):
        pass

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
        text_features = self.extracted_text_features[-1]  # using the average embeddings => if len(templates)==1 then there is no average and the last one represents the only template
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(1, 1, True, True)
        pred = pred.t()

        return pred

    def perform_adaptation(self, x):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x: Input image tensor.
            classes: List of class names.
        """
        text_features = self.extracted_text_features[-1]  # using the average embeddings => if len(templates)==1 then there is no average and the last one represents the only template
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * image_features @ text_features.T
        probas = torch.softmax(logits, dim=1)
        unary = -torch.log(probas + 1e-10)
        if self.affinity == 'knn':
            kernel = kNN_affinity(knn=5)(image_features)
        elif self.affinity == 'rbf':
            kernel = rbf_affinity(knn=5)(image_features)
        else:
            kernel = linear_affinity()(image_features)
        logits = self.laplacian_optimization(unary.type(torch.float32), kernel.type(torch.float32))

        return logits

    def laplacian_optimization(self, unary, kernel, bound_lambda = 1, max_steps = 100):
        E_list = []
        oldE = float('inf')
        Y = (-unary).softmax(-1)  # [N, K]
        for i in range(max_steps):
            pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
            exponent = -unary + pairwise
            Y = exponent.softmax(-1)
            E = self.entropy_energy(Y, unary, pairwise, bound_lambda).item()
            E_list.append(E)

            if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
                # logger.info(f'Converged in {i} iterations')
                break
            else:
                oldE = E

        return Y

    def entropy_energy(self, Y, unary, pairwise, bound_lambda,):
        E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()

        return E

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


class AffinityMatrix:
    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]

        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, **kwargs):
        self.k = kwargs['knn']

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))

        return rbf


class linear_affinity(AffinityMatrix):
    def __call__(self, X):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())