"""
Deep Learning Models for Spam Detection.
Provides Deep Multi-Layer Perceptron (MLP) architectures and sequential models,
optimized for sparse BoW and TF-IDF vector inputs.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DeepMLPScikit:
    """
    Deep Feedforward Multi-Layer Perceptron using scikit-learn backend.
    Architecture: Input(5000) -> Dense(256) -> ReLU -> Dropout/L2 -> Dense(128) -> ReLU -> Dense(64) -> Output(2)
    """

    def __init__(
        self,
        hidden_layer_sizes=(256, 128, 64),
        alpha: float = 1e-4,
        max_iter: int = 40,
        random_state: int = 42
    ):
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=random_state
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


if HAS_TORCH:
    class PyTorchMLP(nn.Module):
        """
        PyTorch Deep Neural Network with BatchNorm, ReLU, Dropout, and Sigmoid output.
        """
        def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout_rate: float = 0.3):
            super().__init__()
            layers = []
            curr_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                curr_dim = h_dim
            layers.append(nn.Linear(curr_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


    class PyTorchDeepClassifier:
        """
        Scikit-learn compatible wrapper around PyTorchMLP for BoW and TF-IDF features.
        """
        def __init__(
            self,
            hidden_dims=(256, 128, 64),
            lr: float = 0.001,
            epochs: int = 15,
            batch_size: int = 64,
            dropout: float = 0.3,
            random_state: int = 42
        ):
            self.hidden_dims = hidden_dims
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.random_state = random_state
            self.model: Optional[PyTorchMLP] = None
            self.classes_ = np.array([0, 1])

        def _to_tensor(self, X):
            if hasattr(X, "toarray"):
                X = X.toarray()
            return torch.tensor(X, dtype=torch.float32)

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            X_t = self._to_tensor(X)
            y_t = torch.tensor(np.array(y), dtype=torch.float32)

            input_dim = X_t.shape[1]
            self.model = PyTorchMLP(input_dim, self.hidden_dims, self.dropout)
            self.model.train()

            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = self.model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()

            return self

        def predict_proba(self, X):
            self.model.eval()
            X_t = self._to_tensor(X)
            with torch.no_grad():
                logits = self.model(X_t)
                probs = torch.sigmoid(logits).numpy()
            probs_2d = np.column_stack([1 - probs, probs])
            return probs_2d

        def predict(self, X):
            probs = self.predict_proba(X)[:, 1]
            return (probs >= 0.5).astype(int)


def get_deep_learning_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns deep learning models for text classification.
    """
    models = {
        "Deep MLP (Dense Neural Net)": DeepMLPScikit(
            hidden_layer_sizes=(256, 128, 64),
            random_state=random_state
        )
    }
    if HAS_TORCH:
        models["PyTorch Deep Neural Net"] = PyTorchDeepClassifier(
            hidden_dims=(256, 128, 64),
            epochs=15,
            random_state=random_state
        )
    return models

