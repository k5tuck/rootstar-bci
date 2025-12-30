"""PyTorch models for neural fingerprint encoding and decoding.

This module provides deep learning models for:
- Encoding multimodal neural patterns into compact embeddings
- Contrastive learning for fingerprint similarity
- Inverse mapping from fingerprints to stimulation parameters
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralFingerprintEncoder(nn.Module):
    """Deep neural network for encoding neural patterns into fingerprint embeddings.

    Architecture: Multi-branch network processing EEG and fNIRS separately
    before fusion into a unified embedding space.

    Args:
        n_eeg_channels: Number of EEG channels (default: 128)
        n_fnirs_channels: Number of fNIRS channels (default: 64)
        n_bands: Number of frequency bands (default: 10)
        embedding_dim: Dimension of output embedding (default: 256)
        n_classes: Number of sensory classes for classification (default: 100)
    """

    def __init__(
        self,
        n_eeg_channels: int = 128,
        n_fnirs_channels: int = 64,
        n_bands: int = 10,
        embedding_dim: int = 256,
        n_classes: int = 100,
    ):
        super().__init__()

        self.n_eeg_channels = n_eeg_channels
        self.n_fnirs_channels = n_fnirs_channels
        self.n_bands = n_bands
        self.embedding_dim = embedding_dim

        # EEG Spectral Branch - Processes band power features
        self.eeg_spectral = nn.Sequential(
            nn.Linear(n_eeg_channels * n_bands, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # EEG Connectivity Branch - Processes coherence matrices
        # Input: (batch, n_bands, n_channels, n_channels)
        self.eeg_connectivity = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
        )

        # fNIRS Branch - Processes hemodynamic patterns
        # Input: (batch, n_fnirs_channels * 2) for HbO + HbR
        self.fnirs_encoder = nn.Sequential(
            nn.Linear(n_fnirs_channels * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # Fusion Network
        # Combines: EEG spectral (256) + EEG connectivity (256) + fNIRS (128) = 640
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embedding_dim),
        )

        # Classification Head (for supervised training)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(
        self,
        eeg_power: torch.Tensor,
        eeg_coherence: torch.Tensor,
        fnirs_activation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through fingerprint encoder.

        Args:
            eeg_power: Band power features (batch, n_channels, n_bands)
            eeg_coherence: Coherence matrices (batch, n_bands, n_channels, n_channels)
            fnirs_activation: fNIRS activation (batch, n_channels, 2) for HbO and HbR

        Returns:
            Dictionary with:
                - embedding: L2-normalized embedding vector
                - logits: Class logits for supervised training
                - eeg_features: Intermediate EEG features
                - fnirs_features: Intermediate fNIRS features
        """
        batch_size = eeg_power.size(0)

        # Process EEG spectral features
        eeg_flat = eeg_power.view(batch_size, -1)
        eeg_spectral_feat = self.eeg_spectral(eeg_flat)

        # Process EEG connectivity
        eeg_conn_feat = self.eeg_connectivity(eeg_coherence)

        # Process fNIRS
        fnirs_flat = fnirs_activation.view(batch_size, -1)
        fnirs_feat = self.fnirs_encoder(fnirs_flat)

        # Fuse modalities
        combined = torch.cat([eeg_spectral_feat, eeg_conn_feat, fnirs_feat], dim=-1)
        embedding = self.fusion(combined)

        # Normalize embedding for cosine similarity
        embedding_norm = F.normalize(embedding, p=2, dim=-1)

        # Classification (for supervised training)
        logits = self.classifier(embedding)

        return {
            "embedding": embedding_norm,
            "logits": logits,
            "eeg_features": eeg_spectral_feat,
            "fnirs_features": fnirs_feat,
        }

    def encode(
        self,
        eeg_power: torch.Tensor,
        eeg_coherence: torch.Tensor,
        fnirs_activation: torch.Tensor,
    ) -> torch.Tensor:
        """Encode neural data to embedding vector only.

        Convenience method for inference without classification.
        """
        with torch.no_grad():
            result = self.forward(eeg_power, eeg_coherence, fnirs_activation)
            return result["embedding"]


class ContrastiveFingerprintLoss(nn.Module):
    """Contrastive loss for learning fingerprint embeddings.

    Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    to bring embeddings of the same sensory experience closer together
    while pushing different experiences apart.

    Args:
        temperature: Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent contrastive loss.

        Args:
            embeddings: L2-normalized embedding vectors (batch, embedding_dim)
            labels: Sensory class labels (batch,)

        Returns:
            Scalar loss value
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive pair mask (same label)
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        positive_mask.fill_diagonal_(0)  # Exclude self-similarity

        # Mask out self-similarity for denominator
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Compute log softmax
        log_prob = F.log_softmax(sim_matrix, dim=-1)

        # Compute loss over positive pairs
        positive_log_prob = (positive_mask * log_prob).sum(dim=-1)
        num_positives = positive_mask.sum(dim=-1)

        # Average over positives (avoid div by zero)
        num_positives = torch.clamp(num_positives, min=1e-8)
        loss = -positive_log_prob / num_positives

        return loss.mean()


class InverseStimulationModel(nn.Module):
    """Neural network for mapping brain state changes to stimulation parameters.

    Learns the inverse mapping from desired neural activation patterns
    to the stimulation parameters needed to achieve them.

    Args:
        n_electrodes: Number of stimulation electrodes (default: 128)
        hidden_dim: Hidden layer dimension (default: 256)
    """

    def __init__(
        self,
        n_electrodes: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.n_electrodes = n_electrodes

        # Encoder for target activation delta
        self.encoder = nn.Sequential(
            nn.Linear(n_electrodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Output heads for different stimulation parameters
        self.amplitude_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output 0-1, scale to 0-2000 µA
        )

        self.frequency_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Output 0-1, scale to 0-100 Hz
        )

        self.electrode_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_electrodes),
            nn.Tanh(),  # Output -1 to 1 for anode/cathode selection
        )

    def forward(self, target_delta: torch.Tensor) -> torch.Tensor:
        """Compute stimulation parameters from target activation delta.

        Args:
            target_delta: Difference between target and current activation
                         (batch, n_electrodes)

        Returns:
            Concatenated parameter tensor:
                [amplitude (1), frequency (1), electrode_weights (n_electrodes)]
        """
        features = self.encoder(target_delta)

        amplitude = self.amplitude_head(features)
        frequency = self.frequency_head(features)
        electrodes = self.electrode_head(features)

        return torch.cat([amplitude, frequency, electrodes], dim=-1)

    def decode_parameters(
        self,
        params: torch.Tensor,
        electrode_names: Optional[list] = None,
    ) -> Dict:
        """Decode raw parameter tensor to interpretable stimulation protocol.

        Args:
            params: Raw parameter tensor from forward pass
            electrode_names: Optional list of electrode names

        Returns:
            Dictionary with decoded stimulation parameters
        """
        params = params.detach().cpu().numpy()

        amplitude_ua = float(params[0] * 2000)  # Scale to 0-2000 µA
        frequency_hz = float(params[1] * 100)  # Scale to 0-100 Hz
        electrode_weights = params[2:]

        # Select anodes (positive weights) and cathodes (negative weights)
        anode_mask = electrode_weights > 0.5
        cathode_mask = electrode_weights < -0.5

        if electrode_names is None:
            electrode_names = [f"E{i}" for i in range(self.n_electrodes)]

        anodes = [electrode_names[i] for i in range(len(anode_mask)) if anode_mask[i]]
        cathodes = [electrode_names[i] for i in range(len(cathode_mask)) if cathode_mask[i]]

        # Default electrodes if none selected
        if not anodes:
            anodes = ["Cz"]
        if not cathodes:
            cathodes = ["Fp1", "Fp2"]

        return {
            "amplitude_ua": amplitude_ua,
            "frequency_hz": frequency_hz,
            "anodes": anodes[:4],  # Limit to 4 electrodes each
            "cathodes": cathodes[:4],
            "electrode_weights": electrode_weights,
        }


class FingerprintDiscriminator(nn.Module):
    """Discriminator for adversarial training of fingerprint generation.

    Used to improve the quality of generated fingerprints by distinguishing
    real captured fingerprints from synthesized ones.

    Args:
        embedding_dim: Dimension of fingerprint embedding (default: 256)
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify embedding as real (1) or fake (0)."""
        return self.network(embedding)


def create_encoder_for_config(
    eeg_channels: int,
    fnirs_channels: int,
    n_bands: int = 10,
    embedding_dim: int = 256,
) -> NeuralFingerprintEncoder:
    """Factory function to create encoder matching hardware configuration.

    Args:
        eeg_channels: Number of EEG channels (8, 32, 64, 128, or 256)
        fnirs_channels: Number of fNIRS channels
        n_bands: Number of frequency bands
        embedding_dim: Output embedding dimension

    Returns:
        Configured NeuralFingerprintEncoder instance
    """
    return NeuralFingerprintEncoder(
        n_eeg_channels=eeg_channels,
        n_fnirs_channels=fnirs_channels,
        n_bands=n_bands,
        embedding_dim=embedding_dim,
    )


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing NeuralFingerprintEncoder...")

    encoder = NeuralFingerprintEncoder(
        n_eeg_channels=8,
        n_fnirs_channels=4,
        n_bands=10,
        embedding_dim=256,
    )

    # Create dummy inputs
    batch_size = 4
    eeg_power = torch.randn(batch_size, 8, 10)
    eeg_coherence = torch.randn(batch_size, 10, 8, 8)
    fnirs_activation = torch.randn(batch_size, 4, 2)

    # Forward pass
    output = encoder(eeg_power, eeg_coherence, fnirs_activation)

    print(f"Embedding shape: {output['embedding'].shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Embedding norm: {output['embedding'].norm(dim=-1)}")

    # Test contrastive loss
    print("\nTesting ContrastiveFingerprintLoss...")
    loss_fn = ContrastiveFingerprintLoss()
    labels = torch.tensor([0, 0, 1, 1])  # Two pairs of same class
    loss = loss_fn(output["embedding"], labels)
    print(f"Contrastive loss: {loss.item():.4f}")

    # Test inverse model
    print("\nTesting InverseStimulationModel...")
    inverse = InverseStimulationModel(n_electrodes=8)
    target_delta = torch.randn(batch_size, 8)
    params = inverse(target_delta)
    print(f"Params shape: {params.shape}")

    decoded = inverse.decode_parameters(params[0])
    print(f"Decoded: {decoded}")

    print("\nAll tests passed!")
