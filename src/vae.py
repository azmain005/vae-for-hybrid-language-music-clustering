"""
VAE Architectures - Base, Convolutional, Hybrid, Beta-VAE, and Conditional VAE
Optimized for GPU usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseVAE(nn.Module):
    """
    Standard Linear VAE for Easy Task
    """
    
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[512, 256]):
        super(BaseVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class ConvVAE(nn.Module):
    """
    Convolutional VAE for Medium Task (Spectrogram inputs)
    """
    
    def __init__(self, input_channels=1, latent_dim=128, img_height=128, img_width=128):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        
        # Encoder
        self.encoder = nn.Sequential(
            # Conv1: (batch, 1, 128, 128) -> (batch, 32, 64, 64)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Conv2: (batch, 32, 64, 64) -> (batch, 64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Conv3: (batch, 64, 32, 32) -> (batch, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Conv4: (batch, 128, 16, 16) -> (batch, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        self.flatten_size = 256 * (img_height // 16) * (img_width // 16)
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # DeConv1: (batch, 256, 8, 8) -> (batch, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # DeConv2: (batch, 128, 16, 16) -> (batch, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # DeConv3: (batch, 64, 32, 32) -> (batch, 32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # DeConv4: (batch, 32, 64, 64) -> (batch, 1, 128, 128)
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, self.img_height // 16, self.img_width // 16)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class HybridVAE(nn.Module):
    """
    Multi-modal VAE with separate encoders for Audio and Lyrics
    Concatenates latent vectors and uses separate decoders
    """
    
    def __init__(self, audio_dim, lyrics_dim, latent_dim=128, hidden_dim=256):
        super(HybridVAE, self).__init__()
        
        self.audio_dim = audio_dim
        self.lyrics_dim = lyrics_dim
        self.latent_dim = latent_dim
        
        # Audio Encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.audio_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.audio_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Lyrics Encoder
        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.lyrics_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.lyrics_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Combined latent is 2 * latent_dim
        combined_latent = 2 * latent_dim
        
        # Audio Decoder
        self.audio_decoder = nn.Sequential(
            nn.Linear(combined_latent, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, audio_dim)
        )
        
        # Lyrics Decoder
        self.lyrics_decoder = nn.Sequential(
            nn.Linear(combined_latent, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, lyrics_dim)
        )
    
    def encode(self, audio, lyrics):
        """Encode both modalities"""
        # Audio encoding
        h_audio = self.audio_encoder(audio)
        audio_mu = self.audio_mu(h_audio)
        audio_logvar = self.audio_logvar(h_audio)
        
        # Lyrics encoding
        h_lyrics = self.lyrics_encoder(lyrics)
        lyrics_mu = self.lyrics_mu(h_lyrics)
        lyrics_logvar = self.lyrics_logvar(h_lyrics)
        
        return audio_mu, audio_logvar, lyrics_mu, lyrics_logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode combined latent vector"""
        audio_recon = self.audio_decoder(z)
        lyrics_recon = self.lyrics_decoder(z)
        return audio_recon, lyrics_recon
    
    def forward(self, audio, lyrics):
        # Encode both modalities
        audio_mu, audio_logvar, lyrics_mu, lyrics_logvar = self.encode(audio, lyrics)
        
        # Reparameterize both
        z_audio = self.reparameterize(audio_mu, audio_logvar)
        z_lyrics = self.reparameterize(lyrics_mu, lyrics_logvar)
        
        # Concatenate latent vectors
        z = torch.cat([z_audio, z_lyrics], dim=1)
        
        # Decode
        audio_recon, lyrics_recon = self.decode(z)
        
        return (audio_recon, lyrics_recon), (audio_mu, audio_logvar, lyrics_mu, lyrics_logvar), z


class BetaVAE(BaseVAE):
    """
    Beta-VAE for disentanglement (Hard Task)
    Extends BaseVAE with beta parameter for KL weighting
    """
    
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[512, 256], beta=4.0):
        super(BetaVAE, self).__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta


class ConditionalVAE(nn.Module):
    """
    Conditional VAE (CVAE) conditioned on Q1-Q4 labels (Hard Task)
    """
    
    def __init__(self, input_dim, num_classes=4, latent_dim=128, hidden_dims=[512, 256]):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder (input + one-hot label)
        encoder_layers = []
        prev_dim = input_dim + num_classes
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (latent + one-hot label)
        decoder_layers = []
        prev_dim = latent_dim + num_classes
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x, labels):
        """Encode input conditioned on label"""
        # Convert labels to one-hot
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate input with label
        x_cond = torch.cat([x, labels_onehot], dim=1)
        
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        """Decode latent vector conditioned on label"""
        # Convert labels to one-hot
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate latent with label
        z_cond = torch.cat([z, labels_onehot], dim=1)
        
        return self.decoder(z_cond)
    
    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar, z


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction Loss + Beta * KL Divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence (1.0 for standard VAE, >1.0 for Beta-VAE)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def hybrid_vae_loss_function(audio_recon, lyrics_recon, audio, lyrics, 
                             audio_mu, audio_logvar, lyrics_mu, lyrics_logvar, 
                             beta=1.0):
    """
    Loss function for Hybrid VAE
    
    Returns:
        total_loss, audio_recon_loss, lyrics_recon_loss, audio_kl_loss, lyrics_kl_loss
    """
    # Audio reconstruction loss
    audio_recon_loss = F.mse_loss(audio_recon, audio, reduction='sum')
    
    # Lyrics reconstruction loss
    lyrics_recon_loss = F.mse_loss(lyrics_recon, lyrics, reduction='sum')
    
    # Audio KL divergence
    audio_kl_loss = -0.5 * torch.sum(1 + audio_logvar - audio_mu.pow(2) - audio_logvar.exp())
    
    # Lyrics KL divergence
    lyrics_kl_loss = -0.5 * torch.sum(1 + lyrics_logvar - lyrics_mu.pow(2) - lyrics_logvar.exp())
    
    # Total loss
    total_loss = audio_recon_loss + lyrics_recon_loss + beta * (audio_kl_loss + lyrics_kl_loss)
    
    return total_loss, audio_recon_loss, lyrics_recon_loss, audio_kl_loss, lyrics_kl_loss
