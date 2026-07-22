import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepContrastiveAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,   # as per your spec
        h1: int = 1024,
        h2: int = 512,
        h3: int = 256,
        temperature: float = 0.5,
        lambda_sup: float = 0.1
    ):
        super().__init__()

        self.temperature = temperature
        self.lambda_sup = lambda_sup

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, latent_dim),
        )

        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3), nn.ReLU(),
            nn.Linear(h3, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def supervised_contrastive_loss(self, z, labels):
        """
        z: [batch_size, latent_dim]
        labels: [batch_size]
        """
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T)  
        sim_matrix = sim_matrix / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()  # positives
        self_mask = torch.eye(mask.shape[0], device=mask.device)
        mask = mask - self_mask
        exp_sim = torch.exp(sim_matrix) * (1 - self_mask)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss


    def compute_loss(self, x, x_hat, z, labels):
       
        recon_loss = F.mse_loss(x_hat, x)
        sup_loss = self.supervised_contrastive_loss(z, labels)
        total_loss = recon_loss + self.lambda_sup * sup_loss

        return total_loss, recon_loss, sup_loss