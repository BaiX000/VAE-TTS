import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, in_channels=512, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_channels),
        ) 
        
    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result
        
    def forward(self, x):
        # x:(B, spk_dim)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), x, mu, log_var]
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def loss_function(self, recons, org_input, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = 0.001 # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, org_input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        #loss = recons_loss + kld_weight * kld_loss
        return [recons_loss, kld_loss]
    
class VSC(nn.Module):
    def __init__(self, in_channels=512, latent_dim=16):
        super(VSC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)
        self.fc_spike = nn.Linear(64, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_channels),
        ) 
        self.c = 50
        self.prior_weight = 0.0002
        
    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_spike = self.fc_spike(result)
        return [mu, log_var, -F.relu(-log_spike)]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result
        
    def forward(self, x):
        # x:(B, spk_dim)
        mu, log_var, log_spike = self.encode(x)
        z = self.reparameterize(mu, log_var, log_spike)
        return  [self.decode(z), x, mu, log_var, log_spike]
    
    def reparameterize(self, mu, logvar, logspike):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        gaussian = eps * std + mu
        
        eta = torch.randn_like(std)
        selection = torch.sigmoid(self.c*(eta + logspike.exp() - 1))
        return selection * gaussian
    
    def loss_function(self, recons, org_input, mu, log_var, log_spike):
     
        # hyper parameters
        alpha = 0.2        
        recons_loss =F.mse_loss(recons, org_input)

        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        spike = torch.clamp(log_spike.exp(), 1e-6, 1.0-1e-6)
        prior = torch.mean(
            -0.5 * torch.sum(spike * (1 + log_var - mu ** 2 - log_var.exp()), dim=1) + \
                        torch.sum((1-spike) * torch.log((1-spike)/(1-alpha)) + spike * torch.log(spike/alpha), dim=1)
        , dim=0)
        
        #loss = recons_loss + self.prior_weight*prior
        return [recons_loss, prior]
    def update_(self): 
        # c: (200-50) / 400000 = 0.000375
        # prior weight: (0.001-0.00025) / 400000 = 1.875e-9
        self.c += 0.0004
        #self.prior_weight += 1.875e-9