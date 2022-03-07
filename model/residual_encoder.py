import torch
import torch.nn as nn


class ResidualEncoder(nn.Module) :
    '''
    Neural network that can be used to parametrize q(z_{l}|x) and q(z_{o}|x)
    '''
    def __init__(self, n_mel_channels, decoder_hidden, residual_encoding_dim, log_min_std_dev=-1):
        super(ResidualEncoder, self).__init__()
        self.conv1 = nn.Conv1d(n_mel_channels, 512, 3, 1)
        self.bi_lstm = nn.LSTM(512, 256, 2, bidirectional = True, batch_first=True)
        self.linear = nn.Linear(512, residual_encoding_dim)
        self.residual_encoding_dim = int(residual_encoding_dim/2)
        self.register_buffer('min_std_dev', torch.exp(torch.tensor([log_min_std_dev]).float()) )
        
    def forward(self, x):
        '''
        x.shape = [batch_size, seq_len, n_mel_channels]
        returns single sample from the distribution q(z_{l}|X) or q(z_{o}|X) of size [batch_size, 16]
        '''
        x = self.conv1(x.transpose(2,1)).transpose(2,1)
        output, (_,_) = self.bi_lstm(x)
        seq_len = output.shape[1]
        output = output.sum(dim=1)/seq_len
        x = self.linear(output)
        mean, log_variance = x[:,:self.residual_encoding_dim], x[:,self.residual_encoding_dim:]
        std_dev = torch.sqrt(torch.exp(log_variance))
        #return  torch.distributions.normal.Normal(mean,torch.max(std_dev, self.min_std_dev))
        return torch.normal(mean, torch.max(std_dev, self.min_std_dev))