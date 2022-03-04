import torch.nn as nn
import torch

class SpeakerClassifier(nn.Module):
    
    def __init__(self, encoder_embedding_dim, hidden_sc_dim, n_speakers) :
        super(SpeakerClassifier, self).__init__()
        self.model = nn.Sequential(nn.Linear(encoder_embedding_dim, hidden_sc_dim),
                                   nn.Linear(hidden_sc_dim, n_speakers))

    def forward(self, encoder_outputs, mask) :
        '''
        input :-
        encoder_outputs = [batch_size, seq_len, encoder_embedding_size]
        mask = [batch_size, seq_len]
        
        output :-
        log probabilities of speaker classification = [batch_size, seq_len, n_speakers]
        '''
        out = self.model(encoder_outputs)
        out = out.masked_fill(mask.unsqueeze(-1), 0)  
        return out