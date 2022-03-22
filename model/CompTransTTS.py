import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .gradient_reversal import grad_reverse
from .speaker_classifier import SpeakerClassifier
from .residual_encoder import ResidualEncoder
class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config

        if model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        elif model_config["block_type"] == "lstransformer":
            from .transformers.lstransformer import TextEncoder, Decoder
        elif model_config["block_type"] == "fastformer":
            from .transformers.fastformer import TextEncoder, Decoder
        elif model_config["block_type"] == "conformer":
            from .transformers.conformer import TextEncoder, Decoder
        elif model_config["block_type"] == "reformer":
            from .transformers.reformer import TextEncoder, Decoder
        else:
            raise NotImplementedError

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )
                    
        # add lid embedding table
        self.language_emb = nn.Embedding(2, model_config["transformer"]["encoder_hidden"])
        
        # add gradient reversal component
        
        self.speaker_classifier = None
        if model_config["gradient_reversal"]["enable"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
            self.speaker_classifier = SpeakerClassifier(
                model_config["transformer"]["encoder_hidden"],
                model_config["gradient_reversal"]["spker_clsfir_hidden"],
                n_speaker,
            )
        '''
        # add residual encoding
        if model_config["residual_encoder"]["enable"]:
            self.residual_encoder = ResidualEncoder(
                n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                decoder_hidden=model_config["transformer"]["decoder_hidden"],
                residual_encoding_dim=model_config["residual_encoder"]["residual_encoder_dim"],
            )
        '''
        
        self.spker_decoder_residual = model_config["spker_decoder_residual"]
        
    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        lids=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)
        
        # add gradient revesal
        spker_clsfir_output = None
        if self.speaker_classifier is not None:
            texts_for_spker_clsfir = grad_reverse(texts)
            spker_clsfir_output = self.speaker_classifier(texts_for_spker_clsfir, src_masks)
        
        
        
        speaker_embeds = None
        # speaker_emb is not None if it is multispeakers. 
        # speaker_emb will be either nn.Linear(n_speaker, encoder_hidden) or nn.Linear(external_dim(512), encoder_hidden)
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds)
              
        language_embeds = self.language_emb(lids)

        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
        ) = self.variance_adaptor(
            speaker_embeds,
            language_embeds,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        #residual_encoding = self.residual_encoder(mels)
        if self.spker_decoder_residual:
            print("decoder! residual!")
            output = output + speaker_embeds.unsqueeze(1).expand(
                -1, output.shape[1], -1
            )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            # -- add --
            spker_clsfir_output,
            p_targets,
            e_targets,            
        )
