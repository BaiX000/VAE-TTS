import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor
from utils.tools import get_mask_from_lengths
from model import vae
import random

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
                
        # add vae model
        self.vae_type = model_config["vae_type"]
        if self.vae_type == "None":
            self.vae = None
        elif self.vae_type == "VAE":  
            self.vae = vae.VAE()
        elif self.vae_type == "VSC":
            self.vae = vae.VSC()
        elif self.vae_type == "Simple_VAE":
            self.vae = vae.Simple_VAE()
            
        self.vae_start_steps = train_config["vae"]["vae_start_steps"]
        self.org_embed_rate = train_config["vae"]["org_embed_rate"]
        
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
        
        speaker_embeds = None
        # speaker_emb is not None if it is multispeakers. 
        # speaker_emb will be either nn.Linear(n_speaker, encoder_hidden) or nn.Linear(external_dim(512), encoder_hidden)
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                # A. Training without vae model
                if self.vae_type is None:
                    vae_results = None
                    speaker_embeds = self.speaker_emb(spker_embeds)
                # B. Training with vae model
                else:
                    if self.vae_type == "VAE":
                        recons_spker_embeds, org_input, mu, log_var = self.vae(spker_embeds)
                        vae_results = [recons_spker_embeds, org_input, mu, log_var]
                        
                    elif self.vae_type == "VSC":
                        recons_spker_embeds, org_input, mu, log_var, log_spike = self.vae(spker_embeds)
                        vae_results = [recons_spker_embeds, org_input, mu, log_var, log_spike]
                    elif self.vae_type == "Simple_VAE":
                        recons_spker_embeds, org_input, mu, log_var = self.vae(spker_embeds)
                        vae_results = [recons_spker_embeds, org_input, mu, log_var]
                        
                            
                    if self.training:
                        if step > self.vae_start_steps:
                            # -- Start VAE finetuning process ---
                            # Not finetune Encoder part
                            texts = texts.detach()
                            text_embeds = text_embeds.detach()        
                            # pick some org embedding when training
                            r = random.uniform(0, 1)
                            speaker_embeds = self.speaker_emb(spker_embeds) if r < self.org_embed_rate else self.speaker_emb(recons_spker_embeds)
                        else:
                            speaker_embeds = self.speaker_emb(spker_embeds)
                            vae_results = None

                    else:                         
                        # evaluation
                        if step > self.vae_start_steps:
                            speaker_embeds = self.speaker_emb(recons_spker_embeds)
                        else:
                            speaker_embeds = self.speaker_emb(spker_embeds)
            
                        

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
            # add
            vae_results,
            # --- 
            p_targets,
            e_targets,            
        )
