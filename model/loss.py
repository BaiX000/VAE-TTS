import torch
import torch.nn as nn
import torch.nn.functional as F
from . import vae


class CompTransTTSLoss(nn.Module):
    """ CompTransTTS Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTSLoss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.binarization_loss_enable_steps = train_config["duration"]["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = train_config["duration"]["binarization_loss_warmup_steps"]
        
        self.vae_start_steps = train_config["vae"]["vae_start_steps"]
        self.kld_loss_enable_steps = train_config["vae"]["kld_loss_enable_steps"]
        self.kld_loss_warmup_steps = train_config["vae"]["kld_loss_warmup_steps"]
        self.kld_final_weight = train_config["vae"]["kld_final_weight"]
        
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # check vae type
        self.vae_type = model_config["vae_type"]
        assert self.vae_type in ['VAE', 'VSC', 'Simple_VAE']
        if self.vae_type == "VAE":
            self.vae_loss = vae.VAE().loss_function
        elif self.vae_type == "VSC":
            self.vae_loss = vae.VSC().loss_function 
        elif self.vae_type == "Simple_VAE":
            self.vae_loss = vae.Simple_VAE().loss_function 
            
    def forward(self, inputs, predictions, step):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            _,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            # vae results
            vae_results,
            
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        if self.learn_alignment:
            attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
            log_duration_targets = torch.log(attn_hard_dur.float() + 1)
        else:
            log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        ctc_loss = bin_loss = torch.zeros(1).to(mel_targets.device)
        if self.learn_alignment:
            ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
            if step < self.binarization_loss_enable_steps:
                bin_loss_weight = 0.
            else:
                bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        # VAE LOSS
        if vae_results is not None:
            if self.vae_type == "VAE" or self.vae_type == "Simple_VAE":             
                recons, org_input, mu, log_var = vae_results
                recons_loss, kld_loss = self.vae_loss(recons, org_input, mu, log_var)          

            elif self.vae_type == "VSC":
                recons, org_input, mu, log_var, log_spike = vae_results
                recons_loss, kld_loss = self.vae_loss(recons, org_input, mu, log_var, log_spike)         
            
            if step < self.vae_start_steps:
                kld_loss = kld_loss * 0.
                recons_loss = recons_loss * 0 
                vae_loss = recons_loss # equal zero
            elif step < self.kld_loss_enable_steps:
                kld_loss = kld_loss * 0.
                vae_loss = recons_loss
            else:
                kld_weight = min((step-self.kld_loss_enable_steps) / self.kld_loss_warmup_steps, 1.0) * self.kld_final_weight
                vae_loss = recons_loss + kld_weight * kld_loss
        # --- end ----
            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + ctc_loss + bin_loss + vae_loss
            )
        else:
            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + ctc_loss + bin_loss
            )
            vae_loss = recons_loss = kld_loss = torch.zeros(1)

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ctc_loss,
            bin_loss,
            vae_loss,
            recons_loss,
            kld_loss
        )


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
