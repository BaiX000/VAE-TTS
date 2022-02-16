import argparse
import os
import re
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import CompTransTTSLoss
from dataset import Dataset
import numpy as np
import json
from g2p_en import G2p
from pypinyin import pinyin, Style
from text import text_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    #print("Raw Text Sequence: {}".format(text))
    #print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    #print("Raw Text Sequence: {}".format(text))
    #print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def custom_eval_synth(device, model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs


    # Synthesis infomation settings
    speaker_id = "SSB1328"
    text = "清新的茶樹香氣，隨身好防護"
    
    
    # Load speaker embed  
    spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(speaker_id),
    ))
    spker_embed = torch.from_numpy(spker_embed).float().to(device)            

    if model_config["vae_type"] == "VAE":
        mu, log_var = model.vae.encode(spker_embed)
        z = model.vae.reparameterize(mu, log_var)

    elif model_config["vae_type"] == "VSC":
        mu, log_var, log_spike = model.vae.encode(spker_embed)
        z = model.vae.reparameterize(mu, log_var, log_spike)
    elif model_config["vae_type"] == "Simple_VAE":
        mu, log_var = model.vae.encode(spker_embed)
        z = model.vae.reparameterize(mu, log_var)
    
    spker_embed = model.vae.decode(z)
    spker_embed = spker_embed.cpu().detach().numpy()
    
    #  creating batch
    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([preprocess_english(text, preprocess_config)])
    elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        texts = np.array([preprocess_mandarin(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    
    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
        speaker_map = json.load(f)
    
    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_map[speaker_id]])
    
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed)]
    
    # synthesis samples
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            output = model(*(batch[2:-1]), spker_embeds=batch[-1], step=step)
            predictions = output[:-2] # Update pitch and energy level
            
    if vocoder is not None:
        from utils.model import vocoder_infer
        mel_len = predictions[9][0].item()
        mel_prediction = predictions[1][0, :mel_len].float().detach().transpose(0, 1)
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    
    # log 
    if logger is not None:
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_custom".format(step, speaker_id),
        )