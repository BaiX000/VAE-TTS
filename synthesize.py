import re
import os
import json
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples
from dataset import TextDataset
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

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
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
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(device, model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-1]),
                spker_embeds=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                step = args.restore_step,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id_vae",
        type=str,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_ids_vae",
        type=str,
        nargs='+', 
        help="speaker IDs for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_gen",
        type=bool,
        help="generated speaker embed",
    )
    parser.add_argument(
        "--speaker_id_cross",
        type=str,
        help="test",
    )
    parser.add_argument(
        "--speaker_vae_cross",
        type=str,
        help="test",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Set Device
    #torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        #torch.cuda.manual_seed(train_config["seed"])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device of CompTransTTS:", device)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
            
        # get speaker embed        
        if args.speaker_id:
            speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
            spker_embed = np.load(os.path.join(
                preprocess_config["path"]["preprocessed_path"],
                "spker_embed",
                "{}-spker_embed.npy".format(args.speaker_id),
            )) if load_spker_embed else None
            model_config["vae_type"] = 'None'
        
        elif args.speaker_id_vae:
            speakers = np.array([speaker_map[args.speaker_id_vae]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
            spker_embed = np.load(os.path.join(
                preprocess_config["path"]["preprocessed_path"],
                "spker_embed",
                "{}-spker_embed.npy".format(args.speaker_id_vae),
            )) if load_spker_embed else None
            spker_embed = torch.from_numpy(spker_embed).float().to(device)  
            
            # print vae infos
            if model_config["vae_type"] == "VAE":
                mu, log_var = model.vae.encode(spker_embed)
                z = model.vae.reparameterize(mu, log_var)
                print(f"mu: {mu.detach()}, std: {0.5*torch.exp(log_var).detach()}, \n z: {z}")
                
            elif model_config["vae_type"] == "VSC":
                mu, log_var, log_spike = model.vae.encode(spker_embed)
                z = model.vae.reparameterize(mu, log_var, log_spike)
                print(
                    f"mu: {mu.detach()}, \n\
                    std: {0.5*torch.exp(log_var).detach()}, \n\
                    spike: {log_spike.exp().detach()}, \n\
                    selection: {torch.sigmoid(150*(torch.randn_like(log_spike)+log_spike.exp()-1))}, \n\
                    z: {z}"
                )
            
            change_dim = [4, 13, 15]
            for i in change_dim:
                z[0][i] = -3
            change_dim = [0, 3]
            for i in change_dim:
                z[0][i] = 2 
            print(z)
            
            spker_embed = model.vae.decode(z)
            spker_embed = spker_embed.cpu().detach().numpy()
        
        elif args.speaker_ids_vae: 
            assert len(args.speaker_ids_vae) == 2
            model_config["external_speaker_embed"] = True
            speakers = np.array([0]) # ??
            spker_A_embed = np.load(os.path.join(preprocess_config["path"]["preprocessed_path"], "spker_embed", "{}-spker_embed.npy".format(args.speaker_ids_vae[0]))) 
            spker_B_embed = np.load(os.path.join(preprocess_config["path"]["preprocessed_path"], "spker_embed", "{}-spker_embed.npy".format(args.speaker_ids_vae[1]))) 
            spker_A_embed = torch.from_numpy(spker_A_embed).float().to(device)
            spker_B_embed = torch.from_numpy(spker_B_embed).float().to(device)
            
            mu_A, log_var_A = model.vae.encode(spker_A_embed)
            z_A = model.vae.reparameterize(mu_A, log_var_A)
            mu_B, log_var_B = model.vae.encode(spker_B_embed)
            z_B = model.vae.reparameterize(mu_B, log_var_B)
            z_mean = (z_A*0.4 + z_B*0.6)
            print(z_mean)
            spker_embed = model.vae.decode(z_mean)
            spker_embed = spker_embed.cpu().detach().numpy()
        elif args.speaker_gen:
            speakers = np.array([0]) # ??
            z = torch.normal(mean=0, std=1, size=(1,16)).float().to(device)
            #z = abs(z)*2
            
            change_dim = [9]
            for i in change_dim:
                z[0][i] = -5
            print(z)
            spker_embed = model.vae.decode(z)
            spker_embed = spker_embed.cpu().detach().numpy()
        
        elif args.speaker_vae_cross:
            speakers = np.array([0]) 
            spker_embed = np.load(os.path.join("preprocessed_data/LibriTTS", "spker_embed", "{}-spker_embed.npy".format(args.speaker_vae_cross)))
            spker_embed = torch.from_numpy(spker_embed).float().to(device)  
            mu, log_var = model.vae.encode(spker_embed)
            z = model.vae.reparameterize(mu, log_var)
            print(f"mu: {mu.detach()}, std: {0.5*torch.exp(log_var).detach()}, \n z: {z}")
            spker_embed = model.vae.decode(z)
            spker_embed = spker_embed.cpu().detach().numpy()
       
        elif args.speaker_id_cross:
            speakers = np.array([0]) 
            spker_embed = np.load(os.path.join("preprocessed_data/LibriTTS", "spker_embed", "{}-spker_embed.npy".format(args.speaker_id_cross)))

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(device, model, args, configs, vocoder, batchs, control_values)
