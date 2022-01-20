import argparse
import matplotlib.pyplot as plt
import os
import yaml
from utils.tools import get_configs_of, to_device, synth_samples
from utils.model import get_model, get_vocoder
import torch
import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import figure

def main(device, model, maps, args, configs):
    preprocess_config, model_config, train_config = configs   
    in_dir = os.path.join(preprocess_config['path']['preprocessed_path'], 'spker_embed')
    
    gender_map = maps['gender_map']
    region_map = maps['region_map']
    age_map = maps['age_map']
    print(age_map, region_map)
    
    # set plots    
    #plt.figure(figsize=(16, 12), dpi=800)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(16, 32))
    fig0, (ax01, ax02, ax03, ax04) = plt.subplots(4, figsize=(16, 32))
    fig1, (ax11, ax12, ax13, ax14) = plt.subplots(4, figsize=(16, 32))
    
    ax1.set(ylabel='mean')
    ax2.set(xlabel='dimension', ylabel='std')

    for i, file in enumerate(tqdm(os.listdir(in_dir))):
        spker = file.split('-')[0]
        spker_embed = np.load(os.path.join(in_dir, file))
        spker_embed = torch.from_numpy(spker_embed).float().to(device)  

        
        with torch.no_grad():
            mu, log_var = model.vae.encode(spker_embed)
            mu = mu.cpu().squeeze().numpy()
            std = (0.5*torch.exp(log_var)).cpu().squeeze().numpy()
        # plot data
        color_map = ['hotpink', '#88c999', 'yellow', 'blue']
        
        for dim in range(len(mu)):            
            ax1.scatter(dim, mu[dim], s=2, c=color_map[gender_map[spker]])
            ax2.scatter(dim, std[dim], s=2, c=color_map[gender_map[spker]])
            ax3.scatter(dim, mu[dim], s=2, c=color_map[region_map[spker]])
            ax4.scatter(dim, std[dim], s=2, c=color_map[region_map[spker]])
            ax5.scatter(dim, mu[dim], s=2, c=color_map[age_map[spker]])
            ax6.scatter(dim, std[dim], s=2, c=color_map[age_map[spker]])
            
            if gender_map[spker] == 0:
                ax01.scatter(dim, mu[dim], s=2, c=color_map[region_map[spker]])
                ax02.scatter(dim, std[dim], s=2, c=color_map[region_map[spker]])
                ax03.scatter(dim, mu[dim], s=2, c=color_map[age_map[spker]])
                ax04.scatter(dim, std[dim], s=2, c=color_map[age_map[spker]])
            else:
                ax11.scatter(dim, mu[dim], s=2, c=color_map[region_map[spker]])
                ax12.scatter(dim, std[dim], s=2, c=color_map[region_map[spker]])
                ax13.scatter(dim, mu[dim], s=2, c=color_map[age_map[spker]])
                ax14.scatter(dim, std[dim], s=2, c=color_map[age_map[spker]])
            
    
    fig.savefig('visualize.pdf')
    fig0.savefig('visualize0.pdf')
    fig1.savefig('visualize1.pdf')
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="name of dataset",
        )
    parser.add_argument("--restore_step", type=int, required=True)
    args = parser.parse_args()
    

    # get config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    
    # set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # get model
    model = get_model(args, configs, device, train=False)
    
    # load gender map
    gender_map = {}
    region_map = {}
    age_map = {}
    if args.dataset == "AISHELL3":
        with open('../Dataset/AISHELL-3/spk-info.txt') as f:
            lines = f.readlines()
            lines = lines[3:]
            for line in lines:
                spker, age , gender, region = line.strip().split('\t')
                gender_map[spker] = 0 if gender == "female" else 1  

                if region == 'north':
                    region_map[spker] = 0
                elif region == 'south':
                    region_map[spker] = 1
                elif region == 'others':
                    region_map[spker] = 2
                
                if age == 'A':
                    age_map[spker] = 0
                elif age == 'B':
                    age_map[spker] = 1
                elif age == 'C':
                    age_map[spker] = 2
                elif age == 'D':
                    age_map[spker] = 3
                    
    maps = {'gender_map': gender_map, 'region_map':region_map, 'age_map': age_map}


    main(device, model, maps, args, configs)