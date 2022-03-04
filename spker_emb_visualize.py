import argparse
import numpy as np
import torch.nn as nn
import os
from sklearn import manifold
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

class TSNE(nn.Module):
    def __init__(
        self, 
        args,
        n_component=2,
        perplexity=30, 
        n_iter=1000,
        init="random",
        random_state=0, 
        verbose=0,
    ):
        super(TSNE, self).__init__()
        self.args = args
        self.n_component = n_component
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.init = init
        self.random_state = random_state
        self.verbose = verbose
        
    def forward(self, X, spker_names, spker_langs, spker_genders):
        #t-SNE
        X_tsne = manifold.TSNE(
            n_components=self.n_component, 
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose
        ).fit_transform(X)

        #Color map
        lang_map = ["green", "blue"]
        gender_map = ["hotpink", "#88c999"]
        lang_text = ["O", "X"]
        gender_text = ["g", "b"]
        
        #Data Visualization
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
        fig = plt.figure(figsize=(10, 10))

        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], gender_text[spker_genders[i]], color=lang_map[spker_langs[i]], 
                     fontdict={'weight': 'bold', 'size': 5})
        plt.xticks([])
        plt.yticks([])
        
        fig.savefig('spk_emb_visualize_{}.pdf'.format(self.args.dataset))
        
    
def main(args):
    in_dir = "preprocessed_data/{}/spker_embed".format(args.dataset)
    
    # load gender info. of AISHELL3 & LibriTTS
    gender_map = {}
    with open('../Dataset/AISHELL-3/spk-info.txt') as f:
        lines = f.readlines()
        lines = lines[3:]
        for line in lines:
            spker, age , gender, region = line.strip().split('\t')
            gender_map[spker] = 0 if gender == "female" else 1  

    with open('../Dataset/LibriTTS/SPEAKERS.txt') as f:
        lines = f.readlines()
        lines = lines[12:]
        for line in lines:
            spker, gender, subset = line.strip().split('|')[:3]
            spker, gender, subset = spker.strip(), gender.strip(), subset.strip()
            if subset != "train-clean-360":
                continue
            gender_map[spker] = 0 if gender == "F" else 1
    
    spker_embs = []
    spker_names = []
    spker_langs = []
    spker_genders = []
    for i, f in enumerate(os.listdir(in_dir)):
        spker = f.split("-")[0]
        spker_lang = 1 if spker.startswith("SSB") else 0
        spker_embs.append(np.load(os.path.join(in_dir, f))[0])
        spker_names.append(spker)
        spker_langs.append(spker_lang)
        spker_genders.append(gender_map[spker])
        
    spker_embs = np.array(spker_embs)
    tsne = TSNE(args)
    tsne(spker_embs, spker_names, spker_langs, spker_genders)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)