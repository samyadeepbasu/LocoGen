from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch.nn.functional as nnf
from torchvision import transforms
import numpy as np
import abc
import argparse 
import tqdm 
from PIL import Image  
import nethook_diffusion
import open_clip
import pickle
import os

from prompts import monet_prompts, dali_prompts, mann_prompts, greg_prompts, van_gogh_prompts, picasso_prompts
from prompts import monet_normal, dali_normal, mann_normal, greg_normal, van_gogh_normal, picasso_normal

# Define the Components of the Diffusion Model 

auth_token = "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"
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77



# Loading Diffusion Model

device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token,).to(device)
tokenizer = ldm_stable.tokenizer
model = ldm_stable



# Loading Key/Value Matrices

key_matrices, value_matrices = [], []

for n, m in model.unet.named_modules():
    if 'attentions' in n and 'attn2' in n:
        if 'to_k' in n:
            key_matrices.append((n, m, ))
        elif 'to_v' in n:
            value_matrices.append((n, m, ))
            
key_matrices = sorted(key_matrices)
value_matrices = sorted(value_matrices)

# Prompts 

all_style_prompts = monet_prompts + dali_prompts + mann_prompts + greg_prompts + van_gogh_prompts + picasso_prompts
all_prompts = monet_normal + dali_normal + mann_normal + greg_normal + van_gogh_normal + picasso_normal


def get_v1_normal_prompts():
    return all_prompts

def get_v2_normal_prompts(artist):
    if artist == 'monet':
        return monet_normal
    
    elif artist == 'salvador dali':
        return dali_normal
    
    elif artist == 'jeremy mann':
        return mann_prompts
    
    elif artist == 'greg rutkowski':
        return greg_prompts
    
    elif artist == 'van gogh':
        return van_gogh_prompts
    
    elif artist == 'pablo picasso':
        return picasso_prompts


def get_v1_style_prompts(artist):
    return [x.format(artist) for x in all_style_prompts]

def get_v2_style_prompts(artist):
    if artist == 'monet':
        return [x.format(artist) for x in monet_prompts]
    
    elif artist == 'salvador dali':
        return [x.format(artist) for x in dali_prompts]
    
    elif artist == 'jeremy mann':
        return [x.format(artist) for x in mann_prompts]
    
    elif artist == 'greg rutkowski':
        return [x.format(artist) for x in greg_prompts]
    
    elif artist == 'van gogh':
        return [x.format(artist) for x in van_gogh_prompts]
    
    elif artist == 'pablo picasso':
        return [x.format(artist) for x in picasso_prompts]


artists = ['monet', 'salvador dali', 'jeremy mann', 'greg rutkowski', 'van gogh', 'pablo picasso']
artist_id = 5 # Which artist to explore
extensive = False # True (all prompts)/ False (specific prompts)

artist = artists[artist_id]

def get_embedding(prompts):
    cond_input = model.tokenizer(prompts, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
    cond_embeddings = model.text_encoder(cond_input.input_ids.to(model.device))[0]
    return cond_embeddings

if extensive:
    normal_prompts = get_v1_normal_prompts()
    artist_prompts = get_v1_style_prompts(artist)
else:
    normal_prompts = get_v2_normal_prompts(artist)
    artist_prompts = get_v2_style_prompts(artist)
    
normal_embd = get_embedding(normal_prompts).detach()
art_embd = get_embedding(artist_prompts).detach()



# Layers 8 and 9 are worth exploring

def get_activations(embd, matrices, layers = [8, 9]):
    activations = []
    for layer_id in layers:
        matrix = matrices[layer_id][1]
        out = matrix(embd[:, 76, :]) # last subject token
        activations.append(out)
    
    return activations



# Obtaining Layer 8 and Layer Key/Value Activations

keys_art = get_activations(art_embd, key_matrices)
values_art = get_activations(art_embd, value_matrices)

keys_normal = get_activations(normal_embd, key_matrices)
values_normal = get_activations(normal_embd, value_matrices)


art_l8_keys = keys_art[0].detach()
n_l8_keys = keys_normal[0].detach()

art_l9_keys = keys_art[1].detach()
n_l9_keys = keys_normal[1].detach()

art_l8_values = values_art[0].detach()
n_l8_values = values_normal[0].detach()

art_l9_values = values_art[1].detach()
n_l9_values = values_normal[1].detach()


def get_z_score(p1, p2):
    p1 = p1.cpu().numpy()
    p2 = p2.cpu().numpy()
    
    X1 = np.mean(p1)
    X2 = np.mean(p2)
    
    s1 = np.std(p1)
    s2 = np.std(p2)
    
    return (X1 - X2) / ((s1 ** 2 / len(p1) + s2 ** 2 / len(p2)) ** 0.5)


results = {}

for (st, no, l_id) in [
    (art_l8_keys, n_l8_keys, 16),
    (art_l8_values, n_l8_values, 17),
    (art_l9_keys, n_l9_keys, 18),
    (art_l9_values, n_l9_values, 19)]:
    
    z_scores = []
    
    for ftr_id in range(st.shape[1]):
        z = get_z_score(st[:, ftr_id], no[:, ftr_id])
        z_scores.append(z)
    
    z = np.array(z_scores)
    z = np.abs(z)
    neurons = np.argsort(z)
    
    results[l_id] = {}

    nums = [10, 30, 50, 100, 200]
    for num_of_neurons in nums:  
        list_of_neurons = neurons[-num_of_neurons: ]
        new_values = np.zeros(num_of_neurons)

        for i, n_id in enumerate(list_of_neurons):
            new_values[i] = np.mean((no[:, n_id]).cpu().numpy())
            
        results[l_id][num_of_neurons] = {
            'indices': torch.from_numpy(list_of_neurons).type(torch.LongTensor),
            'values': torch.from_numpy(new_values),
        }    
    

if extensive:
    torch.save(results, f'styles/v2-{artist}-extensive.pth')
else:
    torch.save(results, f'styles/v2-{artist}-specific.pth')
