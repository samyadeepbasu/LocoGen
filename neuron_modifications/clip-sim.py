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

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from test_prompts import test_prompts as tp
import json


# Define the Components of the Diffusion Model 
auth_token = "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"
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--artist", default="van gogh", type=str, required=False, help = "Artist")
    parser.add_argument("--type", default="extensive", type=str, required=False, help = "Artist")
    args = parser.parse_args()
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', cache_dir='/cmlscratch/krezaei/cache') #, cache_dir = '/sensei-fs/tenants/Sensei-AdobeResearchTeam/share-samyadeepb/diffusion_trace')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    
    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    
    test_prompts = tp[args.artist]
    
    model = model.to(device)
    prompt = f'style of {args.artist}'
    text = tokenizer([prompt]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    
    results = {}
    
    summary = {x: 0 for x in [0, 30, 50, 100]}
    summary['layer'] = 0
    
    with torch.no_grad():
        for _ in tqdm(test_prompts): 
            results[_] = {}
            
            for cnt in [-1, 0, 30, 50, 100,]:
                
                score = 0
                
                for seed in [0, 1, 2, 3]:
                    if cnt == -1:
                        path_to_image = f'extensive_images/{args.artist}/{_.replace(" ", "_")}/{seed}_layer.png'
                    else:
                        path_to_image = f'{args.type}_images/{args.artist}/{_.replace(" ", "_")}/{seed}_{cnt}.png'
                        
                    image = Image.open(path_to_image)
                    image = preprocess(image).unsqueeze(0).to(device)
                    
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    cos_sim = torch.nn.CosineSimilarity()
                    
                    score += cos_sim(image_features, text_features).item()
                
                score /= 4
                
                if cnt == -1:
                    results[_]['layer'] = score
                    summary['layer'] += score
                else:
                    results[_][cnt] = score
                    summary[cnt] += score
                    
        
        for x in summary:
            summary[x] = summary[x] / len(test_prompts)
                
            
        with open(f'results/{args.type}_{args.artist}.json', 'w') as f:
            f.write(
                json.dumps(
                    {
                        'summary': summary,
                        'details': results},
                    indent=4))
