""" 
Research @ Feizi Lab - University of Maryland, College Park

This script edits the set of layers in diffusion models to remove copyrighted material

Notes:
- This is a general script for SD-v1 and SD-v2 versions (with sliding window size of 2), but can be adapted for any text-to-image model
"""

# Libraries
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import argparse 
import tqdm 
from PIL import Image  
#import open_clip
import pickle 
import os 
from torchvision.utils import save_image
#from torchvision.utils import save_image
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from torch import autocast


# Auth token
auth_token = "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"
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

# Total number of computable operations / modules -- 709
def high_level_layers(model):
    # Counter for the list
    c = 0
    # Stores the relevant layers to perform causal tracing on 
    relevant_modules = []
    # Total list of all modules
    named_module_list = []
    for n,m in model.unet.named_modules():
        c += 1
        named_module_list.append(n)


    # Ends with 'attn2', 'attn1'
    attn_list = []
    for item in named_module_list:
        if 'attn2' in item and ('to_k' in item or 'to_v' in item):
            attn_list.append(item)
    
    #print(attn_list)
    # Layernames
    return attn_list

# Model Editing Function - Non-SDXL models
def train_edit(args, ldm_stable, layer_edit_modules, key_embeddings, value_embeddings):
    print(f'############## Editing Function ################')

    # Iterate through each of the modules and then update the modules based on the closed-form expression
    for layer_num in range(0, len(layer_edit_modules)):
        # Updatability Part
        with torch.no_grad():
            
            # Current Weight Matrix; 
            curr_weight_matrix = layer_edit_modules[layer_num].weight

            ############  First part of the solution ############
            id_matrix_mat_1 = args.reg_key * torch.eye(layer_edit_modules[layer_num].weight.shape[1], device = layer_edit_modules[layer_num].weight.device)
            x_matrix = torch.matmul(key_embeddings.T, key_embeddings)
            mat1 = torch.inverse(x_matrix + id_matrix_mat_1)

            ############  Second part of the solution ###########
            # X^{T}Y
            x_matrix_mat_2 = torch.matmul(key_embeddings.T, torch.matmul(value_embeddings, curr_weight_matrix.T))
            additional_reg = args.reg_key * curr_weight_matrix.T 
            mat2 = x_matrix_mat_2 + additional_reg
            

            # Final Update due to the least squared solution
            final_update = torch.matmul(mat1, mat2)


            # Update the layer 
            layer_edit_modules[layer_num].weight = torch.nn.Parameter(final_update.T)
            

        
        #break 

    return 


# LocoEdit function #
def loco_edit():
    # Arg parser
    parser = argparse.ArgumentParser()
    # Basic operations
    parser.add_argument("--attribute", default="style", type=str, required=False, help = "Attribute Label")
    parser.add_argument("--sd_version", default="1", type=str, required=False, help = "SD Label")
    parser.add_argument("--eval", default="True", type=str, required=False, help = "Label")
    parser.add_argument("--prompt", default="a painting", type=str, required=False, help = "Prompt")
    parser.add_argument("--og_prompt", default="a house in the style of van gogh", type=str, required=False, help = "Prompt")
    parser.add_argument("--debug_prompt", default="A photo of a dog", type=str, required=False, help = "Prompt-Debug")
    parser.add_argument("--edit_type", default="style", type=str, required=False, help = "Type of Edit : Style / Object / Facts")
    # Artist
    parser.add_argument("--artist", default="van gogh", type=str, required=False, help = "Artist Name")
    # Object
    parser.add_argument("--object", default="r2d2", type=str, required=False, help = "Object")
    parser.add_argument("--replace", default='False', type=str, required=False, help = "Replace operation")
    parser.add_argument("--device", default='cuda:0', type=str, required=False, help = "Cuda operation")
    parser.add_argument("--do_edit", default='True', type=str, required=False, help = "Cuda operation")
    parser.add_argument("--all", default='False', type=str, required=False, help = "If all layers need to be updated")
    parser.add_argument("--eos", default='False', type=str, required=False, help = "If EOS tokens are used")

    # Regularization strength
    parser.add_argument("--reg_key", default=0.01, type=float, required=False, help = "Cuda operation")
    parser.add_argument("--reg_value", default=0.01, type=float, required=False, help = "Cuda operation")

    # Seed
    parser.add_argument("--seed", default=0, type=int, required=False, help = "Cuda operation")
    parser.add_argument("--seq", default=4, type=int, required=False, help = "Sequence length for operation")
    parser.add_argument("--start_loc", default=8, type=int, required=False, help = "Start location")

    # Arguments
    args = parser.parse_args()

    # Device 
    print(f'Device: {args.device}')

    # Define the device
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    # SD-Version 1.4
    if args.sd_version == '1':
        ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token, safety_checker = None, cache_dir = '/fs/nexus-scratch/sbasu12/projects/cross_edit').to(device)
    
    # SD-Version 2.1
    elif args.sd_version == '2':
        ldm_stable = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", use_auth_token=auth_token, cache_dir = '/fs/nexus-scratch/sbasu12/projects/cross_edit').to(device)
    
    # Prompt-hero
    elif args.sd_version == 'prompthero':
        ldm_stable = StableDiffusionPipeline.from_pretrained("prompthero/openjourney", use_auth_token=auth_token, cache_dir = '/fs/nexus-scratch/sbasu12/projects/cross_edit').to(device)

    # Relevant layers #     
    relevant_layers = sorted(high_level_layers(ldm_stable))

    # Start location # 
    start_loc = args.start_loc # start_loc = 9 is also another option
    relevant_edit_layers = relevant_layers[start_loc*2: start_loc*2 + args.seq]
    print(f'Relevant Editing layers : {relevant_edit_layers}')

    # 
    def design_keys(artist):
        # Prompts : With basic augmentations
        prompts = ['' + artist, 'a painting in the style of ' + artist, 'a photo in the style of ' + artist, 'a picture in the style of ' + artist]

        # Add augmentations : TODO 
        return prompts
    
    def design_objects(artist):
        # Prompts : With basic augmentations
        prompts = ['' + artist, 'an image of ' + artist, 'a photo of ' + artist, 'a picture of ' + artist]

        # Add augmentations : TODO 
        return prompts
    

    # Function to generate output embeddings from the text-encoder
    def generate_text_embeddings(model, key_prompt, value=False):
        # Obtaining the embeddings of the last subject token
        # Key : Text-Embedding
        key_embeddings = []
        key_tokens = []
        for prompt_curr in key_prompt:
            text_input_curr= model.tokenizer(prompt_curr, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
            # Append the embeddings
            with torch.no_grad():
                text_embeddings = model.text_encoder(text_input_curr.input_ids.to(model.device))[0]
                key_embeddings.append(text_embeddings[0])
                key_tokens.append(text_input_curr['input_ids'][0])

        # Storing the final key embeddings
        final_key_embeddings = []

        # Iterate through the text embeddings and extract the last-subject-token embeddings
        c = 0
        for txt_embedding in key_embeddings:
            token_ids = key_tokens[c]
            
            # Position
            pos = 0
            for tok in token_ids:
                # If last subject token is encountered -- then break 
                if tok == 49407:
                    break
                
                pos += 1
            
            # Value == False
            if value == False:
                # Relevant embedding
                if args.eos == 'False':
                    # Embedding
                    rel_embedding = txt_embedding[pos-1]
                    final_key_embeddings.append(rel_embedding.reshape(1,-1))
                
                else:
                    # EOS function
                    print(f'Using EOS')
                    for k in range(pos-1, pos):
                        rel_embedding = txt_embedding[k]
                        final_key_embeddings.append(rel_embedding.reshape(1,-1))


            # Value == True
            else:
                # Embedding
                rel_embedding = txt_embedding[pos-1]
                final_key_embeddings.append(rel_embedding.reshape(1,-1))

            c += 1
        

        
        # Size of embeddings
        final_keys = torch.cat(final_key_embeddings, dim=0)

        # Return the final Keys
        return final_keys 


    # 
    # Key prompts
    key_prompt = design_keys(args.artist)
    layer_edit_modules = []
    for l in relevant_edit_layers:
        # Iterate through the modules in UNet
        for n, m in ldm_stable.unet.named_modules():
            if n == l:
                layer_edit_modules.append(m)

        
    
    # Finished storing the layers which are edited
    print(f'Number of the layers which are edited : {len(layer_edit_modules)}')
    key_embeddings = generate_text_embeddings(ldm_stable, key_prompt)
    target_prompt = ['a painting']*len(key_embeddings)

    # Flag
    value_embeddings = generate_text_embeddings(ldm_stable, target_prompt, value=True)

    print(f'Size of Key Embeddings : {len(key_embeddings)}')
    print(f'Size of Value embeddings : {len(value_embeddings)}')

    ##################################### 
    train_edit(args, ldm_stable, layer_edit_modules, key_embeddings, value_embeddings)

    #generator = torch.Generator().manual_seed(args.seed).to(device)
    save_directory = '/fs/nexus-scratch/sbasu12/projects/cross_edit/edit_results/' + args.sd_version + '/style/' + args.artist

    isExist = os.path.exists(save_directory)
    print(f'Save directory : ########### {save_directory}')
    # Not exist -- make the save directory
    if not isExist:
        os.makedirs(save_directory)
    
    
    prompts = ['An old house house in the style of ' + args.artist, 'painting of a wheat field by '+args.artist]
    print(f'Generating for Artist : {args.artist} <======> Number of Prompts : {len(prompts)}')

    def dummy(images, **kwargs):
        return images, False 

    for pr in prompts:
        print(f'Current Prompt : {pr}')
        # Generate Images 
        with torch.no_grad():
            # Images
            images_ = ldm_stable([pr], output_type="pt").images
            im1 = images_[0]

            # Save the image 
            im = Image.fromarray((im1*255).astype(np.uint8))
            im.save(save_directory + '/style_' + pr + '.png')

    return 


# Main function
if __name__ == "__main__":
    # Primary function which calls for the model edit function 
    loco_edit()
