""" 
Using different prompts in different cross-attention layers
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
import nethook_diffusion
import open_clip
import pickle
import os

from tqdm import tqdm
from test_prompts import test_prompts as tp

# Define the Components of the Diffusion Model 
auth_token = "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"
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

glob_seed = 0
neurons_cnt = 50


## Define the hook inside the generation function
def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

# Function to get the module 
def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


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



def trace_with_patch(model, latents_input, t, context, guidance_scale, trace = False, layer_ = None):
    if trace == False:
        # Takes in the time-steps # 
        with torch.no_grad(): # TODO : Attach nethook.Tracedict() # TODO : Define your own trace dictionary
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"] # (2, 4, 64, 64) -- because of two conditions

        return noise_pred 
    

    # Here the call for layer-manipulation operation will take place 
    with torch.no_grad(), nethook_diffusion.TraceDict(model, layer_): 
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"] 
    
    # 
    return noise_pred #noise_pred



# Diffusion step
def diffusion_step(model, latents, relevant_layers, layer_wise_prompt, modification_layers, info, context, t, guidance_scale, layer_, low_resource=False, window = False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        # latents: (1, 4, 64, 64)
        latents_input = torch.cat([latents] * 2) # (2*B, 4, 64, 64) ; For 2 prompts -- (4, 4, 64, 64) with the first 2 uncodnitional and the rest conditional

        
        global neurons_cnt
        #print(f'Final step')
        with torch.no_grad(), nethook_diffusion.TraceDict(model, relevant_layers, layer_wise_prompt, modification_layers, neurons_cnt, info):
            # This is where the noise prediction network is called -- 
            # The trace function will compose of [layer1, layer2, ......, layer16] prompt-embeddings;
            # Given the particular layer -> it will use the transformation function to get the output of the W_{k}, W_{v} as the transformed embedding
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"] 
    

        # Noise prediction #
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2) # Each is (2*B, 4, 64, 64)

    
    # Effective noise : noise_with_uncond + guidance_scale * (noise_with_cond - noise_pred_uncond) # (Combine from (2, 4, 64, 64) to (1, 4, 64, 64))
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    # Latents after the scheduling step
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    #latents = controller.step_callback(latents)
    return latents


# 
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


# View-images
def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    #display(pil_img)


# Decode tokens
def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


# Trace Patch
def trace_patch(model, latents, relevant_layers, layer_wise_prompt, modification_layers, info, context, guidance_scale, low_resource, layer_= None):
    # Iterate through the diffusion step
    for t in model.scheduler.timesteps:
        # This should returns a dictionary corresponding to output of each of the components (resnet, mlp, attn, cross-attention) - These are the kinds in our system
        # Should contain the average indirect estimation
        # {"0_mlp_1": xx, "0_mlp_2": xx, ........, "0_mlp_3": xx, .......} # 
        #print(f'Timestep: {t}')
        latents = diffusion_step(model, latents, relevant_layers, layer_wise_prompt, modification_layers, info, context, t, guidance_scale, layer_, low_resource)


    

    # Convert from latents to diffusion step
    image = latent2image(model.vae, latents)

    
    return image



""" 
Define the trace-state function which iterates over various generations to obtain the generated latent 
"""
def trace_states(args, layer_wise_prompt, modification_layers, info, model, latents, context, guidance_scale, prompt_curr, low_resource):
    # Define the layers one needs to pass
    relevant_layers = sorted(high_level_layers(model))
  
    
    c = 0
    # Save Path
    # Trace Patch
    image = trace_patch(model, latents, relevant_layers, layer_wise_prompt, modification_layers, info, context, guidance_scale, low_resource) # Shape[3, 3, 512, 512] # (0th index: Clean; 1th index: Corrupted; 2nd index: Restored state)
    
    global neurons_cnt, glob_seed
    
    save_path = f'{args.type}_images/{args.artist}/{prompt_curr.replace(" ", "_")}'
    isExist = os.path.exists(save_path)

    # Not exist
    if not isExist:
        os.makedirs(save_path)
    
    # Image
    image_ = image[0] #.reshape((3,512,512))
    im1 = Image.fromarray(image_)
    
    
    if args.edit == 'neuron':
        im1.save(f'{save_path}/{glob_seed}_{neurons_cnt}.png')
    else:
        im1.save(f'{save_path}/{glob_seed}_layer.png')
    
    # Return score_dict
    return 



# Generate image ldm;
@torch.no_grad()
def generate_image_ldm(args, model, prompt_, layer_wise_prompt, modification_layers, info, noise_level=0.1):
    """ 
        model: ldm_stable
        prompt: Prompt for the diffusion model 
    """

    ################################ Definitions for diffusion model ################################
    # Define the hyper-parameters
    num_inference_steps = 50 
    guidance_scale = 7.5 
    # Height == width == 512
    height = width = 512 
    low_resource = False 
    global glob_seed
    
    generator = torch.Generator().manual_seed(glob_seed)
    latent = None 


    ############################## Definitions for causal tracing #########################
    # Prompt
    prompt = prompt_
    batch_size = len(prompt)
    
    
    # Text input 
    text_input = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0] # Shape: (1, 77, 768) # Batch-size = 1 (B, 77, 768) for prompts of size greater than one
    max_length = text_input.input_ids.shape[-1]

    #################################################################################################
    ##################################################################################################### 


    ###################################### Unconditional processing #####################################
    # Empty tokens for 
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    
    # Unconditional input
    # Unconditional embeddings
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    # Conditional / Unconditional Embeddings
    context = [uncond_embeddings, text_embeddings] # Set of 16 context vectors #
    
    # If not low_resource
    if not low_resource:
        context = torch.cat(context) # (6, 77, 768)

    
    # latent / Latents 
    latent, latents = init_latent(latent, model, height, width, generator, batch_size) # (1, 4, 64, 64)
    # Latents is expanded across the batch-sizes

    model.scheduler.set_timesteps(num_inference_steps)#, **extra_set_kwargs)
    
    # Total timesteps
    total_timesteps = model.scheduler.timesteps  # Tensor of shape [51]

    # Relevant Cross-Attention Layers which are responsible for the prompt injection
    relevant_layers = sorted(high_level_layers(model))

    # Stores the layerwise prompt embeddings of CLIP
    layerwise_prompt_embeddings = []

    # Iterate through
    for prompt_curr in layer_wise_prompt:
        text_input_curr= model.tokenizer(prompt_curr, padding="max_length", max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
        with torch.no_grad():
            text_embeddings = model.text_encoder(text_input_curr.input_ids.to(model.device))[0]
            layerwise_prompt_embeddings.append(text_embeddings[0])

    
    # Trace-States - Layerwise prompt embeddings
    trace_states(args, layerwise_prompt_embeddings, modification_layers, info, model, latents, context, guidance_scale, prompt_[0], low_resource)

    # Return the differences
    return 


def debug():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", default="style", type=str, required=False, help = "Attribute Label")
    parser.add_argument("--sd_version", default="2", type=str, required=False, help = "SD Label")
    parser.add_argument("--artist", default="van gogh", type=str, required=False, help = "Artist")
    parser.add_argument("--eval", default="True", type=str, required=False, help = "Label")
    parser.add_argument("--replace", default='False', type=str, required=False, help = "Replace operation")
    parser.add_argument("--device", default='cuda:0', type=str, required=False, help = "Cuda operation")
    parser.add_argument("--type", default='extensive', type=str, required=False, help = "Cuda operation")
    parser.add_argument("--edit", default='neuron', type=str, required=False, help = "Cuda operation")
    
    
    # Arguments
    args = parser.parse_args()

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token).to(device)
    tokenizer = ldm_stable.tokenizer

    global neurons_cnt, glob_seed
    
    info = torch.load(f'styles/v2-{args.artist}-{args.type}.pth')
    modification_layers = list(info.keys())
    test_prompts = tp[args.artist]
    
    print(f'Test Prompts: {test_prompts}')
    
    for prompt in test_prompts:
        
        print(f'Generating Images for Prompt: {prompt}')
        
        if args.edit == 'neuron':
            layer_wise_prompt = [prompt] * 16
        else:
            layer_wise_prompt = [prompt] * 8 + ['painting'] * 2 + [prompt] * 6
        
        
        prompts = [prompt]
        
        for seed in [0, 1, 2, 3]:
            
            print(f'\tSeed: {seed}')
            
            global glob_seed
            glob_seed = seed
            
            if args.edit == 'neuron':
                for cnt in [0, 30, 50, 100]:
                    
                    print(f'\t\t# of Neurons to edit: {cnt}')
                    
                    if cnt == 0:
                        modification_layers = []
                    else:
                        modification_layers = list(info.keys())
                    
                    global neurons_cnt
                    neurons_cnt = cnt
                    generate_image_ldm(args, ldm_stable, prompts, layer_wise_prompt, modification_layers, info) 
                
            else:
                
                print(f'\t\tLayer-wise modification')
                neurons_cnt = 0
                modification_layers = []
                
                generate_image_ldm(args, ldm_stable, prompts, layer_wise_prompt, modification_layers, info) 
                
                
                
if __name__ == "__main__":
    debug()
