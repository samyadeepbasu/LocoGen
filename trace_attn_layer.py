""" 
Research @ Feizi Lab - University of Maryland, College Park

This script extracts the set of cross-attention layers in diffusion models which stores certain visual concepts (e.g., style / objects / facts)

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
import nethook_diffusion 
#import open_clip
import pickle 
import os 

# Define the Components of the Diffusion Model 
auth_token = "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"
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


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



# Diffusion step
def diffusion_step(model, latents, relevant_layers, layer_wise_prompt, context, t, guidance_scale, layer_, low_resource=False, orig = False, window = False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        # latents: (1, 4, 64, 64)
        latents_input = torch.cat([latents] * 2) # (2*B, 4, 64, 64) ; For 2 prompts -- (4, 4, 64, 64) with the first 2 uncodnitional and the rest conditional

        # Original --> False 
        if orig == False:
            #print(f'Final step')
            with torch.no_grad(), nethook_diffusion.TraceDict(model, relevant_layers, layer_wise_prompt):
                # The trace function will compose of [layer1, layer2, ......, layer16] prompt-embeddings;
                # Given the particular layer -> it will use the transformation function to get the output of the W_{k}, W_{v} as the transformed embedding
                noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"] 
        
        else:
            #print(f'Final step')
            with torch.no_grad():
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
def trace_patch(model, latents, relevant_layers, layer_wise_prompt, context, guidance_scale, low_resource, orig=False, layer_= None):
    # Iterate through the diffusion step
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, latents, relevant_layers, layer_wise_prompt, context, t, guidance_scale, layer_, low_resource, orig)
    
    # Convert from latents to diffusion step
    image = latent2image(model.vae, latents)

    # Image
    return image


# Layerwise prompts where the prompt in a small set of layers are replaced with a target prompt # 
def compute_layer_wise_prompts(args, index):
    # Define the prompt template
    prompt_template = [args.og_prompt]*16

    # Original Index
    prompt_template[index] = args.prompt 

    # Update the second cross-attn layer
    if index + 2 <16:
        # One position right
        prompt_template[index+1] = args.prompt 
        prompt_template[index+2] = args.prompt 

    return prompt_template 


""" 
Define the trace-state function which iterates over various generations to obtain the generated latent 
"""
def trace_states(args, layer_wise_prompt, model, latents, context, guidance_scale, prompt_curr, index, low_resource, orig):
    # Define the layers one needs to pass
    relevant_layers = sorted(high_level_layers(model))

    # Save Path
    save_path = '/fs/nexus-scratch/sbasu12/projects/cross_edit/trace_results' + '/sd_version_'  + args.sd_version + '_seed_' + str(args.seed) + '/' + args.og_prompt
    isExist = os.path.exists(save_path)

    # Not exist
    if not isExist:
        os.makedirs(save_path)
    
    # Trace Patch
    image = trace_patch(model, latents, relevant_layers, layer_wise_prompt, context, guidance_scale, low_resource, orig) # Shape[3, 3, 512, 512] # (0th index: Clean; 1th index: Corrupted; 2nd index: Restored state)

    # Orig == False
    if orig == False:
        ############# Saving the image where a layer is perturbed ############
        image_ = image[0] #.reshape((3,512,512))
        im1 = Image.fromarray(image_)
        im1.save(save_path + '/orig_' + str(index) + '.png')
    
    # Save the original image
    else:
        print(f'######## Saving the original Image #########')
        image_ = image[0] #.reshape((3,512,512))
        im1 = Image.fromarray(image_)
        im1.save(save_path + '/orig' + '.png')
    
    
    return 



# Generate image ldm;
@torch.no_grad()
def generate_image_ldm(args, model, prompt_, layer_wise_prompt, index, orig=False, noise_level=0.1):
    """ 
        model: ldm_stable
        prompt_: Target prompt for the diffusion model
        layer_wise_prompt: Prompts where a particular cross-attention layer has been perturbed 
        index: Index position which is perturbed 
    """

    ################################ Definitions for diffusion model ################################
    # Define the hyper-parameters
    num_inference_steps = 50 
    guidance_scale = 7.5 
    # Height == width == 512
    height = width = 512 
    low_resource = False 
    generator = torch.Generator().manual_seed(args.seed)
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
    
    # Unconditional embeddings
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    #print(text_embeddings.shape) (B, 77, 768)
    
    # Conditional / Unconditional Embeddings
    context = [uncond_embeddings, text_embeddings] # Set of 16 context vectors #
    
    # If not low_resource
    if not low_resource:
        context = torch.cat(context) # (6, 77, 768)


    # latent / Latents 
    latent, latents = init_latent(latent, model, height, width, generator, batch_size) # (1, 4, 64, 64)
    # Latents is expanded across the batch-sizes

    #extra_set_kwargs = {"offset": 1}
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
    trace_states(args, layerwise_prompt_embeddings, model, latents, context, guidance_scale, prompt_[0], index, low_resource, orig)

    # Return the differences
    return 


# Function to trace the cross-attention layers which encode relevant visual concepts # 
def trace_ca_layers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute", default="style", type=str, required=False, help = "Attribute Label")
    parser.add_argument("--sd_version", default="1", type=str, required=False, help = "SD Label")
    parser.add_argument("--eval", default="True", type=str, required=False, help = "Label")
    parser.add_argument("--prompt", default="a painting", type=str, required=False, help = "Target Prompt")
    parser.add_argument("--og_prompt", default="a beautiful house in the style of van gogh", type=str, required=False, help = "Prompt which is used in all layers except the ones which are replaced")
    parser.add_argument("--replace", default='False', type=str, required=False, help = "Replace operation")
    parser.add_argument("--device", default='cuda:0', type=str, required=False, help = "Cuda operation")
    parser.add_argument("--seed", default=0, type=int, required=False, help = "Cuda operation")

    # Arguments
    args = parser.parse_args()

    # Define the device
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')

    # SD-Version 1.4
    if args.sd_version == '1':
        ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=auth_token, cache_dir = '/fs/nexus-scratch/sbasu12/projects/cross_edit').to(device)
    
    # SD-Version 2.1
    elif args.sd_version == '2':
        ldm_stable = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", use_auth_token=auth_token, cache_dir = '/fs/nexus-scratch/sbasu12/projects/cross_edit').to(device)
    
    # Prompt-hero
    elif args.sd_version == 'prompthero':
        ldm_stable = StableDiffusionPipeline.from_pretrained("prompthero/openjourney", use_auth_token=auth_token, cache_dir = '/fs/nexus-scratch/sbasu12/projects/cross_edit').to(device)

    # Tokenizer
    tokenizer = ldm_stable.tokenizer

    # Relevant layers # 
    relevant_layers = sorted(high_level_layers(ldm_stable))

    # Total number of cross-attention layers # 
    num_layers = 16

    # Iterate through the layers and generate the images corresponding to the perturbation operation # 
    for j in range(0, num_layers):
        # Obtain the layerwise prompt # 
        layer_wise_prompt = compute_layer_wise_prompts(args, j)
 
        # Target Prompt
        prompts = [args.prompt]
    
        # Generate image by perturbing the layer corresponding to jth position # 
        generate_image_ldm(args, ldm_stable, prompts, layer_wise_prompt, j)



    # Generate original image # 
    prompts = [args.og_prompt]
    layer_wise_prompt = [args.og_prompt]*16
    generate_image_ldm(args, ldm_stable, prompts, layer_wise_prompt, 0, orig=True)
    


# Main function
if __name__ == "__main__":
    # Primary function which calls for the debug option
    trace_ca_layers()
