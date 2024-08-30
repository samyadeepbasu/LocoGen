## On Mechanistic Knowledge Localization for Text-to-Image Generative Models

Our method can remove copyrighted materials (e.g., style, objects) from pretrained Stable Diffusion models and also update them with newer facts (e.g., President of a country). 

#### Implementation of ICML 2024 Paper: "On Mechanistic Localization in Text-to-Image Generative Models" : https://arxiv.org/abs/2405.01008

###### Script for tracing the concept in the cross-attention layers 
``` python trace_attn_layer.py --og_prompt <Insert prompt containing the visual concept (e.g., van gogh style)> --prompt <Insert prompt containing the target concept (e.g., painting)>```


###### Script for editing the layers in diffusion models 
``` Coming soon! ```


###### Script for editing neurons in diffusion models
``` Coming soon! ```



## Introduction

Identifying layers within text-to-image models which control visual attributes can facilitate efficient model editing through closed-form updates. Recent work, leveraging causal tracing show that early Stable-Diffusion variants confine knowledge primarily to the first layer of the CLIP text-encoder, while it diffuses throughout the UNet.Extending this framework, we observe that for recent models (e.g., SD-XL, DeepFloyd), causal tracing fails in pinpointing localized knowledge, highlighting challenges in model editing. To address this issue, we introduce the concept of Mechanistic Localization in text-to-image models, where knowledge about various visual attributes (e.g., "style", "objects", "facts") can be mechanistically localized to a small fraction of layers in the UNet, thus facilitating efficient model editing. We localize knowledge using our method LocoGen which measures the direct effect of intermediate layers to output generation by performing interventions in the cross-attention layers of the UNet. We then employ LocoEdit, a fast closed-form editing method across popular open-source text-to-image models (including the latest SD-XL)and explore the possibilities of neuron-level model editing. Using Mechanistic Localization, our work offers a better view of successes and failures in localization-based text-to-image model editing


## Citation
If you use this code for your research, please cite our paper.

```bibtex
@misc{basu2024mechanisticknowledgelocalizationtexttoimage,
      title={On Mechanistic Knowledge Localization in Text-to-Image Generative Models}, 
      author={Samyadeep Basu and Keivan Rezaei and Priyatham Kattakinda and Ryan Rossi and Cherry Zhao and Vlad Morariu and Varun Manjunatha and Soheil Feizi},
      year={2024},
      eprint={2405.01008},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.01008},
}

```
