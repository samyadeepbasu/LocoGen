"""  
Script to implement the hooking function for nethook
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
import contextlib
import copy
import inspect
from collections import OrderedDict

# Function to get the module 
def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


# Trace Class -- which performs the operation over only one layer
class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """
    # initialisation function
    def __init__(
        self,
        module,
        layer=None,
        text_embedding = None, 
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        
        # Retainer
        retainer = self
        self.layer = layer

        # This is full module 
        if layer is not None:
            # Get the submodule from the layer
            module = get_module(module.unet, layer) #Extract the module # Embedding(50257, 768) # Module of the layer
            # Module
            #print(f'Module corresponding to layer: {layer} is {module}')
        
        #print(f'Shape of Text Embedding : {text_embedding.shape}')

        # Retain hook --- Forward hook function
        def retain_hook(m, inputs, output):
            #print(f'Into the retain hook function')
            # Replace the corrupted embedding with the uncorrupted one for the state [position-2]
            """ Output replacement operation """
            input_to_ca = inputs[0] # (2, 77, 768)

            #print(type(input_to_ca))
            weight = torch.tensor(list(m.parameters())[0])

            #output_modified = torch.matmul(input_to_ca[1].reshape(-1, 768), torch.transpose(weight, 0, 1)) 
            output_modified = torch.matmul(text_embedding, torch.transpose(weight, 0, 1)) 

            # Update in the conditional output prompt
            output[1] = output_modified
            
            # Stop function
            if stop:
                #print(f'######## Stop operation #########')
                raise StopForward()
            # output
            return output
        

        # Registering the hook for the given module which is representative of the layer
        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

        return 
    
    
    # Enter()
    def __enter__(self):
        #print(f'Enter Function  #########')
        return self

    # Exit()
    def __exit__(self, type, value, traceback):
        #print(f'Exit Function ##########')
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    # Close by removing hook
    def close(self):
        #print(f'Close Function #########')
        self.registered_hook.remove()



#  Function to define the Trace Dictionary which will track and perform the replacement operation
class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    # 
    def __init__(
        self,
        module,
        layers=None,
        layerwise_prompt = None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):  
        # Define the housekeeping variables
        self.stop = stop

        # Flag-Last-Unseen
        def flag_last_unseen(it):
            #print(f'## {it}')
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

    
        # Layerwise Prompt;
        
        # Can take layers as a list of layers or one layer specifically
        if type(layers) == list:
            #print("#### Into trace #######")
            # Iterate 
            c = 0
            for is_last, layer in flag_last_unseen(layers):
                #print(f'Layer: {layer}; Is_last: {is_last}')
                self[layer] = Trace(
                    module=module,
                    layer=layer, 
                    text_embedding = layerwise_prompt[int(c/2)], 
                    retain_output=retain_output,
                    retain_input=retain_input,
                    clone=clone,
                    detach=detach,
                    retain_grad=retain_grad,
                    edit_output=edit_output,
                    stop=stop and is_last,
                )

                c += 1
            
            # Total count  : c 
            #print(f'Total Count : {c}')


        # Else function
        else:
            #print(f' ###### Into trace function ##### ')
            layer = layers
            # Only one element is present
            self[layer] = Trace(
                module=module,
                layer=layer, 
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )


        # Return 
        return 
    

    # Enter
    def __enter__(self):
        return self

    # Exit function
    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
    # 
    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()



# Forward
class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass
