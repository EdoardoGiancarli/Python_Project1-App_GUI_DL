# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 01:51:01 2023

@author: edosc
"""

class _model_insights:
    """
    Model features analysis.
    """
    
    def __init__(self, model):
        
        self.model = model

    def _take_modules(self, model_attribute):
        """
        List of the modules in the input model attribute.
        """
        
        out_modules = [mod for mod in model_attribute]
        
        return out_modules
    
    def _take_parameters(self, model_module, to_numpy = True):
        """
        List of the parameters in the input model module.
        """
        
        out_params = [list(mod.parameters()) for mod in model_module]
        
        if to_numpy:
            out_params = [param.detach().cpu().numpy() for param in out_params]
        
        return out_params
    
    def _show_kernels(self, module_params, channel_kernel=0, save_fig=False, directory=None):
        """
        Plot of the convolutional kernels.
        """
        
        for i in module_params:
            
            plt.figure(None, tight_layout=True)
            
            if len(i.shape) == 4:
                a = plt.imshow(i[channel_kernel, channel_kernel, :, :], cmap='Greys', vmin=-1, vmax=1)
                title = f'Conv. kernel {i.shape}, channel {channel_kernel}'
                
            plt.colorbar(a)
            plt.title(title)
            
            if save_fig:
                if directory is None:
                    raise ValueError("specify directory to save the kernels images.")
                
                plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
                plt.close()
            
            else:
                plt.show()


