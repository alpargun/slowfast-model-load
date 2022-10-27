
#%%
import torch
from torch.fx import symbolic_trace
from torch import nn


#%% Load x3d model

model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
print(model)

#%% 
model



# %% Remove dropout layer
model.blocks[5].dropout = nn.Identity()
model.blocks[5]

#%% Change linear projection dimension
model.blocks[5].proj = nn.Linear(in_features=2048, out_features=128, bias=True)
model.blocks[5]

#%% Remove Output pool
model.blocks[5].output_pool = nn.Identity()
model.blocks[5]

#%% Add Softmax activation function
model.blocks[5].act = nn.Softmax(dim=4)
model.blocks[5]

#%% Show updated block
model.blocks[5]

# %% Load checkpoint (2 GPUs)

PATH = '/home/alpargun/Desktop/cuda113/SlowFast/results/ssl/bdd/moco-x3d/32parts-batch64/checkpoints/ssl_checkpoint_epoch_00050.pyth'
#checkpoint = torch.load(PATH)

# Load state
model.load_state_dict(torch.load(PATH), strict=True)












# %% Checkpoint with 1 GPU

PATH = '/home/alpargun/Desktop/cuda113/SlowFast/results/ssl/bdd/moco-x3d/39parts-1gpu/checkpoints/ssl_checkpoint_epoch_00001.pyth'

checkpoint = torch.load(PATH)
checkpoint


# %%
