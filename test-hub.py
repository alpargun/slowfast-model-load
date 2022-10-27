

#%%
import torch
from torch.fx import symbolic_trace
from torch import nn


#%% Load x3d model

model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
model

#%%


device = 'cpu' #"cuda"
model = model.eval()
model = model.to(device)
model


#%% Remove last layers
# Dropout


model.blocks[5].dropout = nn.Identity()

model.blocks[5].proj = nn.Identity()


model.blocks[5].output_pool = nn.Identity()

model.blocks[5]

#%% Continue removing layers

model.blocks[5].pool = nn.Identity()





# %%

model(inputs[None, ...]).shape
# %%
