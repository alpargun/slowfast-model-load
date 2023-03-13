#%%
import torch
#from torch.fx import symbolic_trace
from torch import nn


#%% Load x3d model

model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True)
print(model)

#%%

# Set to GPU or CPU
device = "cpu"
model = model.eval()
model = model.to(device)

input_tensor = torch.zeros(3, 4, 128, 182)

preds = model(input_tensor[None, ...])
preds.shape


# %% Summary for X3D XS

from torchsummary import summary

summary(model, input_tensor.shape)


# %% Summary for X3D M

model_m = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)

#%% Summary of X3D_M

inputs_m = torch.zeros(3, 16, 182, 182)
summary(model_m, inputs_m.shape)


