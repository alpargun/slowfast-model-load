# Load the saved torch model, alter the layers and test the model using an input video

#%% Restore the saved model
import torch
from torch.fx import symbolic_trace


model_path = 'complete-model-bdd-moco-epoch200.pt'

model_complete = torch.load(model_path)
model_complete

#%% Check state_dict (weights)
model_complete.state_dict()


#%% Get only the backbone
model_backbone = model_complete.backbone
model_backbone

#%% Save original backbone
import torch
from torch.fx import symbolic_trace


torch.save(model_backbone, "trained-backbone-ep200.pt")


#%% Load the original backbone

import torch
from torch.fx import symbolic_trace


altered_backbone = torch.load("trained-backbone-ep200.pt")
altered_backbone


#%% Modify the head

from torch import nn


#Conv
#original_backbone.head.conv_5 = nn.Identity()
##original_backbone.head.conv_5_bn = nn.Identity()
#original_backbone.head.conv_5_relu = nn.Identity()


# AvgPool3d
#original_backbone.head.avg_pool = nn.Identity()

# Lin5
#original_backbone.head.lin_5 = nn.Identity()
#original_backbone.head.lin_5_relu = nn.Identity()

# Projection
#original_backbone.head.projection = nn.Identity()
#original_backbone.head.act = nn.Identity()

altered_backbone.head = nn.Identity()

altered_backbone.head


#%% Save altered model
import torch
from torch.fx import symbolic_trace


torch.save(altered_backbone, "altered-backbone-ep200.pt")



# %%
