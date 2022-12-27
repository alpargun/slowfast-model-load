
#%%

import torch
from torch.fx import symbolic_trace
from torch import nn

trained_backbone = torch.load("output/x3d-moco-bdd/baseline/checkpoints/trained-backbone-ep200.pt",map_location=torch.device('cpu'))
trained_backbone = trained_backbone.eval()
trained_backbone

# %%

input_tensor = torch.zeros(3, 4, 182, 182)

preds = trained_backbone(input_tensor[None, None, ...])
preds

# %%

trained_backbone.head.projection = nn.Identity()
trained_backbone.head.act = nn.Identity()
trained_backbone.head

# %% Add new MLP head

# Layers
lin1 = nn.Linear(in_features=2048, out_features=2048, bias=True)
lin1_relu = nn.ReLU(inplace=True)

lin2 = nn.Linear(in_features=2048, out_features=2048, bias=True)
lin2_relu = nn.ReLU(inplace=True)

lin3 = nn.Linear(in_features=2048, out_features=128, bias=True)
lin3_relu = nn.ReLU(inplace=True)

# Forward propagation

x = trained_backbone(input_tensor[None, None, ...])

x = lin1(x)
x = lin1_relu(x)

x = lin2(x)
x = lin2_relu(x)

x = lin3(x)
x = lin3_relu(x)

x


#%% PPO

num_actions = 1

linear3_mean = nn.Linear(in_features=128, out_features=num_actions, bias=True)
linear3_var = nn.Linear(in_features=128, out_features=num_actions, bias=True) 

tanh = nn.Tanh()
softplus = nn.Softplus()


mean = linear3_mean(x)
mean = tanh(mean)
print('shape of mean: ', mean)

var = linear3_var(x)
var = softplus(var)
print('shape of var: ', var)

x = torch.cat((mean, var), 1)
#x = torch.cat((mean, var))
# x = x.view(-1,1)

print("Shape of output in R50 actor:", x.shape)
# %%



# %%
from torch import nn
from collections import OrderedDict


x3d_head = nn.Sequential(OrderedDict([
    ('conv_5', nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)),
    ('conv_5_bn', nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
    ('conv_5_relu', nn.ReLU(inplace=True)),
    ('avg_pool', nn.AvgPool3d(kernel_size=[4, 5, 5], stride=1, padding=0)),
    ('lin_5', nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)),
    ('lin_5_relu', nn.ReLU(inplace=True)),
    #('projection', nn.Linear(in_features=2048, out_features=128, bias=True)),
    #('act', nn.Softmax(dim=4))
        ]))
x3d_head

# %%
