#%%

from torch import nn
from torch.fx import symbolic_trace



altered_backbone = torch.load("altered-backbone-ep200.pt")
altered_backbone


# %% Build head

conv_5 = nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
conv_5_bn = nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
conv_5_relu = nn.ReLU(inplace=True)

avg_pool = nn.AvgPool3d(kernel_size=[4, 5, 5], stride=1, padding=0)
lin_5 = nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
lin_5_relu = nn.ReLU(inplace=True)
projection = nn.Linear(in_features=2048, out_features=128, bias=True)



# %%
from collections import OrderedDict


x3d_head = nn.Sequential(OrderedDict([
        ('conv_5', nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)),
        ('conv_5_bn', nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('conv_5_relu', nn.ReLU(inplace=True)),

        ('avg_pool', nn.AvgPool3d(kernel_size=[4, 5, 5], stride=1, padding=0)),

        ('lin_5', nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)),
        ('lin_5_relu', nn.ReLU(inplace=True)),
        
        ('projection', nn.Linear(in_features=2048, out_features=128, bias=True)),

    ])
)

x3d_head
# %%

model_hub = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
model_hub

# %%
x3d_head_hub = model_hub.blocks[5]
# %%

input_head = torch.zeros(192, 4, 6, 6)
x3d_head_hub(input_head[None, ...]).shape


# %%
