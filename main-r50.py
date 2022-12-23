# Load SlowFast checkpoint, build the model nad load the weights.


#%%
#from slowfast import models
import torch
from torch.fx import symbolic_trace
from torch import nn

from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
import sys
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
import slowfast.models.optimizer as optim

import os


MODEL_NAME = 'r50_moco'

path_to_config = 'trained-models/r50-moco-with_bdd/finetune_bdd_pretrained_on_k400_MoCo_Slow.yaml'
path_checkpoint = 'trained-models/r50-moco-with_bdd/ssl_checkpoint_epoch_00040.pyth'

sys.argv = [f'--cfg={path_to_config}',]

print(sys.argv)

args = parse_args()


cfg = load_config(args, path_to_config)
cfg = assert_and_infer_cfg(cfg)


# Load the slowfast model based on the config

model_sf = build_model(cfg)

# Config optimizer

optimizer = optim.construct_optimizer(model_sf, cfg)

# Config scaler

scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

# Load checkpoint

checkpoint_epoch = cu.load_checkpoint(
                path_checkpoint,
                model_sf,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

checkpoint_epoch += 1
print('epoch: ', checkpoint_epoch)

# Check state_dict

print(model_sf.state_dict())


# Save model
path_extension = MODEL_NAME + "-epoch" + str(checkpoint_epoch) + ".pt"
path_complete_model = "complete-model-" + path_extension
torch.save(model_sf, path_complete_model)

#%%

# Get the backbone and remove the head

backbone_r50 = model_sf.backbone

# Save original backbone
path_backbone = "trained-backbone-" + path_extension
torch.save(backbone_r50, path_backbone)

#%%

# Modify the head

# backbone_r50.head = nn.Identity()

# print('head: ', backbone_r50.head)

# # Save altered model
# path_altered_backbone = "altered-backbone-" + path_extension
# torch.save(backbone, path_altered_backbone)

print('finished')

# %%
# %%

# propagate through the backbone

from torch import nn 

input_tensor = torch.zeros(3, 8, 256, 256)

preds = backbone_r50(input_tensor[None, None, ...])
preds 

#%%


linear1 = nn.Linear(in_features=128, out_features=64, bias=True)
linear2 = nn.Linear(in_features=64, out_features=32, bias=True)
linear3_mean = nn.Linear(in_features=32, out_features=1, bias=True)
linear3_var = nn.Linear(in_features=32, out_features=1, bias=True) 

flatten = nn.Flatten()
tanh = nn.Tanh()
softplus = nn.Softplus()

x = preds
print('shape of x before applying actor and critic: ', x.shape)
x = linear1(x)
print('after lin1: ', x.shape)

#%%
x = linear2(x)
print('after lin2: ', x.shape)

#%%
print('shape of x before mean and var: ', x.shape)

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
x
# %%
