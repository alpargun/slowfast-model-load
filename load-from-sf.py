
#%%
#from slowfast import models
import torch
from torch.fx import symbolic_trace

from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
import sys


#%%

sys.argv = ['--cfg=/home/alpargun/Desktop/good-sf-models/39parts-1gpu-8workers/BDD_MoCo_x3d.yaml',]

print(sys.argv)

args = parse_args()

path_to_config = '/home/alpargun/Desktop/good-sf-models/39parts-1gpu-8workers/BDD_MoCo_x3d.yaml'


#%%

cfg = load_config(args, path_to_config)
cfg = assert_and_infer_cfg(cfg)
cfg


#%% Load the slowfast model based on the config
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model


model_sf = build_model(cfg)
model_sf

#%% Save initialized model

torch.save(model_sf, "moco-with-x3d-xs-init.pt")

#%% Config optimizer

import slowfast.models.optimizer as optim


optimizer = optim.construct_optimizer(model_sf, cfg)
optimizer

#%% Config scaler

scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
scaler

#%% Load checkpoint

PATH = '/home/alpargun/Desktop/good-sf-models/39parts-1gpu-8workers/checkpoints/ssl_checkpoint_epoch_00200.pyth'

checkpoint_epoch = cu.load_checkpoint(
                PATH,
                model_sf,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

checkpoint_epoch += 1

checkpoint_epoch

#%% Check state_dict

model_sf.state_dict()


#%% Save model
import torch
from torch.fx import symbolic_trace


torch.save(model_sf, "complete-model-bdd-moco-epoch200.pt")


#%% Restore the saved model
import torch
from torch.fx import symbolic_trace


model_new = torch.load("complete-model-bdd-moco-epoch200.pt")
model_new

#%% Check state_dict (weights)
model_new.state_dict()


