
#%%
#from slowfast import models
import torch
from torch.fx import symbolic_trace

from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
import sys


#%%

MODEL_NAME = 'finetune_carla-x3d_xs_moco'
path_to_config = 'trained-models/finetune-carla/finetune_Carla_MoCo_x3d.yaml'
path_checkpoint = 'trained-models/finetune-carla/ssl_checkpoint_epoch_01000.pyth'

sys.argv = [f'--cfg={path_to_config}',]

print(sys.argv)

args = parse_args()




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

# torch.save(model_sf, "x3d_xs-moco-finetune_carla.pt")

#%% Config optimizer

import slowfast.models.optimizer as optim


optimizer = optim.construct_optimizer(model_sf, cfg)
optimizer

#%% Config scaler

scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
scaler

#%% Load checkpoint

#PATH = 'trained-models/finetune-carla/ssl_checkpoint_epoch_01000.pyth'

checkpoint_epoch = cu.load_checkpoint(
                path_checkpoint,
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

path_complete_model = "complete-model-" + MODEL_NAME + "-epoch" + str(checkpoint_epoch) + ".pt"
torch.save(model_sf, path_complete_model)


#%% Restore the saved model
import torch
from torch.fx import symbolic_trace


model_new = torch.load(path_complete_model)
model_new

#%% Check state_dict (weights)
model_new.state_dict()



# %%
