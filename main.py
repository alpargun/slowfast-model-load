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


MODEL_NAME = 'finetune_carla-x3d_xs_moco'

path_to_config = 'trained-models/finetune-carla/finetune_Carla_MoCo_x3d.yaml'
path_checkpoint = 'trained-models/finetune-carla/ssl_checkpoint_epoch_01000.pyth'

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


# Get the backbone and remove the head

backbone = model_sf.backbone

# Save original backbone
path_backbone = "trained-backbone-" + path_extension
torch.save(backbone, path_backbone)

# Modify the head

backbone.head = nn.Identity()

print('head: ', backbone.head)

# Save altered model
path_altered_backbone = "altered-backbone-" + path_extension
torch.save(backbone, path_altered_backbone)

# %%
