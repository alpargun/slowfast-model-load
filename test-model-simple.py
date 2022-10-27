#%%

import torch


altered_backbone = torch.load("altered-backbone-ep200.pt")

# Set to GPU or CPU and call eval()
altered_backbone = altered_backbone.eval()
altered_backbone = altered_backbone.to('cpu')

input_tensor = torch.zeros(3, 4, 182, 182)

preds = altered_backbone(input_tensor[None, None, ...])
preds[0].shape

# %%
