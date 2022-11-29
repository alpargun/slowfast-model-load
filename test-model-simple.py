#%%


import torch
from torch.fx import symbolic_trace


altered_backbone = torch.load("altered-backbone-ep200.pt")

# Set to GPU or CPU and call eval()
altered_backbone = altered_backbone.eval()
altered_backbone = altered_backbone.to('cpu')
altered_backbone

#%%
input_tensor = torch.zeros(3, 4, 182, 182)

preds = altered_backbone(input_tensor[None, None, ...])
preds[0].shape

# %%

trained_backbone = torch.load("trained-backbone-ep200.pt",map_location=torch.device('cpu'))
trained_backbone = trained_backbone.eval()
trained_backbone

#%%

trained_backbone_head = trained_backbone.head

trained_backbone.head = nn.Identity()

#%%
trained_backbone(input_tensor[None,None,...]).shape
# %%
