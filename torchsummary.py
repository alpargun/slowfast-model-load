
#%%

pytorch_total_params = sum(p.numel() for p in head_net.parameters() if p.requires_grad)
pytorch_total_params






# %%
x3d_head_total_params = sum(p.numel() for p in x3d_head.parameters() if p.requires_grad)
x3d_head_total_params