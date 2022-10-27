
#%% Import remaining functions

import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)



#%% Define input transform

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params  = {
    "x3d_xs": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 4,
        "sampling_rate": 12,
    },
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

#%% Get transform parameters based on model

model_name = 'x3d_xs'

transform_params = model_transform_params[model_name]

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(transform_params["crop_size"], transform_params["crop_size"])
            )
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second





# %% Download an example video

url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
video_path = 'archery.mp4'
try: urllib.URLopener().retrieve(url_link, video_path)
except: urllib.request.urlretrieve(url_link, video_path)

#%% Load the video and transform it to the input format required by the model.

# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class and load the video
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
video_data



#%% Load the altered backbone
import torch
from torch.fx import symbolic_trace


altered_backbone = torch.load("altered-backbone-ep200.pt")
altered_backbone

# %% Set to GPU or CPU and call eval()

device = 'cpu' #"cuda"
altered_backbone = altered_backbone.eval()
altered_backbone = altered_backbone.to(device)
altered_backbone


#%% Obtain inputs

# Apply a transform to normalize the video input
video_data = transform(video_data)

# Move the inputs to the desired device
inputs = video_data["video"]
inputs = inputs.to(device)


#%% Get predictions

# Pass the input clip through the model
preds = altered_backbone(inputs[None, None, ...])
preds


#%% Show preds dim
preds[0].shape


#%% Reshape tensor

preds_reshaped = preds[0].reshape([1, 192, 12, 12])
preds_reshaped.shape


#%% Squeeze to obtain 3d

preds_squeezed = preds_reshaped.squeeze()
preds_squeezed.shape


#%% 









# Other operations - UNNECESSARY

# --------------------------------------------------------------

#%%
# Get the predicted classes
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices[0]
pred_classes

# %% Show model summary

from torchsummary import summary

x = torch.randn(4, 3, 182, 182)

summary(altered_backbone, x)
# %%
