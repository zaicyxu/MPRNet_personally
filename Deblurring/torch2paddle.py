from MPRNet import MPRNet
from collections import OrderedDict
from x2paddle.convert import pytorch2paddle
import numpy as np
import torch

ckpt = r'C:\projects\deblur\ckpt\mpr_net\model_deblurring.pth'


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

torch_module = MPRNet()
load_checkpoint(torch_module, ckpt)
# torch_module.load_state_dict(torch.load(ckpt), strict=False)
# 设置为eval模式
torch_module.eval()

input_data = np.random.rand(1, 3, 256, 256).astype("float32")

pytorch2paddle(torch_module,
               save_dir=r'C:\projects\deblur\ckpt\mpr_net\paddle_new',
               jit_type="trace",
               input_examples=[torch.tensor(input_data)])
