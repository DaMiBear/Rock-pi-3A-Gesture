import torch
from model import Lite_HRNet_SimCC

input_size = [128, 128]
model_name = "Lite_HRNet_SimCC"
net = eval(model_name)(num_joints=21, input_size=input_size, simcc_split_ratio=2.0, export_ncnn=True)
net.load_state_dict(torch.load("", map_location='cpu')['model'])    # 模型权重文件
net = net.eval()

x = torch.rand(1, 3, input_size[0], input_size[1])

mod = torch.jit.trace(net, x)

mod.save(f"./ncnn_dir/{model_name}-{input_size[0]}x{input_size[1]}.pt")