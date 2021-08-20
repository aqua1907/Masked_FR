import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from nets.iresnet import iresnet34


model = iresnet34()
ckpt = torch.load('iresnet34_arc_0745.pt')
model.load_state_dict(ckpt)
model.eval()
model.to('cuda')

img = torch.randn(1, 3, 128, 128).to('cuda')
model = torch.quantization.convert(model)
print(model.cls)
scripted = torch.jit.trace(model, img, strict=False)
opt_model = optimize_for_mobile(scripted)
opt_model._save_for_lite_interpreter('iresnet34_arc_0745.ptl')




