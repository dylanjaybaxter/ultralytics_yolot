import ultralytics
from ultralytics import YOLO
from ultralytics.nn.SequenceModel import SequenceModel
from ultralytics.nn.modules.rnn import RConv
import torch


model = SequenceModel(cfg="yolov8Tn.yaml" , device='cpu', verbose=True)
v8model = YOLO("yolov8n.pt")

# Get
source_layers = [layer for layer in v8model.model.modules()]
target_layers = [layer for layer in model.modules()]

offset = 0
RConv_sublayers = 3
rconv_count = 0
skip_rconv = False
global_layers = [0,1]
for i in range(len(target_layers)):
    if type(target_layers[i]) is not RConv and i not in global_layers and not skip_rconv:
        target_layers[i].load_state_dict(source_layers[i-offset].state_dict())
        print(f"Copied {sum(p.numel() for p in source_layers[i-offset].state_dict().values())} Parameters from {type(source_layers[i-offset])} to {type(target_layers[i])}")
    elif type(target_layers[i]) is RConv:
        print("Skipping RConv")
        skip_rconv = True
        rconv_count = RConv_sublayers + 1
    if skip_rconv:
        offset += 1
        rconv_count -= 1
        if rconv_count <= 0:
            skip_rconv = False


# Save Model
save_path = "yolot_pretrained.pt"
print(f"Saving Model as {save_path}")
torch.save(model.state_dict(), save_path)


print("Done!")
