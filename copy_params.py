import ultralytics
from ultralytics import YOLO
from ultralytics.nn.SequenceModel import SequenceModel
from ultralytics.nn.modules.rnn import RConv, ConvGRU
import torch


model = SequenceModel(cfg="yolov8Tn_GRU.yaml" , device='cpu', verbose=True)
v8model = YOLO("yolov8n.pt")

# Get
source_layers = [layer for layer in v8model.model.modules()]
target_layers = [layer for layer in model.modules()]

offset = 0
RConv_sublayers = 3
GRU_sublayers = 4
rconv_count = 0
skip_rconv = False
global_layers = [0,1]
for i in range(len(target_layers)):
    print(f"Comparing Parameters from {type(source_layers[i-offset])} to {type(target_layers[i])}")
    if type(target_layers[i]) is RConv:
        print("Skipping RConv")
        skip_rconv = True
        skip_count = RConv_sublayers + 1
    elif type(target_layers[i]) is ConvGRU:
        skip_rconv = True
        skip_count = GRU_sublayers + 1
    elif i not in global_layers and not skip_rconv:
        target_layers[i].load_state_dict(source_layers[i-offset].state_dict())
        print(f"\tCopied {sum(p.numel() for p in source_layers[i-offset].state_dict().values())}")
    if skip_rconv:
        offset += 1
        skip_count -= 1
        if skip_count <= 0:
            skip_rconv = False


# Save Model
save_path = "yolot_pretrained_GRU_big.pt"
print(f"Saving Model as {save_path}")
torch.save(model.state_dict(), save_path)


print("Done!")
