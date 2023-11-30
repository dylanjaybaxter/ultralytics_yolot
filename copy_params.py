import ultralytics
from ultralytics import YOLO
from ultralytics.nn.SequenceModel import SequenceModel
from ultralytics.nn.modules.rnn import RConv


model = SequenceModel(cfg="yolov8Tn.yaml" , device='cpu', verbose=True)
v8model = YOLO("yolov8n.pt")

# Get
source_layers = [layer for layer in v8model.model.modules()]
target_layers = [layer for layer in model.modules()]


offset = 0
global_layers = [0,1]
for i in range(len(target_layers)):
    if type(target_layers[i]) is not RConv and i not in global_layers:
        target_layers[i].load_state_dict(source_layers[i-offset].state_dict())
        print(f"Copied {sum(p.numel() for p in source_layers[i].state_dict().values())} Parameters from {type(source_layers[i-offset])} to {type(target_layers[i])}")
    elif type(target_layers[i]) is RConv:
        print("Skipping RConv")
        offset += 1



print("Done!")
