import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from os.path import join, exists
from yolot.val import SequenceValidator
from yolot.SequenceModel import SequenceModel
from yolot.BMOTSDataset import BMOTSDataset, single_batch_collate
from ultralytics.data.build import InfiniteDataLoader

canidate_paths = [
    join("weights", "best.pth"),
    join("weights", "best.pt"),
    join("weights", "last.pt"),
    join("mini_check.pt"),
]

known_configs = {
    'big_cell_loss_red':"yolot_gru_bign.yaml",
    'blank_check':      "yolotn.yaml",
    'empty_check':      "yolotn.yaml",
    'gru':              "yolot_grun.yaml",
    'gru_big_cells':    "yolot_gru_bign.yaml",
    'gru_seq16':        "yolot_gru_bign.yaml",
    'loss_fix':         "yolotn.yaml",
    'loss_fix2':        "yolotn.yaml",
    'loss_revert':      "yolotn.yaml",
    'low_lr':           "yolot_gru_bign.yaml",
    'med_gru':          "yolot_gru_bigm.yaml",
    'mpac100':          "yolotn.yaml",
}

def init_parser():
    parser = argparse.ArgumentParser(description="Find folders containing 'best.pt' in a specified path.")
    parser.add_argument("--path", type=str, default="/mnt/c/Users/dylan/Documents/Data/yolot_training_results/", help="The path to search for folders containing 'best.pt'.")
    parser.add_argument("--data", type=str, default="/mnt/c/Users/dylan/Documents/Data/BDD100k_MOT202/bdd100k", help="path to val data")
    parser.add_argument("--save", type=str, default="/mnt/c/Users/dylan/Documents/Data/yolot_training_results/", help="path to dave summary file")
    return parser

def main(args):
    # Read in existing data
    summary_path = join(args.save, "summary.csv")
    if exists(summary_path):
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame()
    # Check for existing runs
    existing_runs = list(summary.index)

    # Get subfolders
    subfolders = list_folders(args.path)

    # Filter out non-result folders
    runs = []
    for folder in subfolders:
        for canidate in canidate_paths:
            if exists(join(args.path, folder, canidate)):
                runs.append(
                    {
                        'name': folder,
                        'model_path': join(args.path, folder, canidate),
                        'last_path': join(args.path, folder, "weights", "last.pt")
                    }
                )
                break
    print(f"Found {len(runs)} runs in {args.path}")

    # Initialize Validation Data
    dataset = BMOTSDataset(args.data,"val", device=0)
    dataloader = InfiniteDataLoader(dataset, num_workers=4, batch_size=1, shuffle=False,
                                            collate_fn=single_batch_collate, drop_last=False, pin_memory=False)
    validator = SequenceValidator(dataloader=dataloader, save_dir=Path(join(args.path, "sum_val")))

    # For Each Run
    for run in runs:
        # Skip Existing Runs
        if run['name'] in existing_runs:
            print(f"! Skipping {run['name']} !")
            continue
        # Print Banner
        banner_text = f"=================================={run['name']}=================================="
        print("="*len(banner_text))
        print(banner_text)
        print("="*len(banner_text))
        # Build model
        try:
            model, metadata = build_model(run['model_path'], join("./cfg/models",known_configs[run['name']]))
            last_metadata = get_metadata(run['last_path']) if exists(run['last_path']) else {}
                
        except Exception as e:
            print(f"Failed to build model from {run['name']}: {e}")
            continue

        # Run validation
        model.eval()
        with torch.no_grad():
            metrics = validator(model=model, fuse=True, class_results=True)

        # Save Metrics to Database
        for k,v in metadata.items():
            metrics[f"meta/{k}"] = float(v)
        for k,v in last_metadata.items():
            metrics[f"meta_last/{k}"] = float(v)

        summary = update_dataframe(summary, metrics, run['name'])
    
    # Save Results
    print(f"Saving results to {summary_path}")
    summary.to_csv(summary_path)
        

def update_dataframe(df, data_dict, name):
    if name not in df.columns:
        df_temp = pd.Series(data_dict, name=name)
        df = df.append(df_temp)
    else:
        print(f"Val Already Recorded For {name}")
    return df

def list_folders(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folders

def filter_items(input_list, check_function):
    return [item for item in input_list if check_function(item)]

def build_model(path, config, device=0):
    model = SequenceModel(cfg=config, device=device, verbose=False)
    model.eval()
    model.model_to(device)
    ckpt = None
    if os.path.exists(path):
        print(f"Loading path from {path}")
        ckpt = torch.load(path)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt)
        if 'metadata' in ckpt:
            metadata = ckpt['metadata']
            print(ckpt['metadata'])
    else:
        ckpt = None
    print(f"Built parallel model_load with device: {torch.device(device)}")
    model.zero_states()
    return model, metadata

def get_metadata(path):
    metadata = {}
    if os.path.exists(path):
        ckpt = torch.load(path)
        if 'metadata' in ckpt:
            metadata = ckpt['metadata']
            print("Last: ", ckpt['metadata'])
    return metadata

if __name__ == "__main__":
    args = init_parser().parse_args()
    main(args)
