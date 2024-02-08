# Imports
from yolot.yolot_trainer import YolotTrainer
import argparse
import yaml

# Parameter Parsing
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/yolot_config.yaml", help="Path to training config")
    return parser

def main_func(args):
    with open(args.config, 'r') as conf_file:
        conf = yaml.safe_load(conf_file)
    trainer = YolotTrainer(cfg=conf)
    trainer.train_model()

    print("Done!")

if __name__ == '__main__':
    args = init_parser().parse_args()
    main_func(args)



