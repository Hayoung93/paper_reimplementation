import argparse
from torchvision import datasets
from efficientnet_pytorch import EfficientNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="efficientnet-b0")
    args = parser.parse_args()
    
    return args


def main(args):
    trainset = 
    pass


if __name__ == "__main__":
    args = get_args()
    main(args)