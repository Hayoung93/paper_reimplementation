import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from randaugment import RandAugment


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl-10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_name", default="efficientnet-b0")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--results_dir", type=str, default="/data/weights/hayoung/")
    parser.add_argument("--randaug", action="store_true")
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    return args


def main(args):
    stl10_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    stl10_unlabel_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            RandAugment(1, 2),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    stl10_tst_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.randaug == True:
        stl10_transform.transforms.insert(-3, RandAugment(1, 2))
    trainset = datasets.STL10(args.datadir, "train", transform=stl10_transform, download=True)
    valset = datasets.STL10(args.datadir, "test", transform=stl10_tst_transform, download=True)
    trainloader = DataLoader(trainset, args.batch_size, True, num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, args.batch_size, False, num_workers=0, pin_memory=True)
    unlabelset = datasets.STL10(args.datadir, "unlabeled", transform=transforms.PILToTensor(), download=True)
    unlabelloader = DataLoader(unlabelset, args.batch_size, True, num_workers=0, pin_memory=True)
    # trainunlabelset = datasets.STL10(args.datadir, "train+unlabeled", transform=stl10_transform, download=True)
    # trainunlabelloader = DataLoader(trainunlabelset, args.batch_size, True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_name(args.model_name)
    model._fc = nn.Linear(model._fc.in_features, args.num_classes)
    model.to(device)
    supcriterion = nn.CrossEntropyLoss()
    unsupcriterion = nn.KLDivLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    
    for ep in range(args.num_epochs):
        scheduler.step()
        print("Epoch {} --------------------------------------------".format(ep + 1))
        train_loss, train_acc, model, optimizer = \
            train(ep, model, trainloader, unlabelloader, stl10_unlabel_transform,
                  supcriterion, unsupcriterion, optimizer)
        print("Train loss: {}\tTrain acc: {}".format(train_loss, train_acc))
        val_loss, val_acc = eval(ep, model, valloader, supcriterion)
        print("Val loss: {}\tVal acc: {}".format(val_loss, val_acc))
        print("--------------------------------------------")
        # scheduler.step(val_loss)
        print("{}".format(optimizer.state_dict))
        if ep == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, True)
        save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, False)


def train(ep, model, suploader, unsuploader, unsuptransform, supcriterion, unsupcriterion, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    sup_generator = iter(suploader)
    for i, (inputs, _) in enumerate(tqdm(unsuploader)):
        optimizer.zero_grad()
        try:
            sup_inputs, labels = next(sup_generator)
        except StopIteration:
            sup_generator = iter(suploader)
            sup_inputs, labels = next(sup_generator)
        sup_inputs, labels = sup_inputs.cuda(), labels.cuda()
        unsup_inputs = inputs.cuda() / 255.
        unsup_aug_inputs = unsuptransform(inputs).cuda()

        # forward
        sup_outputs = model(sup_inputs)
        unsup_aug_outputs = model(unsup_aug_inputs)
        with torch.no_grad():
            unsup_outputs = model(unsup_inputs)

        # backward
        sup_loss = supcriterion(sup_outputs, labels)
        unsup_loss = unsupcriterion(unsup_outputs, unsup_aug_outputs)
        full_loss = sup_loss + unsup_loss  # lambda = 1
        train_loss += full_loss.item()
        full_loss.backward() 
        optimizer.step()

        train_acc += (sup_outputs.argmax(1) == labels).sum().item()
    train_loss /= len(unsuploader)
    train_acc /= len(unsuploader.dataset)
    return train_loss, train_acc, model, optimizer


def eval(ep, model, loader, criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            val_acc += (outputs.argmax(1) == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(loader)
    val_acc /= len(loader.dataset)
    return val_loss, val_acc


def save_checkpoint(ep, model, optimizer, scheduler, savepath, isbest):
    save_dict = {
        "epoch": ep,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if isbest:
        torch.save(save_dict, os.path.join(savepath, "model_best.pth"))
    else:
        torch.save(save_dict, os.path.join(savepath, "model_last.pth"))


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    main(args)