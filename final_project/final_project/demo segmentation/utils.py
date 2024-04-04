import torch
import torchvision
from dataset import WaterDataset
from torch.utils.data import DataLoader
from collections import OrderedDict


def save_checkpoint(state, filename="./checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def load_checkpoint_multigpu(checkpoint, model):
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if 'module.' in k else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def load_optim(checkpoint, optimizer):
    optimizer.load_state_dict(checkpoint['optimizer'])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = WaterDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    val_ds = WaterDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader



def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    acc = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    acc = num_correct / num_pixels * 100
    print(f"Got {num_correct}/{num_pixels} with acc {acc:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return acc


def save_predictions_as_imgs(loader, model, folder="", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

    model.train()