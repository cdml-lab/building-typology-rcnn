"""Training, as configured for the released checkpoint (fasterrcnnv100-12).

Training tiles: split "Data" of every area, both imagery sources (GOV and
ESRI); validation tiles: split "Unseen", GOV only. Augmentation: colour
jitter (p = 0.5), Gaussian blur (p = 0.1) and, for every tile, one extra
copy rotated by a random angle. SGD (lr 0.01, momentum 0.9, weight decay
5e-4), OneCycle schedule (max lr 0.01, 20 % warm-up, cosine), batch 16,
mixed precision, up to 30 epochs, early stopping after 8 epochs without
improvement of the overall weighted F1 on the validation tiles; the best
epoch is kept. Losses: the standard torchvision Faster R-CNN losses
(no class weighting).
"""
from __future__ import annotations

from pathlib import Path

from .dataset import TileDataset, collate, collect_paths
from .evaluate import evaluate
from .model import build_model


def train(root, out_path, areas=None, epochs=30, patience=8, batch_size=16, workers=4,
          lr=0.01, seed=42, log=print):
    import torch
    from torch.amp import GradScaler, autocast
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader
    from torchvision import transforms
    torch.manual_seed(seed)
    tf_train = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.Resize((512, 512)), transforms.ToTensor()])
    imgs, labs = collect_paths(root, areas, ("GOV", "ESRI"), "Data")
    ds = TileDataset(imgs, labs, tf_train, random_rotate=True, seed=seed)
    log(f"[train] {len(ds)} training samples ({ds.dropped_tiles} tiles dropped)")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                    num_workers=workers, pin_memory=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("resnet101", pretrained_backbone=True).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(dl), epochs=epochs, pct_start=0.2,
                       anneal_strategy="cos")
    scaler = GradScaler(dev.type)
    best, wait = -1.0, 0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for images, targets in dl:
            images = [i.to(dev) for i in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets if len(t["boxes"])]
            if not targets:
                continue
            with autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = sum(model(images, targets).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
        model.eval()
        f1 = float(evaluate(model, dev, root, areas, "GOV", "Unseen", classes="network")
                   .set_index("set").loc["all", "overall_weighted_F1"])
        log(f"[train] epoch {ep + 1}: loss {total:.1f}, validation overall weighted F1 {f1:.2f}")
        if f1 > best:
            best, wait = f1, 0
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            wait += 1
            if wait >= patience:
                log("[train] early stopping")
                break
    return best
