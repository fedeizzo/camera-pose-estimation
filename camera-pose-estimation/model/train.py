import torch
import numpy as np

from aim import Run
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict
from criterions.criterions import MapNetCriterion


def train_model(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    aim_run: Run,
    device: str,
) -> torch.nn.Module:
    best_model = model
    best_loss = np.Inf
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            pbar.set_description(f"Epoch {epoch}")
            phases_loss = {}
            for phase, dataloader in dataloaders.items():
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0.0
                for index, (x, labels) in enumerate(dataloader):
                    x = x.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.autocast(device_type=device):
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(x)
                            loss = criterion(predictions, labels)
                            epoch_loss += loss.item()

                            if phase == "train":
                                loss.backward()
                                optimizer.step()
                                scheduler.step()

                epoch_loss /= len(dataloader)
                phases_loss[phase] = epoch_loss
                aim_run.track(
                    epoch_loss,
                    name="loss",
                    epoch=epoch,
                    context={"subset": phase},
                )
                aim_run.track(scheduler.get_last_lr(), name="lr", epoch=epoch)

                if isinstance(criterion, MapNetCriterion):
                    aim_run.track(criterion.sax, name="sax", epoch=epoch)
                    aim_run.track(criterion.saq, name="saq", epoch=epoch)
                    aim_run.track(criterion.srx, name="srx", epoch=epoch)
                    aim_run.track(criterion.srq, name="srq", epoch=epoch)

                if phase == "validation" and abs(epoch_loss) <= abs(best_loss):
                    best_model = model
                    best_loss = epoch_loss
            pbar.set_postfix(phases_loss)
            pbar.update(1)

    return best_model
