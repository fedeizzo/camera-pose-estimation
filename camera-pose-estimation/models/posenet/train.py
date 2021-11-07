import torch
import numpy as np
from aim import Run

from torch.utils.data import DataLoader
from typing import Dict


def train(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    aim_run: Run,
    device: torch.device,
):
    best_model = model
    best_loss = np.Inf
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase, dataloader in dataloaders.items():
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            for index, (x, labels) in enumerate(dataloader):
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
            print(f"\t{phase} loss={epoch_loss}")
            aim_run.track(
                epoch_loss, name="loss", epoch=epoch, context={"subset": phase}
            )
            aim_run.track(scheduler.get_last_lr(), name="lr", epoch=epoch)

            if phase == "val" and epoch_loss <= best_loss:
                best_model = model
                best_loss = epoch_loss

    return best_model
