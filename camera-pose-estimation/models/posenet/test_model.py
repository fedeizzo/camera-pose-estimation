import torch
from torch.utils.data import DataLoader

def test_model(
    model: torch.nn.Module,
    dataloaders: DataLoader
):
    model.eval()
    torch.set_grad_enabled(False)
    predictions = []
    targets = []
    for x, labels in dataloaders:
        predictions.append(model(x))
        targets.append(labels)

    if len(predictions):
        predictions = predictions[0]
    if len(targets):
        targets = targets[0]
    import pdb; pdb.set_trace()
