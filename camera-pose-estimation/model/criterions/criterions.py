import torch
from torch import nn

from typing import Callable, Any


def weighted_mse_loss(
    weight: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def fun(input: torch.Tensor, target: torch.Tensor):
        return (weight * (input - target) ** 2).mean()

    return fun


def dense_custom_loss(
    alpha: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def fun(input: torch.Tensor, target: torch.Tensor):
        return torch.sqrt(
            torch.sum((target[:, :3] - input[:, :3]) ** 2)
        ) + alpha * torch.sqrt(torch.sum((target[:, 3:] - input[:, 3:]) ** 2))

    return fun


def get_loss(config_loss: dict, device: torch.device) -> Any:
    if config_loss["type"] == "mse":
        criterion = torch.nn.MSELoss()
    elif config_loss["type"] == "L1Loss":
        criterion = torch.nn.L1Loss()
    elif config_loss["type"] == "SmoothL1Loss":
        criterion = torch.nn.SmoothL1Loss()
    elif config_loss["type"] == "dense_custom":
        criterion = dense_custom_loss(alpha=config_loss["alpha"])
    elif config_loss["type"] == "weighted":
        criterion = weighted_mse_loss(torch.Tensor(config_loss["weights"]).to(device))
    elif config_loss["type"] == "mapnet_criterion":
        criterion = MapNetCriterion(
            device=device,
            sax=0.0,
            saq=config_loss["beta"],
            srx=0.0,
            srq=config_loss["gamma"],
            learn_beta=config_loss["learn_beta"],
            learn_gamma=config_loss["learn_gamma"],
        )
    else:
        raise ValueError(f"Unknown criterion: {config_loss['name']}")

    return criterion


def calc_vos_simple(poses) -> torch.Tensor:
    """
    Computes relative poses between a sequence of images
    """
    vos = []
    for p in poses:
        pvos = [p[i + 1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)

    return vos


class QuaternionLoss(nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, q1, q2):
        loss = 1 - torch.pow(torch.vdot(q1, q2), 2)
        loss = torch.mean(loss)
        return loss


class MapNetCriterion(nn.Module):
    def __init__(
        self,
        device: torch.device,
        t_loss_fn=nn.MSELoss(),
        q_loss_fn=nn.MSELoss(),
        sax: float = 0.0,
        saq: float = 0.0,
        srx: float = 0,
        srq: float = 0.0,
        learn_beta: bool = False,
        learn_gamma: bool = False,
    ):
        super(MapNetCriterion, self).__init__()
        self.learn_beta = learn_beta
        self.learn_gamma = learn_gamma
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(
            torch.Tensor([sax]).to(device), requires_grad=learn_beta
        )
        self.saq = nn.Parameter(
            torch.Tensor([saq]).to(device), requires_grad=learn_beta
        )
        self.srx = nn.Parameter(
            torch.Tensor([srx]).to(device), requires_grad=learn_gamma
        )
        self.srq = nn.Parameter(
            torch.Tensor([srq]).to(device), requires_grad=learn_gamma
        )

    def forward(self, pred, targ):
        size = pred.size()
        abs_loss = torch.sqrt(
            self.t_loss_fn(
                pred.view(-1, *size[2:])[:, :3],
                targ.view(-1, *size[2:])[:, :3],
            )
        ) + torch.sqrt(
            self.q_loss_fn(
                pred.view(-1, *size[2:])[:, 3:],
                targ.view(-1, *size[2:])[:, 3:],
            )
        )

        pred_vos = calc_vos_simple(pred)
        targ_vos = calc_vos_simple(targ)

        size = pred_vos.size()
        vo_loss = torch.sqrt(
            self.t_loss_fn(
                pred_vos.view(-1, *size[2:])[:, :3],
                targ_vos.view(-1, *size[2:])[:, :3],
            )
        ) + torch.sqrt(
            self.q_loss_fn(
                pred_vos.view(-1, *size[2:])[:, 3:],
                targ_vos.view(-1, *size[2:])[:, 3:],
            )
        )

        loss = abs_loss + vo_loss
        return loss
