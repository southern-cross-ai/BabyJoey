from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.state import State
from torchtnt.utils import TLRScheduler


class BabyJoeyUnit(AutoUnit):
    def __init__(self, module, device=None, rank=None):
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()

        # Check if distributed training is enabled and wrap the module with DDP
        if torch.distributed.is_initialized():
            self.module = DDP(module, device_ids=[device], output_device=device)
            print(f"Module wrapped with DDP on device {device} for rank {rank}")

    def compute_loss(self, state: State, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implement this with loss computation. This will be called every train_step/eval_step.

        Args:
            state (State): A State object which is passed from the train_step/eval_step
            data (torch.Tensor): A batch of data which is passed from the train_step/eval_step

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the loss and the output of the model
        """
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        key_padding_mask = (attention_mask == 0).bool()
        logits = self.module(input_ids, key_padding_mask=key_padding_mask)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module: torch.nn.Module) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:
        """Implement this with optimizer and learning rate scheduler construction code. This will be called upon initialization of the AutoUnit.

        Args:
            module (torch.nn.Module): The module with which to construct optimizer and lr_scheduler

        Returns:
            Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]: A tuple containing optimizer and optionally the learning rate scheduler
        """
        optimizer = optim.AdamW(module.parameters(), lr=1e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer, scheduler
