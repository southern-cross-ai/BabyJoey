from typing import List, Optional, Tuple, TypeVar, Union, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.state import State
from torchtnt.utils import TLRScheduler
from torchtnt.utils.prepare_module import Strategy

TData = TypeVar("TData")


class BabyJoeyUnit(AutoUnit):
    def __init__(
        self,
        *,                                      # users must pass following args as positional args
        module: nn.Module,                      # model to trained/evaluate/predict
        device: Optional[torch.device] = None,  # device to be used for training/evaluation/prediction
        # TODO: Integrate lr, weight_decay and step_lr_interval into optimiser
        # optimiser: Optimizer = AdamW,
        # TODO: Integrate step_size, gamma, gradient_accumulation_steps into lr_scheduler
        # lr_scheduler: LRScheduler = StepLR,
        lr: float = 1e-5,
        weight_decay: float = 1e-3,
        step_size: int = 1,
        gamma: float = 0.9,
        # step_lr_interval: Literal["step", "epoch"] = "epoch",  # time to step scheduler
        # gradient_accumulation_steps: int = 1,  # number of batches to accumulate gradients for back propagation
        # TODO: Add DDP arg
        # strategy: Optional[Union[Strategy, str]] = None,  # data parallelisation strategy
        # devices_ids: Union[List[torch.device], List[int]] = None,  # DDP
        # output_device: Union[torch.device, int] = None,            # DDP
        # rank: int = None                                           # DDP
    ) -> None:                                 
        """Customised AutoUnit class for BabyJoey

        Args:
            module (nn.Module): BabyJoey model
            device (Optional[torch.device], optional): the device to be used. Defaults to None.
            devices_ids (Union[List[torch.device]  |  List[int]], optional): DDP devices to be used. Defaults to None.
            lr (float, optional): learning rate for the optimizer. Defaults to 1e-5.
            weight_decay (float, optional): weight decay for the optimizer. Defaults to 1e-3.
            step_size (int, optional): step size for the learning rate scheduler. Defaults to 1.
            gamma (float, optional): gamma for the learning rate scheduler. Defaults to 0.9.
        """
        
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()  # TODO: Allow user to specify loss function?
        
        self.device = device
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        
        # # Check if distributed training is enabled and wrap the module with DDP
        # if torch.distributed.is_initialized():
        #     self.device_ids = devices_ids
        #     self.output_device = output_device
        #     self.rank = rank
        #     self.module = DDP(module, device_ids=self.device_ids, output_device=self.output_device)
        #     # TODO: check other params when initialising the AutoUnit class
        #     print(f"Module wrapped with DDP on devices {self.device_ids} for rank {self.rank}."\
        #            "Output device is {self.output_device}.")
            
            
    def compute_loss(self, state: State, data: TData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implement this with loss computation. This will be called every train_step/eval_step.

        Args:
            state (State): A State object which is passed from the train_step/eval_step
            data (torch.Tensor): A batch of data which is passed from the train_step/eval_step

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the loss and the output of the model
        """
        
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        key_padding_mask = (attention_mask == 0).bool()
        logits = self.module(input_ids, attn_mask=None, key_padding_mask=key_padding_mask)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module: nn.Module) -> Tuple[Optimizer, LRScheduler]:
        """Implement this with optimizer and learning rate scheduler construction code. This will be called upon initialization of the AutoUnit.

        Args:
            module (torch.nn.Module): The module with which to construct optimizer and lr_scheduler

        Returns:
            Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]: A tuple containing optimizer and optionally the learning rate scheduler
        """
        
        optimizer = AdamW(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        
        return optimizer, scheduler
