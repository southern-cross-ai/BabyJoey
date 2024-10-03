import deepspeed
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Tuple, TypeVar, Union
from deepspeed.ops.adam import FusedAdam

TData = TypeVar("TData")


class BabyJoeyUnit:
    def __init__(self,
                 module: nn.Module,
                 device: Optional[torch.device] = None,
                 lr: float = 1e-5,
                 weight_decay: float = 1e-3,
                 step_size: int = 1,
                 gamma: float = 0.9,
                 devices_ids: Union[List[torch.device], List[int]] = None,  # DDP
                 output_device: Union[torch.device, int] = None,            # DDP
                 rank: int = None,                                         # DDP
                 use_fp16: bool = True                                     # FP16 for DeepSpeed
                 ) -> None:
        """Customised AutoUnit class for BabyJoey with DeepSpeed support

        Args:
            module (nn.Module): BabyJoey model
            device (Optional[torch.device], optional): the device to be used. Defaults to None.
            devices_ids (Union[List[torch.device] | List[int]], optional): Devices for DDP, not needed for DeepSpeed. Defaults to None.
            lr (float, optional): learning rate for the optimizer. Defaults to 1e-5.
            weight_decay (float, optional): weight decay for the optimizer. Defaults to 1e-3.
            step_size (int, optional): step size for the learning rate scheduler. Defaults to 1.
            gamma (float, optional): gamma for the learning rate scheduler. Defaults to 0.9.
            use_fp16 (bool): whether to use FP16 mixed precision. Defaults to True.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = module.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.loss_fn = nn.CrossEntropyLoss()  # Loss function
        self.global_step = 0  # Track global step for logging

        # DeepSpeed configuration
        self.ds_config = {
            "train_batch_size": 2,  # Batch size; customize if necessary
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            },
            "fp16": {
                "enabled": use_fp16
            },
            "zero_optimization": {
                "stage": 2  # ZeRO optimization stage; customize based on your needs
            }
        }

        # Initialize DeepSpeed
        self.module_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.module,
            config=self.ds_config
        )

    def compute_loss(self, state: Optional[None], data: TData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss using DeepSpeed's backward pass

        Args:
            state (State): Ignored in this context
            data (torch.Tensor): A batch of input data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the loss and logits
        """
        input_ids, attention_mask = data['input_ids'].to(self.device), data['attention_mask'].to(self.device)
        key_padding_mask = (attention_mask == 0).bool()

        # Forward pass
        logits = self.module_engine(input_ids, key_padding_mask=key_padding_mask)

        # Shift tokens for next token prediction task
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()

        # Calculate loss
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        return loss, logits

    def train_step(self, data: TData) -> None:
        """Run one training step using DeepSpeed"""
        loss, _ = self.compute_loss(None, data)

        # Backward pass using DeepSpeed
        self.module_engine.backward(loss)

        # Step optimizer with DeepSpeed
        self.module_engine.step()

        # Print loss every 100 steps
        self.global_step += 1
        if self.global_step % 100 == 0:
            print(f"Step {self.global_step}, Loss: {loss.item()}")

    def configure_optimizers_and_lr_scheduler(self) -> Tuple[torch.optim.Optimizer, Optional[None]]:
        """DeepSpeed handles optimizer configuration internally"""
        return self.optimizer, None
