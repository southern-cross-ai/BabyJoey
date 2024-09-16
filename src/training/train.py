from torchtnt.framework.auto_unit import AutoUnit
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class BabyJoeyUnit(AutoUnit):
    def __init__(self, module, device=None, rank=None):
        super().__init__(module=module, device=device)
        self.loss_fn = nn.CrossEntropyLoss()

        # Check if distributed training is enabled and wrap the module with DDP
        if torch.distributed.is_initialized():
            self.module = DDP(module, device_ids=[device], output_device=device)
            print(f"Module wrapped with DDP on device {device} for rank {rank}")

    def compute_loss(self, state, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        key_padding_mask = (attention_mask == 0).bool()
        logits = self.module(input_ids, key_padding_mask=key_padding_mask)
        targets = input_ids[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss, logits

    def configure_optimizers_and_lr_scheduler(self, module):
        optimizer = optim.AdamW(module.parameters(), lr=1e-5, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return optimizer, scheduler
