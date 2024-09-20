from typing import TypeVar

import pytorch_lightning as pl
import torch
from torch import nn

from mpq import MPQModule, Quantizer, mpq_resnet

T = TypeVar("T", bound=nn.Module)


class CIFAR10MPQTrainer(pl.LightningModule):
    def __init__(self, arch: str, config: dict) -> None:
        from torchmetrics import Accuracy

        super().__init__()
        self.network = network = getattr(mpq_resnet, arch)()
        self.qlayers = self._detect_layers(network, MPQModule)
        self.quantizers = self._detect_layers(network, Quantizer)
        for qlayer in self.qlayers.values():
            qlayer.set_config(config)
        self.top1 = Accuracy("multiclass", num_classes=10, top_k=1)
        self.config = config
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        from torch.nn.functional import cross_entropy

        x, y = batch
        logits = self.network(x)
        ce_loss = cross_entropy(logits, y)
        weight_kb, activ_kb = self.compute_sizes_kb()
        self.log("train/task_loss", ce_loss)
        self.log("train/weight_kb", weight_kb)
        self.log("train/activ_kb", activ_kb)
        for name, quantizer in self.quantizers.items():
            self.log(f"qbits/{name}", quantizer.b())
        # NOTE: When Quantizers are in bypass mode, weight_kb and activ_kb are constant
        # so it's fine to add them in.
        return ce_loss + 0.1 * weight_kb + 0.1 * activ_kb

    def validation_step(self, batch, batch_idx):
        from torch.nn.functional import cross_entropy

        x, y = batch
        logits = self.network(x)
        ce_loss = cross_entropy(logits, y)
        self.log("val/task_loss", ce_loss)
        top1_acc = self.top1(logits, y)
        self.log("val/top1", top1_acc)
        return top1_acc

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        train_set = self._train_val_dataloader("lightning_logs/data_cifar10")[0]
        return DataLoader(
            train_set,
            batch_size=128,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        from torch.utils.data import DataLoader

        test_set = self._train_val_dataloader("lightning_logs/data_cifar10")[1]
        return DataLoader(
            test_set,
            batch_size=128,
            num_workers=4,
            shuffle=False,
            persistent_workers=True,
        )

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import MultiStepLR
        from torch.optim.sgd import SGD

        optimizer = SGD(
            self.network.parameters(), lr=self.config["lr"], momentum=0.9, weight_decay=5e-4
        )
        scheduler = MultiStepLR(optimizer, **self.config["lr_schedule"])
        return [optimizer], [scheduler]

    def compute_sizes_kb(self):
        """Compute the size of the weights and activations in kilobytes."""
        zero = torch.tensor(0.0)
        weight = sum(
            (quantizer.get_weight_bytes() for quantizer in self.qlayers.values()),
            start=zero,
        )
        activ = sum(
            (quantizer.get_activ_bytes() for quantizer in self.qlayers.values()),
            start=zero,
        )
        return weight / 1024.0, activ / 1024.0

    def _train_val_dataloader(self, dataset_path):
        from torchvision import transforms as tr
        from torchvision.datasets import CIFAR10

        val_trs = [
            tr.ToTensor(),
            tr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        train_tr = tr.Compose(
            [tr.Pad(4), tr.RandomCrop(32), tr.RandomHorizontalFlip(), *val_trs]
        )
        test_tr = tr.Compose(val_trs)
        train_set = CIFAR10(dataset_path, train=True, transform=train_tr, download=True)
        test_set = CIFAR10(dataset_path, train=False, transform=test_tr, download=True)
        return train_set, test_set

    @classmethod
    def _detect_layers(cls, module: nn.Module, to_detect: type[T]) -> dict[str, T]:
        quantizers: dict[str, T] = {}
        for name, child in module.named_modules():
            if isinstance(child, to_detect):
                quantizers[name] = child
        return quantizers


def train_mpq():
    from pytorch_lightning import callbacks as plcb
    from yaml import safe_load

    with open("training_config.yaml") as f:
        config = safe_load(f)
    model = CIFAR10MPQTrainer("resnet20", config)
    val_metric = "val/top1"
    filename = "epoch={epoch}-metric={%s:.3f}" % val_metric
    ckpt = plcb.ModelCheckpoint(
        monitor=val_metric,
        filename=filename,
        mode="max",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    lrm = plcb.LearningRateMonitor()
    # TODO: Sony impl has custom gradient clipping for d and qmax only
    trainer = pl.Trainer(
        devices=1,
        max_epochs=config["n_epochs"],
        callbacks=[ckpt, lrm],
        gradient_clip_val=0.1,
    )
    trainer.fit(model)


if __name__ == "__main__":
    train_mpq()
