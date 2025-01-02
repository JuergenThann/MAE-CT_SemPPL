import kappaprofiler as kp
from kappadata import LabelSmoothingWrapper
from torch.nn.functional import cross_entropy

from losses import loss_fn_from_kwargs
from losses.bce_loss import bce_loss
from utils.factory import create
from utils.object_from_kwargs import objects_from_kwargs
from .base.sgd_trainer import SgdTrainer


class DummyTrainer(SgdTrainer):
    def __init__(self, dataset_key="train", forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.dataset_key = dataset_key

    @property
    def output_shape(self):
        return self.data_container.get_dataset(self.dataset_key, mode=self.dataset_mode).n_classes,

    @property
    def dataset_mode(self):
        return "index x class"

    def forward(self, model, batch, dataset):
        (idx, x, y), ctx = batch
        with kp.named_profile_async("to_device"):
            x = x.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        with kp.named_profile_async("forward"):
            predictions = model(x, **self.forward_kwargs)
        # wrap model output into a dictionary in case it isn't already
        if not isinstance(predictions, dict):
            predictions = dict(main=predictions)
        return dict(predictions=predictions, x=x, y=y, idx=idx, **{f"ctx.{k}": v for k, v in ctx.items()})

    def get_loss(self, outputs, model):
        return outputs["y"], outputs
