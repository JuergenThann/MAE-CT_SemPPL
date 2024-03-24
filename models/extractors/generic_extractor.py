import torch.nn as nn
import re

from .base.extractor_base import ExtractorBase
from .base.forward_hook import ForwardHook
from .base.forward_input_hook import ForwardInputHook


class GenericExtractor(ExtractorBase):
    def __init__(self, model_property_path=None, allow_multiple_outputs=False, use_inputs=False, **kwargs):
        super().__init__(**kwargs)
        self.allow_multiple_outputs = allow_multiple_outputs
        self.model_property_path = model_property_path
        self.use_inputs = use_inputs

    def to_string(self):
        pooling_str = f"({self.pooling})" if not isinstance(self.pooling, nn.Identity) else ""
        return f"GenericExtractor{pooling_str}"

    def _register_hooks(self, model):
        if self.model_property_path:
            object_to_hook = eval(f'model.{self.model_property_path}', {}, {'model': model})
        else:
            object_to_hook = model
        if self.use_inputs:
            hook = ForwardInputHook(
                inputs=self.outputs,
                input_name=self.model_path,
                allow_multiple_inputs=self.allow_multiple_outputs
            )
        else:
            hook = ForwardHook(
                outputs=self.outputs,
                output_name=self.model_path,
                allow_multiple_outputs=self.allow_multiple_outputs,
            )
        object_to_hook.register_forward_hook(hook)
        self.hooks.append(hook)
