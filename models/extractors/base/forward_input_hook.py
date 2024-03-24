class ForwardInputHook:
    def __init__(self, inputs: dict, input_name: str, allow_multiple_inputs: bool = False):
        self.inputs = inputs
        self.input_name = input_name
        self.allow_multiple_inputs = allow_multiple_inputs
        self.enabled = True

    def __call__(self, _, input_, __):
        if not self.enabled:
            return

        if self.allow_multiple_inputs:
            # e.g. contrastive heads have multiple forward passes
            input_name = f"{self.input_name}.{len(self.inputs)}"
            assert input_name not in self.inputs
            self.inputs[input_name] = input_[0]
        else:
            assert self.input_name not in self.inputs, "clear output before next forward pass to avoid memory leaks"
            self.inputs[self.input_name] = input_[0]
