import torch
import torch.distributed as dist


# from https://github.com/vturrisi/solo-learn/blob/main/solo/utils/misc.py#L180
# had the problem that backward was not called for some reason
# noinspection PyAbstractClass
class AllGatherGradAutograd(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x):
        shapes = [torch.empty_like(torch.tensor(x.shape, device=x.device)) for _ in range(dist.get_world_size())]
        dist.all_gather(shapes, torch.tensor(x.shape, device=x.device))
        output = [
            torch.empty(*shape.tolist(), dtype=x.dtype, device=x.device) if shape.numel() > 0
            else torch.tensor(0, dtype=x.dtype, device=x.device)
            for shape in shapes
        ]
        dist.all_gather(output, x)
        # without the tuple call here, the gradient is not propagated for some reason
        # (therefore the backward is then not called)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        for i, grad in enumerate(grads):
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            if i == dist.get_rank():
                grad_out = grad
        return grad_out
