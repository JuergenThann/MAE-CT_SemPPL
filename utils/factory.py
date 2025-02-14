import importlib
import logging
from functools import partial
from itertools import product

from utils.reflection_util import get_all_ctor_kwarg_names


def create(obj_or_kwargs, from_kwargs_fn, instantiate_if_ctor=True, **kwargs):
    """
    avoid boilerplate code when allowing ctor arguments to be either an object or a dict with the object parameters
    e.g. a model can be instantiated with either act=torch.nn.ReLU or act=dict(kind='relu') and the ctor has to only
    call self.act_ctor = create(act, act_ctor_from_kwargs) instead
    """
    if isinstance(obj_or_kwargs, dict):
        if len(obj_or_kwargs) == 0:
            return None
        return from_kwargs_fn(**obj_or_kwargs, **kwargs)
    if instantiate_if_ctor and isinstance(obj_or_kwargs, (partial, type)):
        # allow passing partials to objects which are then instantiated automatically
        # useful for e.g. passing model_ctors to autoencoder and autoencoder ctor passes latent_dim to decoder
        return obj_or_kwargs(**kwargs)
    # don't allow this as this would make the configs no longer 1:1 comparable (e.g. Linear(pooling="cls") would have
    # model.pooling == "cls" in the config whereas Linear(pooling=dict(kind="cls")) would have
    # model.pooling == dict(kind="cls")
    # if isinstance(obj_or_kwargs, str):
    # allow setting kind with only a string (e.g. Linear(pooling="cls") will create a pooling object
    # return from_kwargs_fn(kind=obj_or_kwargs, **kwargs)
    return obj_or_kwargs


def create_collection(collection, from_kwargs_fn, collate_fn=None, **kwargs):
    if isinstance(collection, list):
        objs = []
        for ckwargs in collection:
            if isinstance(ckwargs, dict):
                objs.append(create({**kwargs, **ckwargs}, from_kwargs_fn))
            else:
                objs.append(ckwargs)
    elif isinstance(collection, dict):
        objs = {key: create(ckwargs, from_kwargs_fn, **kwargs) for key, ckwargs in collection.items()}
    elif collection is None:
        objs = []
    else:
        raise NotImplementedError(f"invalid collection type {type(collection).__name__} (expected dict or list)")
    if collate_fn is not None:
        return collate_fn(objs)
    return objs


def get_ctor(module_names, type_names, **kwargs):
    obj_type = type_from_name(module_names=module_names, type_names=type_names)
    return partial(obj_type, **kwargs)


def instantiate(module_names, type_names, error_on_not_found=True, ctor_kwargs=None, optional_kwargs=None, **kwargs):
    obj_type = type_from_name(module_names=module_names, type_names=type_names, error_on_not_found=error_on_not_found)
    ctor_kwargs = {} if ctor_kwargs is None else dict(ctor_kwargs=ctor_kwargs)
    try:
        # e.g. pass update_counter to SchedulableLoss but not to e.g. torch.nn.MSELoss
        if optional_kwargs is not None:
            ctor_kwarg_names = get_all_ctor_kwarg_names(obj_type)
            for key in list(optional_kwargs.keys()):
                if key not in ctor_kwarg_names:
                    optional_kwargs.pop(key)
        else:
            optional_kwargs = {}
        return obj_type(**kwargs, **ctor_kwargs, **optional_kwargs)
    except TypeError as e:
        logging.error(f"error creating object of type {obj_type.__name__}: {e}")
        raise


def type_from_name(module_names, type_names, error_on_not_found=True):
    """
    tries to import type_name from any of the modules identified by module_names
    e.g. module_names=[loss_functions, torch.nn] type_name=bce_loss will import torch.nn.BCELoss
    """
    for module_name, type_name in product(module_names, type_names):
        module_name = module_name.lower()
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # this also fails if some module could not be imported from within the module to import
            # (e.g. failed to import torchmetrics when importing accuracy_logger)
            if not module_name.startswith(e.name):
                raise e
            continue

        type_ = _get_type_from_module(module, type_name)
        if type_ is not None:
            return type_

    # check if module was set in code (used for unittesting)
    # e.g. models.mock_model = test_unit.mock.mock_model
    for module_name, type_name in product(module_names, type_names):
        module_name = module_name.lower()
        parent_module_name = ".".join(module_name.split(".")[:-1])
        try:
            parent_module = importlib.import_module(parent_module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(parent_module, type_name):
            module = getattr(parent_module, type_name)
            type_ = _get_type_from_module(module, type_name)
            if type_ is not None:
                return type_

    if error_on_not_found:
        # ModuleNotFoundError from above are swollowed for stuff like torchmetrics if packagemanagement is not correct
        # but here the error then occours as a module can't be found
        raise RuntimeError(f"can't find class {' or '.join(type_names)} in {' or '.join(module_names)}")
    else:
        return None


def _get_type_from_module(module, type_name):
    type_name_lowercase = type_name.lower().replace("_", "")
    possible_type_names = list(filter(
        lambda kv: isinstance(kv[1], type) and kv[0].lower() == type_name_lowercase,
        module.__dict__.items()
    ))
    assert len(possible_type_names) <= 1, f"error found more than one possible type for {type_name_lowercase}"

    if len(possible_type_names) == 1:
        return getattr(module, possible_type_names[0][0])
    return None
