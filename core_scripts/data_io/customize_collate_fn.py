

from __future__ import absolute_import

import os
import sys
import torch
import re
from torch._six import container_abcs, string_classes, int_classes


np_str_obj_array_pattern = re.compile(r'[SaUO]')

customize_collate_err_msg = (
    "customize_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def pad_sequence(batch, padding_value=0.0):

    max_size = batch[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in batch])
    
    if all(x.shape[0] == max_len for x in batch):
        return batch
    else:
        out_dims = (max_len, ) + trailing_dims
        
        output_batch = []
        for i, tensor in enumerate(batch):
            if tensor.size()[1:] != trailing_dims:
                print("Data in batch has different dimensions:")
                for data in batch:
                    print(str(data.size()))
                raise RuntimeError('Fail to create batch data')
            out_tensor = tensor.new_full(out_dims, padding_value)
            out_tensor[:tensor.size(0), ...] = tensor
            output_batch.append(out_tensor)
        return output_batch


def customize_collate(batch):


    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        batch_new = pad_sequence(batch)
        
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = max([x.numel() for x in batch_new]) * len(batch_new)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch_new, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(customize_collate_err_msg.format(elem.dtype))
            return customize_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
        
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customize_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customize_collate(samples) \
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        transposed = zip(*batch)
        return [customize_collate(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))



def customize_collate_from_batch(batch):


    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        batch_new = pad_sequence(batch)        
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = max([x.numel() for x in batch_new]) * len(batch_new)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch_new, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(customize_collate_err_msg.format(elem.dtype))
            return customize_collate_from_batch(
                [torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple):
        tmp = elem
        for tmp_elem in batch[1:]:
            tmp += tmp_elem 
        return tmp
    elif isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in batch should be of equal size')
        transposed = zip(*batch)
        return [customize_collate_from_batch(samples) for samples in transposed]

    raise TypeError(customize_collate_err_msg.format(elem_type))


if __name__ == "__main__":
    print("Definition of customized collate function")
