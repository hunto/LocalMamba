import torch
import numpy as np


def fast_collate(batch, memory_format=torch.contiguous_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].shape[2]
    h = imgs[0].shape[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, nump_array in enumerate(imgs):
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        #nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class DataPrefetcher():
    def __init__(self, loader, transforms, mixup_transform=None):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.transforms = transforms
        self.mixup_transform = mixup_transform
        self.stream = torch.cuda.Stream()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader_iter)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.transforms(self.next_input.float())
            if self.mixup_transform is not None:
                self.next_input, self.next_target = \
                    self.mixup_transform(self.next_input, self.next_target)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def __iter__(self):
        self.loader_iter = iter(self.loader)   # re-generate an iter for each epoch
        self.preload()
        return self

    def __next__(self):
        input, target = self.next()
        if input is None:
            raise StopIteration
        return input, target

    def __len__(self):
        return len(self.loader)


