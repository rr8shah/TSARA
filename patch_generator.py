import numpy as np
import torch
from torchvision import transforms


class DenseSpatialCrop(object):
    """Densely crop an image, where stride is equal to the output size.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, stride):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        assert isinstance(stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            assert len(stride) == 2
            self.stride = stride

    def __call__(self, image):
        w, h = image.size[:2]
        new_h, new_w = self.output_size
        stride_h, stride_w = self.stride

        h_start = np.arange(0, h - new_h, stride_h)
        w_start = np.arange(0, w - new_w, stride_w)

        patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in w_start]

        to_tensor = transforms.ToTensor()
        patches = [to_tensor(patch) for patch in patches]
        patches = torch.stack(patches, dim=0)
        return patches
