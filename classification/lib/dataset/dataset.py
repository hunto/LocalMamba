import io
import torch
import warnings
from PIL import Image
from torch.utils.data import Dataset

try:
    import mc
    from .file_io import PetrelMCBackend
    _has_mc = True
except ModuleNotFoundError:
    warnings.warn('mc module not found, using original '
                  'Image.open to read images')
    _has_mc = False


class ImageNetDataset(Dataset):
    r"""
    Dataset using memcached to read data.

    Arguments
        * root (string): Root directory of the Dataset.
        * meta_file (string): The meta file of the Dataset. Each line has a image path
          and a label. Eg: ``nm091234/image_56.jpg 18``.
        * transform (callable, optional): A function that transforms the given PIL image
          and returns a transformed image.
    """
    def __init__(self, root, meta_file, transform=None):
        self.root = root
        if _has_mc:
            with open('./data/mc_prefix.txt', 'r') as f:
                prefix = f.readline().strip()
            self.root = prefix + '/' + \
                ('train' if 'train' in self.root else 'val')
        self.transform = transform
        with open(meta_file) as f:
            meta_list = f.readlines()
        self.num = len(meta_list)
        self.metas = []
        for line in meta_list:
            path, cls = line.strip().split()
            self.metas.append((path, int(cls)))
        self._mc_initialized = False

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self._mc_initialized:
            '''
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self._mc_initialized = True
            '''
            self.backend = PetrelMCBackend()

    def __getitem__(self, index):
        filename = self.root + '/' + self.metas[index][0]
        cls = self.metas[index][1]

        if _has_mc:
            # memcached
            self._init_memcached()
            '''
            value = mc.pyvector()
            self.mclient.Get(filename, value)
            value_buf = mc.ConvertBuffer(value)
            buff = io.BytesIO(value_buf)
            '''
            buff = self.backend.get(filename)
            with Image.open(buff) as img:
                img = img.convert('RGB')
        else:
            img = Image.open(filename).convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
