import os
import io
import warnings


with warnings.catch_warnings():
    # ignore warnings when importing mc
    warnings.simplefilter("ignore")
    try:
        import mc
    except ModuleNotFoundError:
        pass


class PetrelMCBackend():
    """Petrel storage backend with multiple clusters (for internal use).

    Args:
        path_mapping (dict|None): path mapping dict from local path to Petrel
            path. When `path_mapping={'src': 'dst'}`, `src` in `filepath` will
            be replaced by `dst`. Default: None.
        enable_mc (bool): whether to enable memcached support. Default: True.
    """
    def __init__(self, path_mapping=None, enable_mc=True):
        self.enable_mc = enable_mc
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

        self._client = None
        self._mc_client = None

    def _init_clients(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')
        self._client = client.Client(enable_mc=self.enable_mc,
                                     boto=True,
                                     enable_multi_cluster=True,
                                     conf_path='{}/.s3cfg'.format(
                                         os.environ['HOME']))
        server_list_cfg = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_cfg = "/mnt/lustre/share/memcached_client/client.conf"
        self._mc_client = mc.MemcachedClient.GetInstance(
            server_list_cfg, client_cfg)
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        if self._client is None:
            self._init_clients()
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        if filepath.startswith('cluster'):
            value = self._client.Get(filepath)
        else:
            self._mc_client.Get(filepath, self._mc_buffer)
            value = mc.ConvertBuffer(self._mc_buffer)
        value_buf = memoryview(value)
        buff = io.BytesIO(value_buf)
        return buff
