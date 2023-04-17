import time
import numpy as np

from tqdm import tqdm

from . import helpers

import torch
import torch.nn.functional as F

from scipy.ndimage.filters import gaussian_filter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from monai.inferers import Inferer
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple
from torch.utils.data import Dataset, DataLoader
from monai.visualize import CAM, GradCAM, GradCAMpp
from monai.data.meta_tensor import MetaTensor
 
from torch.utils.data import Dataset

class ROGInferer(Inferer):
    """
    SimpleInferer is the normal inference method that run model forward() directly.
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    """

    def __init__(self) -> None:
        ROGInferer.__init__(self)
        
    def __init__(
            self,
            roi_size: Union[Sequence[int], int] = None,
            sw_batch_size: int = 1,
            mode: Union[BlendMode, str] = BlendMode.CONSTANT,
            sigma_scale: Union[Sequence[float], float] = 0.125,
            padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
            cval: float = 0.0,
            threshold: float = 0.5,
            sw_device: Union[torch.device, str, None] = None,
            device: Union[torch.device, str, None] = None,
            progress: bool = False,
            cache_roi_weight_map: bool = False,
        ) -> None:
        
        Inferer.__init__(self)
        
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.progress = progress
        self.threshold = threshold
        
        test_dataset = ROGInfLoader('datasets/ROG_monai/', [128,128,64])
        self.loader =  DataLoader(
            test_dataset, sampler=None, shuffle=False, batch_size=1, num_workers=0)
        # compute_importance_map takes long time when computing on cpu. We thus
        # compute it once if it's static and then save it for future usage
        self.roi_weight_map = None


    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
        image_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        
        # Add more weight to the central voxels
        w_patch = np.zeros([128,128,64])
        sigmas = np.asarray([128,128,64]) // 8
        center = torch.Tensor([128,128,64]) // 2
        w_patch[tuple(center.long())] = 1
        w_patch = gaussian_filter(w_patch, sigmas, 0, mode='constant', cval=0)
        w_patch = torch.Tensor(w_patch / w_patch.max()).to(self.device).double()

        shape, name, affine, pad = self.loader.dataset.update(image_path)
        prediction = torch.zeros((2,) + shape).to(self.device).double()
        
        weights = torch.zeros(shape).to(self.device).double()
        print('Doing inference...')
        for sample in tqdm(self.loader):
            data = sample['data'].float()  # .squeeze_(0)
            with torch.no_grad():
                output = network(data.to(self.device))[0]
            # output = dataloader.test_data(output, False)
            output *= w_patch

            low = (sample['target'][0] - center).long()
            up = (sample['target'][0] + center).long()
            prediction[:, low[0]:up[0], low[1]:up[1], low[2]:up[2]] += output
            weights[low[0]:up[0], low[1]:up[1], low[2]:up[2]] += w_patch

        prediction /= weights
        prediction = F.softmax(prediction, dim=0)
        
        prediction = (prediction[1] >= self.threshold).double()
        if pad is not None:
            prediction = prediction[pad[0][0]:shape[0] - pad[0][1],
                                    pad[1][0]:shape[1] - pad[1][1],
                                    pad[2][0]:shape[2] - pad[2][1]]
        prediction =  MetaTensor(prediction, affine=affine)
        return prediction
        
    
class ROGInfLoader(Dataset):

    def __init__(self, root_dir, patch_size):
        super(ROGInfLoader, self).__init__()
        # ! Modified to read annotations in UNETR format
        # ! May 17th by N. Valderrama.
        self.root_dir = root_dir
        self.patch_size = np.asarray(patch_size)
        self.fg = 0

    def __len__(self):
        return len(self.voxel)
    
    def __getitem__(self, idx):
        patches = helpers.extract_patch(
            self.image, self.voxel[idx], self.patch_size)
        label = torch.Tensor(self.voxel[idx])
        # patches = helpers.test_data(patches)
        patches = torch.from_numpy(patches)
        info = 0

        return {'data': patches, 'target': label, 'info': info}

    def update(self, patient):
        # This is only for testing
        name = patient[0][9:]  # ./imagesTr/XXXX.nii.gz (or imagesTs)
        print('Loading data of patient {} ---> {}'.format(
            name, time.strftime("%H:%M:%S")))
        # TODO verify if this is the correct way to load the image
        image, _, affine = helpers.load_image(
            patient, self.root_dir, False)
        self.image, pad = helpers.verify_size(image, self.patch_size)
        im_shape, multimodal = helpers.image_shape(self.image)
        if multimodal and pad is not None:
            pad = pad[1:]

        self.voxel = helpers.test_voxels(self.patch_size, im_shape)
        return im_shape, name, affine, pad