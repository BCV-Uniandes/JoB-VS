# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from re import S
from typing import Callable, Sequence, Union

import torch
import logging

from job_vs.inferer import ROGInferer

from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelDeepEditd,
)
from monai.inferers import Inferer, SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored

logger = logging.getLogger(__name__)

class DeepEdit(InferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        dimension=3,
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        number_intensity_ch=1,
        threshold=0.5,
        description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
        config=None,
        preload=False,
        # **kwargs,
    ):
        if config:
            config.update({"threshold": threshold})
        else:
            config = {"threshold": threshold}
            
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
            config=config
            # **kwargs,
        )
        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.number_intensity_ch = number_intensity_ch  
        self.threshold = config["threshold"]
        
    def pre_transforms(self, data=None):
        t = [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ]
        if self.type == InferType.DEEPEDIT:
            t.extend(
                [
                    AddGuidanceFromPointsDeepEditd(ref_image="image", guidance="guidance", label_names=self.labels),
                    # Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                    # ResizeGuidanceMultipleLabelDeepEditd(guidance="guidance", ref_image="image"),
                    # AddGuidanceSignalDeepEditd(
                    #     keys="image", guidance="guidance", number_intensity_ch=self.number_intensity_ch
                    # ),
                ]
            )
        else:
            t.extend(
                [
                    # Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
                    # DiscardAddGuidanced(
                    #     keys="image", label_names=self.labels, number_intensity_ch=self.number_intensity_ch
                    # ),
                ]
            )

        t.append(EnsureTyped(keys="image", device=data.get("device") if data else None))
        return t

    def inferer(self, data=None, threshold=0.5) -> Inferer:
        return ROGInferer(device=data.get("device") if data else None, 
                          threshold=threshold)

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            # Activationsd(keys="pred", softmax=True),
            # AsDiscreted(keys="pred", argmax=True),
            # SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
    
    def run_inferer(self, data, threshold, convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param threshold: threshold to make the prediction
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """

        logger.info(f"Inference with {threshold} threshold")
        inferer = self.inferer(data, threshold)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")

        network = self._get_network(device)
        if network:
            inputs = data[self.input_key]
            inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
            inputs = inputs[None] if convert_to_batch else inputs
            inputs = inputs.to(torch.device(device))

            with torch.no_grad():
                outputs = inferer(inputs, network, data['image_path'])

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            convert_to_batch = False
            outputs = outputs[0] if convert_to_batch else outputs
            data[self.output_label_key] = outputs
        else:
            # consider them as callable transforms
            # data = run_transforms(data, inferer, log_prefix="INF", log_name="Inferer")
            data = data
            # run_transforms(data, inferer, log_prefix="INF", log_name="Inferer")
        return data
        
