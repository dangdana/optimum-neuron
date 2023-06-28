# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes handling causal-lm related architectures on neuron devices."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
from tempfile import TemporaryDirectory

import torch

from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.utils import ModelOutput
from transformers.generation import GenerationMixin

from ..modeling_base import OptimizedModel
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from .utils import is_transformers_neuronx_available


if is_transformers_neuronx_available():
    from transformers_neuronx.module import save_pretrained_split, PretrainedModel as NeuronxPretrainedModel


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


class NeuronModelDecoder(OptimizedModel):

    def __init__(
        self,
        model: torch.nn.Module,
        config: "PreTrainedConfig",
        generation_config: Optional[GenerationConfig] = None
    ):
        if not is_transformers_neuronx_available() or not isinstance(model, NeuronxPretrainedModel):
            raise ValueError("The source model must be a transformers_neurons.PreTrainedModel.")

        super().__init__(model, config)
        self.device = torch.device('cpu')
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        tp_degree: int = 1,
        amp: str = 'f32',
        **kwargs_shapes
    ) -> "NeuronModelDecoder":
        """
        Exports a vanilla Transformers model into a neuronx-optimized HLO Module.

        Args:
            kwargs_shapes (`Dict[str, int]`):
                Shapes to use during inference. This argument allows to override the default shapes used during the export.
        """
        if not is_transformers_neuronx_available():
            raise ModuleNotFoundError("The transformers_neuronx package is required to export the model.")

        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        model = TasksManager.get_model_from_task(
            task=task,
            model_name_or_path=model_id,
            subfolder=subfolder,
            revision=revision,
            framework="pt",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        # Save the model in a temporary directory
        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        save_pretrained_split(model, save_dir_path)

        # Evaluate the conversion configuration parameters
        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="neuron", task=task
        )

        # Check if we were passed the right mandatory shape args and update config
        input_shapes = {}
        for name in neuron_config_constructor.func.get_mandatory_axes_for_task(task):
            static_shape = kwargs_shapes.get(name, None) or getattr(config, "neuron_" + name, None)
            if static_shape is None:
                raise AttributeError(
                    f"Cannot find the value of `{name}` from arguments nor the `config`. `{name}` is mandatory"
                    " for exporting the model to the neuron format, please set the value explicitly."
                )
            else:
                input_shapes[name] = static_shape
                config.__setattr__("neuron_" + name, static_shape)

        neuron_config = neuron_config_constructor(model.config, **input_shapes)

        # Extract the conversion configuration parameters from the shape kwargs
        # Could it be done in the loop before ?
        conversion_kwargs = {}
        if "batch_size" in input_shapes:
            conversion_kwargs['batch_size'] = input_shapes['batch_size']
        if "n_positions" in input_shapes:
            conversion_kwargs['n_positions'] = input_shapes['sequence_length']

        # Instantiate the optimized neuronx model from the transformers checkpoint
        neuronx_model = neuron_config.neuronx_class().from_pretrained(save_dir_path,
                                                                      amp=amp,
                                                                      tp_degree=tp_degree,
                                                                      **conversion_kwargs)

        # Compile the model
        neuronx_model.to_neuron()

        return cls(neuronx_model, config)


    def forward(self, *args, **kwargs):
        raise NotImplementedError()


    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.neuron.modeling_base.NeuronBaseModel.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        raise NotImplementedError()


class NeuronModelForCausalLM(NeuronModelDecoder, GenerationMixin):

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.cur_len = 0

    def reset_generation(self):
        self.cur_len = 0

    def forward(self, input_ids, cache_ids, start_ids=None, output_hidden_states=False, output_attentions=False,
            attention_mask=None, return_dict=False):

        if  output_hidden_states or output_attentions or attention_mask is not None:
            warnings.warn("Warning: These arguments are not used by forward(): \
                (output_hidden_states, output_attentions, attention_mask)")
        out_logits = self.model.forward(input_ids, cache_ids, start_ids)
        out_logits = out_logits[:, None, :]
        if return_dict:
            return ModelOutput(
                [("logits", out_logits)]
            )
        return (out_logits,)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # convert attention_mask to start_ids
        attention_mask = None
        start_ids = None
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]

        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)

        if self.cur_len > 0:
            input_ids = input_ids[:, -1:]
            cache_ids = torch.as_tensor([self.cur_len], dtype=torch.int32)
        else:
            cache_ids = torch.arange(input_ids.shape[-1], dtype=torch.int32)

        self.cur_len += input_ids.shape[-1]
        model_inputs = {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

        return model_inputs

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True