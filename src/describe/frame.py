"""Pipeline component that describes a frame in an image."""

from typing import Any
import torch
import PIL
import numpy as np

from transformers import (
    GitProcessor,
    GitForCausalLM,
    GenerationConfig,
    GitConfig,
    GitVisionConfig,
)
from transformers.utils import ModelOutput

from ..log import get_pipeline_logger
from pydantic import BaseModel, ConfigDict


class CaptionOutput(BaseModel):
    """Stores the output of captioning an image."""

    image_path: str
    caption_tokens: list[str]
    caption_text: str
    attentions_: Any

    model_config: ConfigDict = ConfigDict(
        # Allow tensors
        arbitrary_types_allowed=True
    )

    def get_attention(self, id_: int, head: int = None, layer: int = None) -> torch.Tensor:
        """Retrieves attention weights for specific word

        If head is passed, averaged across layers. If layer is
        pass, averaged across heads. If both, a single matrix is returned.
        If none is passed, an averaged over both heads and layers is returned.

        Args:
            id_ (int): The id of the word to get attention for.
            head (int, optional): The head to get attention for. Defaults to None.
            layer (int, optional): The layer to get attention for. Defaults to None.

        Returns:
            torch.Tensor: The attention weights for the word.
            If head is passed, the shape is (num_layers, num_patches).
            If layer is passed, the shape is (num_heads, num_patches).
            If both are passed, the shape is (num_patches).
            If none are passed, the shape is (num_patches).
        """
        len_attentions = len(self.attentions_)
        if len_attentions <= id_:
            raise ValueError(f"word id is out of bounds {id_} >= {len_attentions}")

        x = None
        if head is not None and layer is not None:
            x = self.attentions_[id_][layer][head]

        if head is not None:
            x = self.attentions_[id_][:, head, :].mean(dim=0)
        if layer is not None:
            x = self.attentions_[id_][layer, :].mean(dim=0)

        if x is not None:
            x = self.attentions_[id_].mean(dim=0).mean(dim=0)

        return x


class FrameDescriptorConfiguration(BaseModel):
    """The configuration for the FrameDescriptor model."""

    model_name: str = "microsoft/git-large-r-textcaps"
    generation_config: GenerationConfig = None
    use_gpu: bool = False
    
    model_config: ConfigDict = ConfigDict(
        # Allow generation config
        arbitrary_types_allowed=True
    )

    def __init__(self, **kwargs: Any):
        """Initialize the FrameDescriptorConfiguration class.

        Args:
            **kwargs (Any): The configuration for the model.
        """
        super().__init__(**kwargs)
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        
        # Override the default generation config
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_attentions = True
        # self.generation_config.num_beams = 6
        # self.generation_config.max_length = 26
        # self.generation_config.no_repeat_ngram_size = 2
        # self.generation_config.early_stopping = True
        # self.generation_config.length_penalty = 2.0
        # self.generation_config.num_return_sequences = 1
        # self.generation_config.top_k = 50
        # self.generation_config.top_p = 0.95
        # self.generation_config.temperature = 0.9
        # self.generation_config.do_sample = True


class FrameDescriptor:
    def __init__(
        self,
        *,
        config: FrameDescriptorConfiguration = None,
        **kwargs,
    ) -> None:
        self._config = config or FrameDescriptorConfiguration(**kwargs)

        self.logger = get_pipeline_logger("FrameDescriptor", "green")

        self.git_model = GitForCausalLM.from_pretrained(self._config.model_name)
        self.git_config = GitConfig.from_pretrained(self._config.model_name)
        self.git_vision_config = GitVisionConfig.from_pretrained(
            self._config.model_name
        )
        self.git_processor = GitProcessor.from_pretrained(self._config.model_name)
        
        if self._config.use_gpu and torch.cuda.is_available():
            self.git_model.cuda()

    def _get_text(self, outputs: ModelOutput) -> list[str]:
        if 'sequences' not in outputs:
            raise ValueError("The model output does not contain the 'sequences' key.")
        
        return self.git_processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    
    def _get_attentions(self, outputs: ModelOutput) -> Any:
        if 'attentions' not in outputs:
            raise ValueError("The model output does not contain the 'attentions' key.")
        
        sequences = len(outputs.sequences)
        sequence_length = len(outputs.attentions)
        num_hidden_layers = self.git_config.num_hidden_layers
        num_attention_heads = self.git_config.num_attention_heads
        num_patches = (self.git_vision_config.image_size // self.git_vision_config.patch_size) ** 2
        attentions = outputs.attentions
        
        att = torch.zeros(
            sequences,
            sequence_length,
            num_hidden_layers,
            num_attention_heads,
            num_patches,
        )
        
        for i in range(len(attentions)):
            for k in range(len(attentions[i])):
                for j in range(attentions[i][k].shape[0]):
                    att[j][i][k] = attentions[i][k][j][:, -1, :num_patches]
        
                    
        return att       
        

    
    def _get_words(self, outputs: ModelOutput) -> list[list[str]]:
        return [
            self.git_processor.tokenizer.convert_ids_to_tokens(
                outputs.sequences[i], skip_special_tokens=True
            ) for i in range(len(outputs.sequences))
        ]

    def describe_batch(self, images: list[str]) -> list[CaptionOutput]:
        """Describe a batch of images.

        Args:
            images (list[str]): The list of images to describe.

        Returns:
            CaptionsOutput
        """
        read_images = []
        for image in images:
            read_images.append(np.array(PIL.Image.open(image).convert("RGB")))

        inputs = self.git_processor(
            images=read_images, 
            return_tensors="pt", 
            padding=True
        )

        inputs = inputs.to("cuda") if self._config.use_gpu else inputs
        generated = self.git_model.generate(
            pixel_values=inputs['pixel_values'],
            generation_config=self._config.generation_config,
        )
        attentions = self._get_attentions(generated)
       
        return [
            CaptionOutput(
                image_path=image,
                caption_tokens=self._get_words(generated)[i],
                caption_text=self._get_text(generated)[i],
                attentions_=attentions[i],
            ) for i, image in enumerate(images)
        ]

