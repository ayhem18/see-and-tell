from itertools import product
import torch
from .noun_phrases_detection import extract_noun_phrases
from ..describe.frame import CaptionOutput
from pydantic import BaseModel


class CaptionAttentionConfig(BaseModel):
    gamma: float = torch.inf
    top_k: int = 16


class CaptionAttentionSystem:
    def __init__(
        self,
        patch_size,
        num_heads,
        num_layers,
        image_size=224,
        config: CaptionAttentionConfig = None,
        **kwargs,
    ) -> None:
        self._conv = torch.nn.Conv2d(
            kernel_size=patch_size, stride=patch_size, bias=False, padding=0, in_channels=1, out_channels=1)
        self._conv.weight.requires_grad = False
        self._conv.weight.fill_(1.0 / (patch_size ** 2))
        self._image_size = image_size
        self._patch_size = patch_size
        self._num_heads = num_heads
        self._num_layers = num_layers

        self._config = config or CaptionAttentionConfig(**kwargs)

    def __entropy(self, inputs: torch.Tensor) -> float:
        probs = inputs / inputs.sum()
        return -(probs * torch.log(probs)).sum(dim=-1).item()

    def __get_indices(self, words_: list[str], phrases: list[list[str]]) -> list[tuple[int, int]]:
        """Finds the maximum contagious subarray of words in the phrase."""
        
        # Copy the words
        words = [x for x in words_]
        best_indices = []
        for phrase in phrases:
            for i in range(len(words)):
                if words[i:i + len(phrase)] == phrase:
                    best_indices.append((i, i + len(phrase)))
                    words[i:i + len(phrase)] = ['' for _ in range(len(phrase))]
                    break
        return best_indices

        

    def __apply_attention_weights(
        self,
        box: list[float],
        attention_weights: torch.Tensor,
        height=224,
        width=224,
    ):
        attention_weights = torch.stack(attention_weights, dim=0).sum(dim=0)
        bx = torch.zeros(self._image_size, self._image_size)

        x0, y0, x1, y1 = box
        y0 = (y0 / height) * self._image_size
        y1 = (y1 / height) * self._image_size
        x0 = (x0 / width) * self._image_size
        x1 = (x1 / width) * self._image_size
        x0, x1 = int(x0), int(x1)
        y0, y1 = int(y0), int(y1)

        bx[y0:y1, x0:x1] = 1
        bx = bx.unsqueeze(0).unsqueeze(0)
        bx = self._conv(bx)
        bx = bx.squeeze(0).squeeze(0)

        attention_weights = attention_weights * bx.flatten()
        return attention_weights

    def match(
        self,
        output: CaptionOutput,
        words: list[list[str]],
        phrases: list[list[str]],
        boxes: list[list[float]],
        height=224,
        width=224,
    ) -> tuple[tuple[int, int], int]:
        matched = []
        pairs = [*product(range(self._num_heads), range(self._num_layers))]

        match_indices = self.__get_indices(words, phrases)

        matched = []
        for i, box in enumerate(boxes):
            probs = [0 for _ in range(len(match_indices))]
            for j, (left, right) in enumerate(match_indices):
                w = words[left:right]
                length = len(w)
                atts = [[] for _ in range(length)]
                E = [[] for _ in range(length)]
                for head, layer in pairs:
                    for k in range(length):
                        _att = output.get_attention(left + k, head, layer)
                        _e = self.__entropy(_att)

                        if _e < self._config.gamma:
                            atts[k].append(_att)
                            E[k].append(self.__entropy(atts[k][-1]))

                for k in range(length):
                    _ei = [h for h, _ in sorted(
                        enumerate(E[k]), key=lambda x: x[1])]
                    _ei = _ei[:self._config.top_k]
                    atts[k] = [atts[k][h] for h in _ei]
                    E[k] = [E[k][h] for h in _ei]

                for att in atts:
                    probs[j] += self.__apply_attention_weights(box, att, height, width)
            if not probs:
                continue
            probs = torch.stack(probs, dim=0)
            ind = probs.sum(dim=-1).argmax().item()
            matched.append((match_indices[ind], i, probs.sum(dim=-1)))
        return matched
