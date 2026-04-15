"""
Qwen3-VL 推理引擎封装
对外暴露:
  - predict_multi_image: 多图(多窗位拼图)同时输入
  - predict_text:        纯文本(用于 Reduce 阶段融合)
"""
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from configs.config import cfg

logger = logging.getLogger(__name__)


class QwenVLEngine:
    def __init__(self):
        logger.info(f"Loading {cfg.model_id} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model.eval()

    @torch.no_grad()
    def _generate(self, messages: List[dict]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
            temperature=cfg.temperature,
            do_sample=True,
        )
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def predict_multi_image(
        self, images_np: List[np.ndarray], prompt: str
    ) -> str:
        """
        输入多张拼图(每张代表一个窗位), 输出模型生成文本
        images_np: list of (H, W, 3) uint8 numpy 数组
        """
        content = []
        for img_np in images_np:
            pil = Image.fromarray(img_np.astype(np.uint8))
            content.append({"type": "image", "image": pil})
        content.append({"type": "text", "text": prompt})
        return self._generate([{"role": "user", "content": content}])

    def predict_text(self, prompt: str) -> str:
        return self._generate(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )
