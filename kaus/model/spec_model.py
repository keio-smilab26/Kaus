import copy
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from .draft_network import Model
from .config import EConfig
from .kv_cache import initialize_past_key_values
from .qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as KVQwen2_5_VLForConditionalGeneration,
)
from .utils import *


class SpecModel(nn.Module):

    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        spec_model_path,
        total_token,
        depth,
        top_k,
        threshold,
        spec_layer_state_dict,
        num_q: int = 2,
        kv_max_length: int = None,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        if hasattr(base_model, "language_model"):
            base_model = base_model.language_model

        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )
        config = EConfig.from_pretrained(spec_model_path)
        with open(spec_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        self.spec_layer = Model(
            config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            num_q=num_q,
        )

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.spec_layer.diff_device = True
            if not low_memory:
                self.spec_layer.headweight = base_model.lm_head.weight.clone().to(
                    device
                )
            else:
                self.spec_layer.layer_device = device

        else:
            self.spec_layer.diff_device = False
        if spec_layer_state_dict is not None:
            # self.spec_layer.load_state_dict(spec_layer_state_dict, strict=True)
            missing_keys, unexpected_keys = self.spec_layer.load_state_dict(
                spec_layer_state_dict, strict=False
            )
            if len(missing_keys) > 0:
                print(f"missing_keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"unexpected_keys: {unexpected_keys}")
        self.spec_layer.to(self.base_model.dtype).to(device)
        self.spec_layer.init_tree()
        self.kv_max_length = kv_max_length

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        Type="LLaMA",
        base_model_path=None,
        spec_model_path=None,
        total_token=30,
        depth=3,
        top_k=8,
        threshold=1.0,
        num_q: int = 2,
        kv_max_length: int = None,
        **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        kwargs.setdefault("attn_implementation", "sdpa")
        if Type == "Qwen2_5_VLForConditionalGeneration":
            base_model = KVQwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Model type {Type} is not supported. Kaus supports Qwen2_5_VLForConditionalGeneration only."
            )

        configpath = os.path.join(spec_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download("JLKang/ViSpec-Qwen2.5-VL-7B-Instruct", "config.json")

        try:
            load_model_path = os.path.join(spec_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(spec_model_path, "pytorch_model.bin")
            spec_layer_state_dict = torch.load(
                load_model_path, map_location=base_model.device
            )
        except:
            from safetensors.torch import load, load_file

            load_model_path = os.path.join(spec_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(spec_model_path, "model.safetensors")
            with open(load_model_path, "rb") as f:
                spec_layer_state_dict = load(f.read())
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            spec_layer_state_dict,
            num_q=num_q,
            kv_max_length=kv_max_length,
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(
                    0, model.config.vocab_size - 200, (1, length)
                ).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.spec_layer.total_tokens = total_token - 1

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        inputs_embeds=None,
        output_real_hidden=False,
        **kwargs,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_hidden_states=True,
                **kwargs,
            )
            if output_orig:
                orig = outputs.logits
            hidden_states = outputs.hidden_states[-1]

        if output_real_hidden:
            return None, orig, hidden_states, outputs.hidden_states
        if output_orig:
            return None, orig, hidden_states
        else:
            return None, hidden_states

    @torch.no_grad()
    def specgenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        inputs_embeds=None,
        return_acceptance_len=False,
        return_decode_time=False,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        max_length = max_length - self.spec_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.spec_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model, max_length=self.kv_max_length)
            except:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model, max_length=self.kv_max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        embed_weights = None
        special_image_mask = None
        if self.base_model.config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            pixel_values_videos = kwargs.get("pixel_values_videos")
            video_grid_thw = kwargs.get("video_grid_thw")
            second_per_grid_ts = kwargs.get("second_per_grid_ts")

            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(
                        pixel_values, grid_thw=image_grid_thw
                    )
                    n_image_tokens = (
                        (input_ids == self.base_model.config.image_token_id)
                        .sum()
                        .item()
                    )
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == self.base_model.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask, image_embeds
                    )

                    special_image_mask = mask

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(
                        self.base_model.visual.dtype
                    )
                    video_embeds = self.base_model.visual(
                        pixel_values_videos, grid_thw=video_grid_thw
                    )
                    n_video_tokens = (
                        (input_ids == self.base_model.config.video_token_id)
                        .sum()
                        .item()
                    )
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )

                    mask = input_ids == self.base_model.config.video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)

                    video_embeds = video_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        video_mask, video_embeds
                    )

                    # special_video_mask = mask
                    special_image_mask = mask

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            logits,
            hidden_state,
            sample_token,
        ) = initialize_tree(
            input_ids,
            self,
            past_key_values,
            logits_processor,
            inputs_embeds,
            embed_weights,
            image_mask=special_image_mask,
            **kwargs,
        )
        new_token = 0

        if return_acceptance_len:
            acceptance_len = []
        if return_decode_time:
            torch.cuda.synchronize()
            start_time = time.time()
        prob_history = [] if log else None

        for idx in range(max_length):
            # with Timer("all"):
            if not hasattr(self.base_model, "language_model"):
                self.base_model.model.tree_mask = tree_mask
            else:
                self.base_model.language_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            if log:
                # 承認された長さ（中間トークン数）
                acc_len = int(accept_length)
                # best_candidate が指す「採用されたパス」に沿ってロジットを回収
                # i は系列内の位置
                for i in range(acc_len):
                    # i 番目のステップにおける予測分布を抽出
                    # (これは次のトークン candidates[best_candidate, i+1] を決めた瞬間の分布)
                    occured_logits = logits[best_candidate, i]
                    
                    if logits_processor is None:
                        prob_history.append(torch.softmax(occured_logits, dim=-1))
                    else:
                        # Processor適用が必要な場合はここで処理
                        p_logits = logits_processor(None, occured_logits.unsqueeze(0))[0]
                        prob_history.append(torch.softmax(p_logits, dim=-1))

                # 最後に、確定したトークンの「次」の分布を追加
                if logits_processor is None:
                    prob_history.append(torch.softmax(sample_p, dim=-1))
                else:
                    prob_history.append(sample_p)
            # print(accept_length)
            if return_acceptance_len:
                acceptance_len.append(int(accept_length))
            # with Timer("update_inference_inputs"):
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
                # inputs_embeds,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
                # inputs_embeds,
                # embed_weights,
                # image_mask=special_image_mask,
            )

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if new_token > max_new_tokens:
                break
        # if input_ids.shape[1] > max_length:
        #     break
        # if not log:
        #     return input_ids
        # else:
        #     return input_ids, new_token, idx

        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx, prob_history)
            # ni = int(special_image_mask.int().sum())
            # nt = input_len - ni + 1
            # n = input_len
            # d = self.spec_layer.embed_tokens.embedding_dim
            # i = self.spec_layer.layers[0].mlp.intermediate_size
            # flops = (
            #     4 * ni * d**2
            #     + 8 * ni * d
            #     + 4 * d**2
            #     + 12 * nt * d**2
            #     + 4 * nt**2 * d
            #     + 8 * nt * d * i
            # )
            # flops_ori = 12 * n * d**2 + 4 * n**2 * d + 4 * n * d * i
            # outputs += (new_token, idx, flops, flops_ori)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            torch.cuda.synchronize()
            end_time = time.time()
            outputs += (end_time - start_time,)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
