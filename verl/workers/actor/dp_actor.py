# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    EDGE_DISTILLATION_BATCH_KEYS,
    agg_loss,
    compute_edge_distillation_loss,
    compute_edge_teacher_topk_distillation_loss,
    compute_policy_loss,
    compute_policy_loss_gspo,
    kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_padded_response_token_selection_ids(
    token_selection_ids: torch.Tensor,
    batch_size: int,
    seqlen: int,
    response_length: int,
) -> torch.Tensor:
    """Place response-token candidate ids at their next-token logit positions."""
    if token_selection_ids.dim() != 3 or token_selection_ids.shape[:2] != (batch_size, response_length):
        raise ValueError(
            "token_selection_ids must have shape [batch, response_length, top_k], "
            f"got {tuple(token_selection_ids.shape)}"
        )
    padded_ids = torch.zeros(
        (batch_size, seqlen, token_selection_ids.size(-1)),
        dtype=token_selection_ids.dtype,
        device=token_selection_ids.device,
    )
    # Response labels occupy the final R input positions. Their predictor
    # logits are one position earlier: [-R-1:-1].
    padded_ids[:, -response_length - 1 : -1] = token_selection_ids
    return padded_ids


class _MemoryEfficientLogSumExp(torch.autograd.Function):
    """Compute fp32 logsumexp over the vocabulary without storing an extra fp32 [N, V] tensor."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        lse = torch.logsumexp(logits.float(), dim=-1)
        ctx.save_for_backward(logits, lse)
        return lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, lse = ctx.saved_tensors
        grad = torch.exp(logits.float() - lse.unsqueeze(-1)) * grad_output.float().unsqueeze(-1)
        return grad.to(logits.dtype)


def _memory_efficient_logsumexp(logits: torch.Tensor) -> torch.Tensor:
    if logits.requires_grad:
        return _MemoryEfficientLogSumExp.apply(logits)
    return torch.logsumexp(logits.float(), dim=-1)


def _active_training_row_mask(
    batch_dict,
    *,
    tree_batch: bool = False,
    dedup: bool = False,
    edge_enabled: bool = False,
    sft_enabled: bool = False,
    path_tier: int = 2,
):
    """Identify rows with at least one non-zero gradient signal (PG, OPSD, or SFT)."""
    if not tree_batch or os.environ.get("T75_SKIP_DEAD_ROWS", "1") == "0":
        return None
    if "responses" not in batch_dict:
        return None
    bsz = batch_dict["responses"].shape[0]
    device = batch_dict["responses"].device
    valid_row = (
        ~batch_dict["is_padding_copy"].to(dtype=torch.bool, device=device)
        if "is_padding_copy" in batch_dict
        else torch.ones(bsz, dtype=torch.bool, device=device)
    )
    if dedup and "A_pos" in batch_dict and "A_neg" in batch_dict:
        has_pg = (batch_dict["A_pos"] != 0) | (batch_dict["A_neg"] != 0)
    elif "advantages" in batch_dict:
        adv = batch_dict["advantages"]
        has_pg = (adv != 0).any(dim=-1) if adv.dim() > 1 else (adv != 0)
    elif "tree_advantage" in batch_dict:
        has_pg = batch_dict["tree_advantage"] != 0
    else:
        return None
    has_pg = has_pg.to(device=device, dtype=torch.bool)

    if edge_enabled and "edge_distill_weight" in batch_dict:
        kd_w = batch_dict["edge_distill_weight"].to(device=device)
        if "kd_canonical" in batch_dict:
            kd_w = kd_w * batch_dict["kd_canonical"].to(device=device)
        if int(path_tier) == 1 and "kd_path_has_T" in batch_dict:
            kd_w = kd_w * (~batch_dict["kd_path_has_T"].to(device=device, dtype=torch.bool))
        has_kd = kd_w > 0
    else:
        has_kd = torch.zeros(bsz, dtype=torch.bool, device=device)

    if sft_enabled and "sft_weight" in batch_dict:
        has_sft = batch_dict["sft_weight"].to(device=device) > 0
    else:
        has_sft = torch.zeros(bsz, dtype=torch.bool, device=device)

    return valid_row & (has_pg | has_kd | has_sft)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")
        # T062: project hidden states to vocabulary only at positions that predict response tokens.
        self.response_only_logits = bool(self.config.get("response_only_logits", False))
        print(f"Actor response_only_logits={self.response_only_logits}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()
        from verl.utils.replica_sync import uses_explicit_replica_sync
        self._explicit_replica_sync = uses_explicit_replica_sync(self.actor_module)
        # FSDP sharded ranks must issue matching parameter collectives even when
        # sparse KD/SFT gates differ between ranks. The actor handles inactive
        # branches with zero-contribution placeholder forwards and backwards.
        self._fsdp_sync_optional_forwards = (
            isinstance(self.actor_module, FSDP)
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
            and not self._explicit_replica_sync
        )
        if self._fsdp_sync_optional_forwards:
            print("T073_FSDP_OPTIONAL_FORWARD_SYNC enabled", flush=True)
        if self._explicit_replica_sync:
            if self.use_ulysses_sp or self.config.loss_agg_mode != "token-mean":
                raise ValueError("Explicit full-replica synchronization requires token-mean loss and SP=1")
            for parameter in self.actor_module.parameters():
                if parameter.requires_grad:
                    torch.distributed.broadcast(parameter.data, src=0)
            print("T069_SYNC explicit global replica averaging enabled", flush=True)

    def _forward_response_only_rmpad(
        self,
        input_ids,
        attention_mask,
        position_ids,
        response_length,
        temperature,
        calculate_entropy,
        calculate_log_probs,
        token_selection_top_k,
        token_selection_ids,
    ):
        """Remove-padding forward that applies lm_head only where logits are consumed.

        The default path materialises (total_nnz, vocab) logits for every prompt and
        response token and then keeps only the response slice [-R-1:-1]. In EDGE search
        rows the prompt (question + retrieved history) dominates total_nnz, so this
        computes exactly the same response-position values with far less memory and
        compute. Returned tensors have the same (bsz, R[, k]) shapes and zero padding
        at invalid positions as the default path.
        """
        batch_size, seqlen = input_ids.shape
        device = input_ids.device
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)
        labels_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)  # (total_nnz,)

        # Map the response predictor positions [-R-1:-1] of every row to rmpad indices.
        total_nnz = input_ids_rmpad.size(1)
        inverse = torch.full((batch_size * seqlen,), -1, dtype=torch.long, device=device)
        inverse[indices] = torch.arange(total_nnz, device=device)
        seq_pos = torch.arange(seqlen - response_length - 1, seqlen - 1, device=device)
        flat = torch.arange(batch_size, device=device).unsqueeze(1) * seqlen + seq_pos.unsqueeze(0)  # (bsz, R)
        rm_idx = inverse[flat]
        valid = rm_idx >= 0
        row_idx, resp_idx = valid.nonzero(as_tuple=True)
        keep_idx = rm_idx[row_idx, resp_idx]  # ascending

        output = self.actor_module(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids_rmpad,
            use_cache=False,
            logits_to_keep=keep_idx,
        )
        logits = output.logits.squeeze(0)  # (n_keep, vocab)
        logits.div_(temperature)
        labels = labels_rolled[keep_idx]

        entropy = log_probs = selected_logits = selected_token_ids = None
        sel_logits_kept = sel_ids_kept = None
        if token_selection_top_k is not None:
            if token_selection_ids is None:
                sel_logits_kept, sel_ids_kept = torch.topk(logits, k=min(token_selection_top_k, logits.size(-1)), dim=-1)
            else:
                if token_selection_ids.dim() != 3 or token_selection_ids.shape[:2] != (batch_size, response_length):
                    raise ValueError(
                        "token_selection_ids must have shape [batch, response_length, top_k], "
                        f"got {tuple(token_selection_ids.shape)}"
                    )
                sel_ids_kept = token_selection_ids[row_idx, resp_idx]
                sel_logits_kept = logits.gather(dim=-1, index=sel_ids_kept)

        if sel_logits_kept is not None:
            lse_kept = _memory_efficient_logsumexp(logits)
            self._last_selected_lse = torch.zeros((batch_size, response_length), dtype=torch.float32, device=device).index_put((row_idx, resp_idx), lse_kept)
            with torch.no_grad():
                mass_kept = torch.exp(torch.logsumexp(sel_logits_kept.float(), -1) - lse_kept.detach()).clamp(0.0, 1.0)
                self._last_selected_mass = torch.zeros((batch_size, response_length), device=device).index_put((row_idx, resp_idx), mass_kept)

        if calculate_log_probs:
            inplace_backward = not (calculate_entropy or token_selection_top_k is not None)
            lp_kept = logprobs_from_logits(logits=logits, labels=labels, inplace_backward=inplace_backward)
            log_probs = torch.zeros((batch_size, response_length), dtype=lp_kept.dtype, device=device).index_put((row_idx, resp_idx), lp_kept)
        if calculate_entropy:
            ent_kept = self.compute_entropy_from_logits(logits)
            entropy = torch.zeros((batch_size, response_length), dtype=ent_kept.dtype, device=device).index_put((row_idx, resp_idx), ent_kept)
        if sel_logits_kept is not None:
            k = sel_logits_kept.size(-1)
            selected_logits = torch.zeros((batch_size, response_length, k), dtype=sel_logits_kept.dtype, device=device).index_put((row_idx, resp_idx), sel_logits_kept)
            selected_token_ids = torch.zeros((batch_size, response_length, k), dtype=sel_ids_kept.dtype, device=device).index_put((row_idx, resp_idx), sel_ids_kept)
        return entropy, log_probs, selected_logits, selected_token_ids

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        calculate_log_probs: bool = True,
        token_selection_top_k: Optional[int] = None,
        token_selection_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            selected_logits: # (bs, response_len, top_k), if token selection is requested
            selected_token_ids: # (bs, response_len, top_k), if teacher top-k is requested
        """
        response_length = micro_batch["responses"].size(-1)
        if token_selection_top_k is not None:
            if token_selection_top_k < 1:
                raise ValueError("token_selection_top_k must be positive")
            if self.use_fused_kernels:
                raise ValueError("Token-selection distillation does not support fused kernels because logits are unavailable")
            if self.use_ulysses_sp:
                raise ValueError("Token-selection distillation does not support Ulysses sequence parallelism")
        if token_selection_ids is not None and token_selection_top_k is None:
            raise ValueError("token_selection_ids requires token_selection_top_k")

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            log_probs = None
            selected_logits = None
            selected_token_ids = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if (
                self.use_remove_padding
                and self.response_only_logits
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not multi_modal_inputs
                and position_ids.dim() == 2
            ):
                entropy, log_probs, selected_logits, selected_token_ids = self._forward_response_only_rmpad(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    response_length=response_length,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    calculate_log_probs=calculate_log_probs,
                    token_selection_top_k=token_selection_top_k,
                    token_selection_ids=token_selection_ids,
                )
            elif self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    if token_selection_top_k is not None:
                        if token_selection_ids is None:
                            selected_logits_rmpad, selected_token_ids_rmpad = torch.topk(
                                logits_rmpad,
                                k=min(token_selection_top_k, logits_rmpad.size(-1)),
                                dim=-1,
                            )
                        else:
                            selected_token_ids_rmpad = _build_padded_response_token_selection_ids(
                                token_selection_ids, batch_size, seqlen, response_length
                            )
                            selected_token_ids_rmpad = index_first_axis(
                                rearrange(selected_token_ids_rmpad, "b s k -> (b s) k"), indices
                            )
                            selected_logits_rmpad = logits_rmpad.gather(dim=-1, index=selected_token_ids_rmpad)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    if calculate_log_probs:
                        # FlashAttention's in-place cross-entropy backward is
                        # unsafe when response logits also feed token selection.
                        inplace_backward = not (calculate_entropy or token_selection_top_k is not None)
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp and calculate_log_probs:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_log_probs:
                    full_log_probs = pad_input(
                        hidden_states=log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if token_selection_top_k is not None and not self.use_fused_kernels:
                    lse_rmpad = _memory_efficient_logsumexp(logits_rmpad)
                    full_lse = pad_input(
                        hidden_states=lse_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    self._last_selected_lse = full_lse.squeeze(-1)[:, -response_length - 1 : -1]
                    full_selected_logits = pad_input(
                        hidden_states=selected_logits_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_selected_token_ids = pad_input(
                        hidden_states=selected_token_ids_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    selected_logits = full_selected_logits[:, -response_length - 1 : -1]
                    selected_token_ids = full_selected_token_ids[:, -response_length - 1 : -1]
                    with torch.no_grad():
                        self._last_selected_mass = torch.exp(
                            torch.logsumexp(selected_logits.float(), -1) - self._last_selected_lse.detach()
                        ).clamp(0.0, 1.0)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if token_selection_top_k is not None:
                        if token_selection_ids is None:
                            selected_logits, selected_token_ids = torch.topk(
                                logits,
                                k=min(token_selection_top_k, logits.size(-1)),
                                dim=-1,
                            )
                        else:
                            if token_selection_ids.dim() != 3 or token_selection_ids.shape[:2] != (batch_size, response_length):
                                raise ValueError(
                                    "token_selection_ids must have shape [batch, response_length, top_k], "
                                    f"got {tuple(token_selection_ids.shape)}"
                                )
                            selected_token_ids = token_selection_ids
                            selected_logits = logits.gather(dim=-1, index=selected_token_ids)
                        self._last_selected_lse = _memory_efficient_logsumexp(logits)
                        with torch.no_grad():
                            self._last_selected_mass = torch.exp(
                                torch.logsumexp(selected_logits.float(), -1) - self._last_selected_lse.detach()
                            ).clamp(0.0, 1.0)
                    if calculate_log_probs:
                        log_probs = logprobs_from_logits(
                            logits,
                            micro_batch["responses"],
                            # Keep the cross-entropy backward out-of-place when
                            # selected logits are another graph consumer.
                            inplace_backward=not (calculate_entropy or token_selection_top_k is not None),
                        )
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, selected_logits, selected_token_ids

    def _optimizer_step(self, diagnostic_component=None):
        assert self.config.grad_clip is not None
        if diagnostic_component is not None:
            from verl.utils.replica_sync import average_replica_gradients
            average_replica_gradients(self.actor_module)
            gradients = [
                p.grad.detach().reshape(-1).float() if p.grad is not None
                else torch.zeros(p.numel(), device=p.device, dtype=torch.float32)
                for p in self.actor_module.parameters() if p.requires_grad
            ]
            vector = torch.cat(gradients)
            norm = torch.linalg.vector_norm(vector)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite diagnostic gradient")
            self._last_diagnostic_gradient = vector.cpu()
            return norm

        if getattr(self, "_explicit_replica_sync", False):
            from verl.utils.replica_sync import average_replica_gradients
            average_replica_gradients(self.actor_module)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            if getattr(self, "_explicit_replica_sync", False):
                raise FloatingPointError("Non-finite globally synchronized actor gradient")
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
            self._optimizer_steps_this_update = getattr(self, "_optimizer_steps_this_update", 0) + 1
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    @torch.no_grad()
    def _edge_teacher_forward_active(self, data, responses, temperature, top_k=None):
        """Teacher (experience-conditioned) forward restricted to gate-active rows.

        Both EDGE distillation losses multiply by active_mask = response_mask * (weight > 0),
        so inactive rows never reach the loss.  Their teacher outputs are filled with zeros
        (finite, masked away); the loss is that of the full-batch teacher pass up to
        kernel-level float noise from packing fewer rows.  Set actor.edge_teacher_active_only=False to restore it.
        Returns (teacher_log_prob | None, teacher_topk_logits | None, teacher_topk_ids | None).
        """
        bsz, resp_len = responses.shape
        device = responses.device
        if self.config.get("edge_teacher_active_only", True):
            active = torch.gt(data["edge_distill_weight"].to(device), 0).nonzero(as_tuple=True)[0]
        else:
            active = torch.arange(bsz, device=device)
        n_active = int(active.numel())
        self._edge_teacher_rows = getattr(self, "_edge_teacher_rows", 0) + n_active
        self._edge_teacher_rows_total = getattr(self, "_edge_teacher_rows_total", 0) + bsz

        k = top_k
        self._last_teacher_lse = None
        log_prob_full = None if top_k is not None else torch.zeros((bsz, resp_len), dtype=torch.float32, device=device)
        logits_full = ids_full = None
        if top_k is not None:
            ids_full = torch.zeros((bsz, resp_len, k), dtype=torch.long, device=device)
        # In sharded FSDP a rank with zero active KD rows must still issue the
        # teacher forward. Otherwise active ranks do a parameter all-gather here
        # while inactive ranks start the student forward/backward instead.
        dummy_for_collective = n_active == 0 and getattr(self, "_fsdp_sync_optional_forwards", False)
        if n_active == 0 and not dummy_for_collective:
            if top_k is not None:
                logits_full = torch.zeros((bsz, resp_len, k), dtype=torch.float32, device=device)
                self._last_teacher_lse = torch.zeros((bsz, resp_len), dtype=torch.float32, device=device)
            return log_prob_full, logits_full, ids_full
        if dummy_for_collective:
            active = torch.zeros((1,), dtype=torch.long, device=device)

        teacher_batch = {
            "responses": responses.index_select(0, active),
            "input_ids": data["edge_teacher_input_ids"].index_select(0, active),
            "attention_mask": data["edge_teacher_attention_mask"].index_select(0, active),
            "position_ids": data["edge_teacher_position_ids"].index_select(0, active),
        }
        if top_k is not None:
            self._last_selected_lse = None
            _, _, sel_logits, sel_ids = self._forward_micro_batch(
                micro_batch=teacher_batch, temperature=temperature, calculate_entropy=False,
                calculate_log_probs=False, token_selection_top_k=top_k,
            )
            k = sel_logits.size(-1)
            logits_full = torch.zeros((bsz, resp_len, k), dtype=sel_logits.dtype, device=device)
            logits_full.index_copy_(0, active, sel_logits)
            ids_full = torch.zeros((bsz, resp_len, k), dtype=sel_ids.dtype, device=device)
            ids_full.index_copy_(0, active, sel_ids)
            if getattr(self, "_last_selected_lse", None) is not None and not dummy_for_collective:
                teacher_lse_full = torch.zeros((bsz, resp_len), dtype=torch.float32, device=device)
                teacher_lse_full.index_copy_(0, active, self._last_selected_lse.detach().float())
                self._last_teacher_lse = teacher_lse_full
        else:
            _, lp, _, _ = self._forward_micro_batch(
                micro_batch=teacher_batch, temperature=temperature, calculate_entropy=False,
            )
            log_prob_full = torch.zeros((bsz, resp_len), dtype=lp.dtype, device=device)
            log_prob_full.index_copy_(0, active, lp)
        if dummy_for_collective:
            # Preserve the original zero-KD behavior. The placeholder forward
            # exists only to align FSDP collectives and must not affect loss.
            return (torch.zeros_like(log_prob_full) if log_prob_full is not None else None), None, None
        return log_prob_full, logits_full, ids_full

    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        if (
            bool(data.meta_info.get("tree_mode", False))
            and not bool(data.meta_info.get("_skip_dead_rows_done", False))
            and not getattr(self, "_fsdp_sync_optional_forwards", False)
            and not getattr(self, "use_ulysses_sp", False)
            and "multi_modal_inputs" not in data.non_tensor_batch
        ):
            edge_cfg_lp = self.config.get("edge_distillation", {}) or {}
            sft_cfg_lp = self.config.get("success_sft", {}) or {}
            active_mask = _active_training_row_mask(
                data.batch,
                tree_batch=True,
                dedup=bool(data.meta_info.get("dedup_shared_prefix", False)),
                edge_enabled=bool(edge_cfg_lp.get("enable", False)),
                sft_enabled=bool(sft_cfg_lp.get("enable", False)),
                path_tier=int(edge_cfg_lp.get("path_tier", 2)),
            )
            if active_mask is not None and not bool(active_mask.all()):
                total_bsz, resp_len = data.batch["responses"].shape
                dev = data.batch["responses"].device
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                if active_idx.numel() == 0:
                    zero_lp = torch.zeros((total_bsz, resp_len), dtype=torch.float32, device=dev)
                    zero_ent = torch.zeros((total_bsz, resp_len), dtype=torch.float32, device=dev) if calculate_entropy else None
                    return zero_lp, zero_ent
                sub_data = data.select_idxs(active_idx)
                sub_data.meta_info = dict(data.meta_info, _skip_dead_rows_done=True)
                sub_lp, sub_ent = self.compute_log_prob(sub_data, calculate_entropy=calculate_entropy)
                full_lp = torch.zeros((total_bsz, resp_len), dtype=sub_lp.dtype, device=sub_lp.device)
                full_lp.index_copy_(0, active_idx.to(sub_lp.device), sub_lp)
                full_ent = None
                if calculate_entropy and sub_ent is not None:
                    full_ent = torch.zeros((total_bsz, resp_len), dtype=sub_ent.dtype, device=sub_ent.device)
                    full_ent.index_copy_(0, active_idx.to(sub_ent.device), sub_ent)
                return full_lp, full_ent

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _, _ = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._optimizer_steps_this_update = 0
        self._explicit_replica_sync = getattr(self, "_explicit_replica_sync", False)
        # Some lightweight test actors bypass __init__; treat optional FSDP
        # synchronization as disabled unless the FSDP constructor enabled it.
        self._fsdp_sync_optional_forwards = getattr(self, "_fsdp_sync_optional_forwards", False)
        diagnostic_component = data.meta_info.get("gradient_diagnostic_component")
        gradient_diagnostic_metrics = {}
        if (self._explicit_replica_sync and diagnostic_component is None
                and os.environ.get("ENABLE_GRADIENT_DIAGNOSTICS", "0") == "1"
                and int(data.meta_info.get("global_step", -1)) in (1, 50, 100)
                and data.meta_info.get("gradient_diagnostic_dir")
                and self.config.get("edge_distillation", {}).get("enable", False)):
            from verl.utils.gradient_diagnostics import measure_actor_gradients
            gradient_diagnostic_metrics = measure_actor_gradients(self, data)
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        edge_cfg = self.config.get("edge_distillation", {})
        edge_distillation_requested = edge_cfg.get("enable", False)
        edge_missing_keys = [key for key in EDGE_DISTILLATION_BATCH_KEYS if key not in data.batch.keys()]
        edge_distillation_enabled = edge_distillation_requested and not edge_missing_keys
        edge_token_selection = edge_cfg.get("token_selection", "sampled")
        if edge_token_selection not in ("sampled", "teacher_topk"):
            raise ValueError(f"Unsupported EDGE token_selection: {edge_token_selection}")
        edge_top_k = edge_cfg.get("top_k", 20)
        if edge_token_selection == "teacher_topk":
            if not isinstance(edge_top_k, int) or isinstance(edge_top_k, bool) or edge_top_k < 2:
                raise ValueError("EDGE teacher_topk requires edge_distillation.top_k to be an integer >= 2")
            if self.use_fused_kernels:
                raise ValueError("EDGE teacher_topk does not support fused kernels because logits are unavailable")
            if self.use_ulysses_sp:
                raise ValueError("EDGE teacher_topk does not support Ulysses sequence parallelism")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if edge_distillation_enabled:
            select_keys.extend(EDGE_DISTILLATION_BATCH_KEYS)
        elif edge_distillation_requested:
            logger.warning("EDGE distillation skipped because batch is missing keys: %s", edge_missing_keys)
        tree_batch = "kd_student_input_ids" in data.batch
        sft_cfg = self.config.get("success_sft", {})
        sft_enabled = tree_batch and bool(sft_cfg.get("enable", False))
        sft_coef = float(sft_cfg.get("coef", 0.05))
        if sft_enabled:
            sft_keys = ["sft_weight", "sft_student_input_ids",
                        "sft_student_attention_mask", "sft_student_position_ids"]
            missing = [key for key in sft_keys if key not in data.batch]
            if missing:
                raise ValueError("Success SFT enabled but rollout lacks fields: " + str(missing))
            select_keys.extend(sft_keys)
        if tree_batch:
            select_keys.extend(["kd_student_input_ids", "kd_student_attention_mask", "kd_student_position_ids",
                "kd_canonical", "kd_path_has_T", "is_padding_copy"])
            # Even when OPSD is disabled, tree row padding and the common gate
            # masking path still consult this zero-valued tensor.
            if "edge_distill_weight" not in select_keys:
                select_keys.append("edge_distill_weight")
        dedup = "w_tok" in data.batch
        if dedup:
            select_keys.extend(["w_tok", "A_pos", "A_neg", "positive_adv_count", "negative_adv_count", "dedup_mini", "dedup_mini_weight"])
        self._tree_monitoring = tree_batch
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            fixed_n_mini = int(self.config.get("fixed_num_minibatches", 0) or 0)
            if dedup:
                dataloader = [batch[batch["dedup_mini"] == mini] for mini in batch["dedup_mini"].unique(sorted=True)]
            elif fixed_n_mini > 0:
                n_chunks = min(fixed_n_mini, len(batch))
                dataloader = list(batch.chunk(n_chunks)) if n_chunks > 0 else []
            else:
                dataloader = batch.split(self.config.ppo_mini_batch_size)

        if self._explicit_replica_sync or self._fsdp_sync_optional_forwards:
            count = torch.tensor([len(dataloader)], dtype=torch.int64, device=get_torch_device().current_device())
            counts = [torch.empty_like(count) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(counts, count)
            if len({int(value.item()) for value in counts}) != 1:
                raise RuntimeError(f"Actor minibatch counts differ across replica ranks: {[int(value.item()) for value in counts]}")
        trace_fsdp_first_step = (
            self._fsdp_sync_optional_forwards and
            int(data.meta_info.get("global_step", -1)) == 1
        )
        metrics = {}
        from verl.utils.actor_metric_totals import ActorMetricTotals
        statistic_totals = ActorMetricTotals(get_torch_device().current_device())
        micro_batch_count = 0
        if edge_distillation_requested:
            append_to_dict(metrics, {"edge/skipped_missing_inputs": float(bool(edge_missing_keys))})
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                self.actor_optimizer.zero_grad()
                # 1. Compute global/minibatch denominators BEFORE any dead-row filtering
                #    so every loss term retains 100% identical scaling.
                if self._explicit_replica_sync:
                    from verl.utils.replica_sync import summed_denominators
                    mb = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    mask = mb["loss_mask"][:, -mb["responses"].shape[1]:] if multi_turn else mb["attention_mask"][:, -mb["responses"].shape[1]:]
                    if tree_batch:
                        mask = mask * (~mb["is_padding_copy"]).unsqueeze(-1)
                    row_weights = mb["w_tok"] if dedup else torch.ones(len(mask), device=mask.device)
                    kd_weights = mb["edge_distill_weight"] if edge_distillation_enabled else torch.zeros(len(mask), device=mask.device)
                    if tree_batch and edge_distillation_enabled:
                        kd_weights = kd_weights * mb["kd_canonical"] * (~mb["is_padding_copy"])
                        if int(edge_cfg.get("path_tier", 2)) == 1:
                            kd_weights = kd_weights * (~mb["kd_path_has_T"])
                    global_denominators = summed_denominators(
                        mask, row_weights, kd_weights,
                        sft_weights=mb["sft_weight"] if sft_enabled else None,
                    )
                if tree_batch and edge_distillation_enabled and not self._explicit_replica_sync:
                    mb_kd = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    kd_mask = (mb_kd["loss_mask"] if multi_turn else mb_kd["attention_mask"])[:, -mb_kd["responses"].shape[1]:]
                    kd_mask = kd_mask * (~mb_kd["is_padding_copy"]).unsqueeze(-1)
                    kd_active = (mb_kd["edge_distill_weight"] > 0) & mb_kd["kd_canonical"]
                    if int(edge_cfg.get("path_tier", 2)) == 1:
                        kd_active = kd_active & (~mb_kd["kd_path_has_T"])
                    mini_kd_denominator = (kd_mask * kd_active.unsqueeze(-1)).sum().clamp_min(1)
                if sft_enabled and not self._explicit_replica_sync:
                    mb = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    mask_for_sft = (mb["loss_mask"] if multi_turn else mb["attention_mask"])[:, -mb["responses"].shape[1]:]
                    if tree_batch:
                        mask_for_sft = mask_for_sft * (~mb["is_padding_copy"]).unsqueeze(-1)
                    mini_sft_denominator = (mask_for_sft * mb["sft_weight"].unsqueeze(-1)).sum().clamp_min(1)

                # 2. Lossless dead-row skipping when global denominators cancel local token counts
                if (
                    tree_batch
                    and self._explicit_replica_sync
                    and not self._fsdp_sync_optional_forwards
                    and not has_multi_modal_inputs
                    and not self.use_ulysses_sp
                    and diagnostic_component is None
                ):
                    is_proto = isinstance(mini_batch, DataProto)
                    mb_t = mini_batch.batch if is_proto else mini_batch
                    active_mask = _active_training_row_mask(
                        mb_t,
                        tree_batch=True,
                        dedup=dedup,
                        edge_enabled=edge_distillation_enabled,
                        sft_enabled=sft_enabled,
                        path_tier=int(edge_cfg.get("path_tier", 2)) if edge_distillation_enabled else 2,
                    )
                    if active_mask is not None and not bool(active_mask.all()):
                        dev = get_torch_device().current_device()
                        r_len = mb_t["responses"].size(1)
                        resp_m = (mb_t["loss_mask"] if multi_turn else mb_t["attention_mask"])[:, -r_len:].to(dev)
                        resp_m = resp_m * (~mb_t["is_padding_copy"].to(dev)).unsqueeze(-1)
                        w_t = mb_t["w_tok"].to(dev) if dedup else torch.ones(len(resp_m), device=dev)
                        dead_d = (~active_mask).to(dev)
                        dead_expanded = resp_m[dead_d] * w_t[dead_d].unsqueeze(-1)
                        dead_total = dead_expanded.sum()
                        dead_rows = (w_t[dead_d] * resp_m[dead_d].any(-1)).sum()
                        statistic_totals.add("actor/expanded_tokens", dead_total)
                        statistic_totals.add("kd/expanded_rows", dead_rows)
                        statistic_totals.ratio("actor/response_length_expanded", dead_total, dead_rows)
                        statistic_totals.ratio("actor/pg_loss_expanded", dead_total * 0.0, dead_total)
                        statistic_totals.ratio("actor/pg_clipfrac", dead_total * 0.0, dead_total)
                        statistic_totals.ratio("actor/pg_clipfrac_lower", dead_total * 0.0, dead_total)
                        statistic_totals.ratio("actor/ppo_kl", dead_total * 0.0, dead_total)
                        if edge_distillation_enabled:
                            statistic_totals.ratio("kd/active_token_rate", dead_total * 0.0, dead_total)
                            statistic_totals.ratio("kd/effective_weight_sum_over_N", dead_total * 0.0, dead_total)
                            statistic_totals.ratio("kd/weighted_loss_all_tokens", dead_total * 0.0, dead_total)
                            statistic_totals.ratio("edge/actor_active_row_fraction", dead_rows * 0.0, dead_rows)
                        dead_real_rows = int((~mb_t["is_padding_copy"][(~active_mask).to(mb_t["is_padding_copy"].device)]).sum())
                        self._edge_teacher_rows_total = getattr(self, "_edge_teacher_rows_total", 0) + dead_real_rows

                        def _slice_mb(src, idx_t):
                            if is_proto:
                                return src.select_idxs(idx_t.cpu())
                            return src[idx_t.to(src.device)].clone()

                        def _zero_mb(dst):
                            target = dst.batch if is_proto else dst
                            for z_key in ("w_tok", "positive_adv_count", "negative_adv_count", "A_pos", "A_neg",
                                          "kd_weight", "edge_distill_weight", "sft_weight", "advantages"):
                                if z_key in target:
                                    target[z_key].zero_()
                            if "kd_canonical" in target:
                                target["kd_canonical"].fill_(False)
                            if "is_padding_copy" in target:
                                target["is_padding_copy"].fill_(True)
                            return dst

                        active_idx = active_mask.nonzero(as_tuple=True)[0]
                        if active_idx.numel() == 0:
                            dummy_n = 1 if self.config.use_dynamic_bsz else self.config.ppo_micro_batch_size_per_gpu
                            z_idx = torch.zeros(dummy_n, dtype=torch.long)
                            mini_batch = _zero_mb(_slice_mb(mini_batch, z_idx))
                        else:
                            selected_mb = _slice_mb(mini_batch, active_idx)
                            if not self.config.use_dynamic_bsz:
                                rem = len(selected_mb) % self.config.ppo_micro_batch_size_per_gpu
                                if rem != 0:
                                    pad_n = self.config.ppo_micro_batch_size_per_gpu - rem
                                    pad_mb = _zero_mb(_slice_mb(selected_mb, torch.zeros(pad_n, dtype=torch.long)))
                                    selected_mb = DataProto.concat([selected_mb, pad_mb]) if is_proto else torch.cat([selected_mb, pad_mb], dim=0)
                            mini_batch = selected_mb

                # 3. Split mini_batch into micro_batches
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self._fsdp_sync_optional_forwards:
                    micro_count_per_rank = torch.tensor(
                        [len(micro_batches)], dtype=torch.int64,
                        device=get_torch_device().current_device()
                    )
                    micro_counts = [torch.empty_like(micro_count_per_rank)
                                    for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(micro_counts, micro_count_per_rank)
                    if len({int(value.item()) for value in micro_counts}) != 1:
                        raise RuntimeError(
                            f"Actor microbatch counts differ across ranks in mini {batch_idx}: "
                            f"{[int(value.item()) for value in micro_counts]}"
                        )

                from contextlib import nullcontext
                accumulation_context = self.actor_module.no_sync() if self._explicit_replica_sync else nullcontext()
                with accumulation_context:
                    for data in micro_batches:
                        micro_batch_count += 1
                        # Support all hardwares
                        if isinstance(data, DataProto):
                            data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                        else:
                            data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                        responses = data["responses"]
                        response_length = responses.size(1)
                        attention_mask = data["attention_mask"]
                        if multi_turn:
                            response_mask = data["loss_mask"][:, -response_length:]
                        else:
                            response_mask = attention_mask[:, -response_length:]

                        if tree_batch:
                            response_mask = response_mask * (~data["is_padding_copy"]).unsqueeze(-1)
                            data["edge_distill_weight"] *= data["kd_canonical"] * (~data["is_padding_copy"])
                            if int(edge_cfg.get("path_tier", 2)) == 1:
                                data["edge_distill_weight"] *= ~data["kd_path_has_T"]
                        old_log_prob = data["old_log_probs"]
                        advantages = data["advantages"]

                        clip_ratio = self.config.clip_ratio
                        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        teacher_topk_logits = None
                        teacher_topk_ids = None
                        active_tree_kd = tree_batch and bool((data["edge_distill_weight"] > 0).any())
                        if trace_fsdp_first_step:
                            print(
                                f"T073_FSDP_MICRO rank={torch.distributed.get_rank()} "
                                f"mini={batch_idx} micro={micro_batch_count} phase=start "
                                f"kd_active={int(active_tree_kd)} "
                                f"sft_active={int((data['sft_weight'] > 0).sum()) if sft_enabled else 0}",
                                flush=True,
                            )
                        if (tree_batch and edge_distillation_enabled and not active_tree_kd
                                and not self._fsdp_sync_optional_forwards):
                            # No teacher forward is needed, but these GRPO rows belong in
                            # the teacher-row-fraction denominator.
                            self._edge_teacher_rows = getattr(self, "_edge_teacher_rows", 0)
                            self._edge_teacher_rows_total = getattr(self, "_edge_teacher_rows_total", 0) + responses.size(0)
                        self._last_teacher_lse = None
                        self._last_selected_lse = None
                        if (edge_distillation_enabled and edge_token_selection == "teacher_topk"
                                and (not tree_batch or active_tree_kd or self._fsdp_sync_optional_forwards)):
                            # Same actor module under no_grad, evaluated before the student,
                            # but only on rows the gate activated (weight > 0): every other row
                            # is masked out of the distillation loss, so its teacher pass is waste.
                            _, teacher_topk_logits, teacher_topk_ids = self._edge_teacher_forward_active(
                                data, responses, temperature, top_k=edge_top_k
                            )
                        teacher_topk_lse = getattr(self, "_last_teacher_lse", None)

                        # all return: (bsz, response_length)
                        self._last_selected_lse = None
                        calculate_entropy = (entropy_coeff != 0) or (
                            os.environ.get("T75_FORCE_ACTOR_ENTROPY", "0") == "1"
                        )
                        entropy, log_prob, student_topk_logits, _ = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            token_selection_top_k=edge_top_k if teacher_topk_ids is not None else None,
                            token_selection_ids=teacher_topk_ids,
                        )

                        kd_noexp_dummy_loss = None
                        tree_student_lse = None
                        if tree_batch and (active_tree_kd or (edge_distillation_enabled and self._fsdp_sync_optional_forwards)):
                            tree_student_logits = student_topk_logits
                            tree_student_lse = getattr(self, "_last_selected_lse", None)
                            tree_student_mass = getattr(self, "_last_selected_mass", None)
                            tree_gap = torch.zeros_like(log_prob)
                            extra = ((data["edge_distill_weight"] > 0) & data["kd_path_has_T"]).nonzero(as_tuple=True)[0]
                            kd_dummy = extra.numel() == 0 and self._fsdp_sync_optional_forwards
                            if extra.numel() or kd_dummy:
                                forward_ids = extra if not kd_dummy else torch.zeros(
                                    (1,), dtype=torch.long, device=responses.device
                                )
                                noexp = {
                                    "input_ids": data["kd_student_input_ids"].index_select(0, forward_ids),
                                    "attention_mask": data["kd_student_attention_mask"].index_select(0, forward_ids),
                                    "position_ids": data["kd_student_position_ids"].index_select(0, forward_ids),
                                    "responses": responses.index_select(0, forward_ids),
                                }
                                selected_ids = (
                                    teacher_topk_ids.index_select(0, forward_ids)
                                    if teacher_topk_ids is not None else
                                    torch.zeros(
                                        (len(forward_ids), responses.size(1), edge_top_k),
                                        device=responses.device, dtype=torch.long
                                    )
                                )
                                self._last_selected_lse = None
                                _, noexp_lp, noexp_logits, _ = self._forward_micro_batch(
                                    noexp, temperature, token_selection_top_k=edge_top_k,
                                    token_selection_ids=selected_ids
                                )
                                if kd_dummy:
                                    # Keep the optional FSDP graph alive for a matching backward,
                                    # without changing KD loss or creating a training signal.
                                    kd_noexp_dummy_loss = noexp_lp.sum() * 0.0
                                else:
                                    tree_student_logits = student_topk_logits.index_copy(0, extra, noexp_logits)
                                    if tree_student_lse is not None and getattr(self, "_last_selected_lse", None) is not None:
                                        tree_student_lse = tree_student_lse.index_copy(0, extra, self._last_selected_lse)
                                    if tree_student_mass is not None and getattr(self, "_last_selected_mass", None) is not None:
                                        tree_student_mass = tree_student_mass.index_copy(0, extra, self._last_selected_mass)
                                    tree_gap = tree_gap.index_copy(
                                        0, extra, log_prob.index_select(0, extra) - noexp_lp
                                    )

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        if loss_mode == "vanilla":
                            policy_loss_fn = compute_policy_loss
                        elif loss_mode == "gspo":
                            policy_loss_fn = compute_policy_loss_gspo
                        else:
                            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                        if dedup:
                            from agent_system.tree.dedup import weighted_node_policy_loss
                            pg_loss = weighted_node_policy_loss(old_log_prob,log_prob,data["A_pos"],data["A_neg"],response_mask,data["w_tok"],clip_ratio_low,clip_ratio_high,clip_ratio_c)

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = data["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if edge_distillation_enabled:
                            if tree_batch:
                                if active_tree_kd:
                                    from agent_system.tree.loss import tree_topk_loss
                                    edge_loss, tree_kl, tree_gradient = tree_topk_loss(
                                        tree_student_logits,
                                        teacher_topk_logits,
                                        response_mask,
                                        data["edge_distill_weight"],
                                        response_mask * data["w_tok"].unsqueeze(-1) if dedup else response_mask,
                                        student_lse=tree_student_lse,
                                        teacher_lse=teacher_topk_lse,
                                    )
                                    edge_metrics = {}
                                else:
                                    edge_loss = log_prob.sum()*0
                                    edge_metrics = {"kd/weighted_loss_all_tokens":0., "kd/active_token_rate":0.}
                            elif edge_token_selection == "teacher_topk":
                                edge_loss, edge_metrics = compute_edge_teacher_topk_distillation_loss(
                                    student_topk_logits=student_topk_logits,
                                    teacher_topk_logits=teacher_topk_logits,
                                    response_mask=response_mask,
                                    edge_distill_weight=data["edge_distill_weight"],
                                    loss_agg_mode=loss_agg_mode,
                                )
                            else:
                                teacher_log_prob, _, _ = self._edge_teacher_forward_active(
                                    data, responses, temperature, top_k=None
                                )
                                edge_loss, edge_metrics = compute_edge_distillation_loss(
                                    student_log_prob=log_prob,
                                    teacher_log_prob=teacher_log_prob,
                                    response_mask=response_mask,
                                    edge_distill_weight=data["edge_distill_weight"],
                                    loss_agg_mode=loss_agg_mode,
                                )
                            policy_loss = policy_loss + edge_cfg.get("coef", 1.0) * edge_loss
                            metrics["edge/coef"] = edge_cfg.get("coef", 1.0)
                            if edge_token_selection == "teacher_topk":
                                metrics["edge/teacher_topk_size"] = edge_top_k

                        # Success-only SFT on the complete response (Think/Search/Answer).
                        # Shared-prefix multiplicity is stored in sft_weight by node dedup.
                        # The student always sees the no-T/S version of the same history.
                        sft_numerator = log_prob.sum() * 0.0
                        sft_noexp_dummy_loss = None
                        if sft_enabled:
                            weights = data["sft_weight"].to(log_prob.dtype)
                            active = (weights > 0) & (~data["is_padding_copy"])
                            sft_log_prob = log_prob
                            changed_prompt = (
                                data["sft_student_input_ids"] != data["input_ids"]
                            ).any(-1) | (
                                data["sft_student_attention_mask"] != data["attention_mask"]
                            ).any(-1)
                            changed = (active & changed_prompt).nonzero(as_tuple=True)[0]
                            sft_dummy = changed.numel() == 0 and self._fsdp_sync_optional_forwards
                            if changed.numel() or sft_dummy:
                                forward_ids = changed if not sft_dummy else torch.zeros(
                                    (1,), dtype=torch.long, device=responses.device
                                )
                                noexp_inputs = {
                                    "input_ids": data["sft_student_input_ids"].index_select(0, forward_ids),
                                    "attention_mask": data["sft_student_attention_mask"].index_select(0, forward_ids),
                                    "position_ids": data["sft_student_position_ids"].index_select(0, forward_ids),
                                    "responses": responses.index_select(0, forward_ids),
                                }
                                _, noexp_log_prob, _, _ = self._forward_micro_batch(
                                    noexp_inputs, temperature
                                )
                                if sft_dummy:
                                    sft_noexp_dummy_loss = noexp_log_prob.sum() * 0.0
                                else:
                                    sft_log_prob = log_prob.index_copy(0, changed, noexp_log_prob)
                            sft_mask = response_mask * weights.unsqueeze(-1)
                            sft_numerator = -(sft_log_prob * sft_mask).sum()
                            sft_count = sft_mask.sum()
                            statistic_totals.add("sft/active_tokens", sft_count)
                            statistic_totals.add("sft/active_rows", (active & response_mask.bool().any(-1)).sum())
                            statistic_totals.ratio("sft/loss_token_mean", sft_numerator, sft_count)

                        with torch.no_grad():
                            stat_expanded, stat_positive, stat_negative = statistic_totals.policy(
                                data, response_mask, old_log_prob, log_prob, advantages, pg_loss,
                                entropy, clip_ratio_low, clip_ratio_high, clip_ratio_c)
                            if self.config.use_kl_loss:
                                statistic_totals.ratio("actor/kl_loss",
                                    (kld.detach() * stat_expanded).sum(), stat_expanded.sum())
                            if edge_distillation_enabled:
                                stat_kl = stat_mass = stat_gap = stat_gradient = stat_ids = None
                                if tree_batch and active_tree_kd:
                                    stat_kl, stat_mass, stat_gap = tree_kl, tree_student_mass, tree_gap
                                    stat_gradient, stat_ids = tree_gradient, teacher_topk_ids
                                elif not tree_batch and edge_token_selection == "teacher_topk":
                                    stat_s = torch.log_softmax(student_topk_logits.detach().float(),dim=-1)
                                    stat_t = torch.log_softmax(teacher_topk_logits.detach().float(),dim=-1)
                                    stat_kl = (stat_s.exp() * (stat_s-stat_t)).sum(-1)
                                statistic_totals.kd(data, response_mask, stat_expanded, edge_loss,
                                    tree_batch, stat_kl, stat_mass, stat_gap, stat_gradient, stat_ids,
                                    stat_positive, stat_negative)

                        if diagnostic_component in ("pg", "pg_sft"):
                            policy_loss = pg_loss
                            edge_loss = log_prob.sum() * 0
                        elif diagnostic_component == "kd":
                            policy_loss = edge_cfg.get("coef", 1.0) * edge_loss
                        elif diagnostic_component == "sft":
                            # The SFT branch below provides the only gradient.
                            policy_loss = log_prob.sum() * 0.0
                        expanded_mask = response_mask * data["w_tok"].unsqueeze(-1) if dedup else response_mask
                        expanded_tokens = expanded_mask.sum()
                        if self._explicit_replica_sync:
                            # Accumulate local numerators with one shared global denominator.
                            # The gradient collective below averages ranks, hence world_size.
                            scale = torch.distributed.get_world_size() * expanded_tokens / global_denominators[0]
                            loss = policy_loss * scale
                            if edge_distillation_enabled and diagnostic_component != "sft":
                                # Tree KD and ordinary EDGE use independent
                                # active-token means, with a globally shared
                                # four-replica denominator per minibatch.
                                active_tokens = (response_mask * (data["edge_distill_weight"] > 0).unsqueeze(-1)).sum()
                                kd_scale = torch.distributed.get_world_size() * active_tokens / global_denominators[1]
                                kd_term = edge_cfg.get("coef", 1.0) * edge_loss
                                loss = (policy_loss - kd_term) * scale + kd_term * kd_scale
                        elif self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            scale = data["w_tok"].sum() / data["dedup_mini_weight"][0] if dedup else len(data) / self.config.ppo_mini_batch_size
                            loss = policy_loss * scale
                            if tree_batch and edge_distillation_enabled and diagnostic_component != "sft":
                                active_tokens = (response_mask * (data["edge_distill_weight"] > 0).unsqueeze(-1)).sum()
                                kd_term = edge_cfg.get("coef", 1.0) * edge_loss
                                loss = (policy_loss - kd_term) * scale + kd_term * active_tokens / mini_kd_denominator
                        else:
                            loss = policy_loss / self.gradient_accumulation
                            if tree_batch and edge_distillation_enabled and diagnostic_component != "sft":
                                active_tokens = (response_mask * (data["edge_distill_weight"] > 0).unsqueeze(-1)).sum()
                                kd_term = edge_cfg.get("coef", 1.0) * edge_loss
                                loss = (policy_loss - kd_term) / self.gradient_accumulation + kd_term * active_tokens / mini_kd_denominator
                        # Do not let PG token/row denominators rescale SFT.
                        if sft_enabled and diagnostic_component in (None, "sft"):
                            if self._explicit_replica_sync:
                                loss = loss + sft_coef * (
                                    torch.distributed.get_world_size() * sft_numerator /
                                    global_denominators[2]
                                )
                            else:
                                loss = loss + sft_coef * sft_numerator / mini_sft_denominator
                        # Every FSDP rank must backward through the same optional
                        # student passes, including zero-utility placeholder passes.
                        if kd_noexp_dummy_loss is not None:
                            loss = loss + kd_noexp_dummy_loss
                        if sft_noexp_dummy_loss is not None:
                            loss = loss + sft_noexp_dummy_loss
                        loss.backward()
                        if trace_fsdp_first_step:
                            print(
                                f"T073_FSDP_MICRO rank={torch.distributed.get_rank()} "
                                f"mini={batch_idx} micro={micro_batch_count} phase=backward_done",
                                flush=True,
                            )



                grad_norm = self._optimizer_step(diagnostic_component) if diagnostic_component else self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                print(f"T066_MINI index={batch_idx} pg_loss={float(pg_loss.detach())} grad_norm={float(grad_norm)}", flush=True)
        self.actor_optimizer.zero_grad()
        self._policy_updates_seen = getattr(self, "_policy_updates_seen", 0) + int(diagnostic_component is None)
        if self._explicit_replica_sync and diagnostic_component is None and self._policy_updates_seen <= 2:
            from verl.utils.replica_sync import verify_replica_parameters
            digest = verify_replica_parameters(self.actor_module)
            print(f"T069_RANK_HASH update={self._policy_updates_seen} rank={torch.distributed.get_rank()} sha256={digest}", flush=True)
        statistic_totals.add("perf/edge_teacher_rows", getattr(self, "_edge_teacher_rows", 0))
        statistic_totals.add("perf/edge_student_rows", getattr(self, "_edge_teacher_rows_total", 0))
        self._edge_teacher_rows = self._edge_teacher_rows_total = 0
        metrics.update(statistic_totals.finalize(float(edge_cfg.get("coef", 1.0))))
        if gradient_diagnostic_metrics:
            # The values are globally averaged and identical on all ranks;
            # the worker/verl metrics transport expects list-valued entries.
            for name, value in gradient_diagnostic_metrics.items():
                append_to_dict(metrics, {name: float(value)})
        metrics["actor/pg_loss"] = metrics["actor/pg_loss_expanded"]
        if "actor/entropy_expanded" in metrics:
            metrics["actor/entropy_loss"] = metrics["actor/entropy_expanded"]
        metrics["actor/optimizer_steps_per_train_step"] = [self._optimizer_steps_this_update]
        metrics["actor/effective_minibatch_rows_per_rank"] = self.config.ppo_mini_batch_size
        metrics["actor/micro_batch_count"] = micro_batch_count
        return metrics
