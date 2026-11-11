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


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
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

                if token_selection_top_k is not None:
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
        log_prob_full = None if top_k is not None else torch.zeros((bsz, resp_len), dtype=torch.float32, device=device)
        logits_full = ids_full = None
        if top_k is not None:
            ids_full = torch.zeros((bsz, resp_len, k), dtype=torch.long, device=device)
        if n_active == 0:
            if top_k is not None:
                logits_full = torch.zeros((bsz, resp_len, k), dtype=torch.float32, device=device)
            return log_prob_full, logits_full, ids_full

        teacher_batch = {
            "responses": responses.index_select(0, active),
            "input_ids": data["edge_teacher_input_ids"].index_select(0, active),
            "attention_mask": data["edge_teacher_attention_mask"].index_select(0, active),
            "position_ids": data["edge_teacher_position_ids"].index_select(0, active),
        }
        if top_k is not None:
            _, _, sel_logits, sel_ids = self._forward_micro_batch(
                micro_batch=teacher_batch, temperature=temperature, calculate_entropy=False,
                calculate_log_probs=False, token_selection_top_k=top_k,
            )
            k = sel_logits.size(-1)
            logits_full = torch.zeros((bsz, resp_len, k), dtype=sel_logits.dtype, device=device)
            logits_full.index_copy_(0, active, sel_logits)
            ids_full = torch.zeros((bsz, resp_len, k), dtype=sel_ids.dtype, device=device)
            ids_full.index_copy_(0, active, sel_ids)
        else:
            _, lp, _, _ = self._forward_micro_batch(
                micro_batch=teacher_batch, temperature=temperature, calculate_entropy=False,
            )
            log_prob_full = torch.zeros((bsz, resp_len), dtype=lp.dtype, device=device)
            log_prob_full.index_copy_(0, active, lp)
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
        diagnostic_component = data.meta_info.get("gradient_diagnostic_component")
        if (self._explicit_replica_sync and diagnostic_component is None
                and int(data.meta_info.get("global_step", -1)) in (1, 50, 100)
                and data.meta_info.get("gradient_diagnostic_dir")
                and self.config.get("edge_distillation", {}).get("enable", False)):
            from verl.utils.gradient_diagnostics import measure_actor_gradients
            measure_actor_gradients(self, data)
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
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        from verl.utils.actor_metric_totals import ActorMetricTotals
        statistic_totals = ActorMetricTotals(get_torch_device().current_device())
        if edge_distillation_requested:
            append_to_dict(metrics, {"edge/skipped_missing_inputs": float(bool(edge_missing_keys))})
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                if self._explicit_replica_sync:
                    from verl.utils.replica_sync import summed_denominators
                    mb = mini_batch.batch if isinstance(mini_batch, DataProto) else mini_batch
                    mask = mb["loss_mask"][:, -mb["responses"].shape[1]:] if multi_turn else mb["attention_mask"][:, -mb["responses"].shape[1]:]
                    kd_weights = mb["edge_distill_weight"] if edge_distillation_enabled else torch.zeros(len(mask), device=mask.device)
                    global_denominators = summed_denominators(mask, torch.ones(len(mask), device=mask.device), kd_weights)

                from contextlib import nullcontext
                accumulation_context = self.actor_module.no_sync() if self._explicit_replica_sync else nullcontext()
                with accumulation_context:
                    for data in micro_batches:
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
                        if edge_distillation_enabled and edge_token_selection == "teacher_topk":
                            # Same actor module under no_grad, evaluated before the student,
                            # but only on rows the gate activated (weight > 0): every other row
                            # is masked out of the distillation loss, so its teacher pass is waste.
                            _, teacher_topk_logits, teacher_topk_ids = self._edge_teacher_forward_active(
                                data, responses, temperature, top_k=edge_top_k
                            )

                        # all return: (bsz, response_length)
                        calculate_entropy = True
                        entropy, log_prob, student_topk_logits, _ = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            token_selection_top_k=edge_top_k if teacher_topk_ids is not None else None,
                            token_selection_ids=teacher_topk_ids,
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
                            if edge_token_selection == "teacher_topk":
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

                        with torch.no_grad():
                            stat_expanded, stat_positive, stat_negative = statistic_totals.policy(
                                data, response_mask, old_log_prob, log_prob, advantages, pg_loss,
                                entropy, clip_ratio_low, clip_ratio_high, clip_ratio_c)
                            if self.config.use_kl_loss:
                                statistic_totals.ratio("actor/kl_loss",
                                    (kld.detach() * stat_expanded).sum(), stat_expanded.sum())
                            if edge_distillation_enabled:
                                stat_kl = stat_mass = stat_gap = stat_gradient = stat_ids = None
                                if False and active_tree_kd:
                                    stat_kl, stat_mass, stat_gap = tree_kl, tree_student_mass, tree_gap
                                    stat_gradient, stat_ids = tree_gradient, teacher_topk_ids
                                elif not False and edge_token_selection == "teacher_topk":
                                    stat_s = torch.log_softmax(student_topk_logits.detach().float(),dim=-1)
                                    stat_t = torch.log_softmax(teacher_topk_logits.detach().float(),dim=-1)
                                    stat_kl = (stat_s.exp() * (stat_s-stat_t)).sum(-1)
                                statistic_totals.kd(data, response_mask, stat_expanded, edge_loss,
                                    False, stat_kl, stat_mass, stat_gap, stat_gradient, stat_ids,
                                    stat_positive, stat_negative)

                        if diagnostic_component == "pg":
                            policy_loss = pg_loss
                            edge_loss = log_prob.sum() * 0
                        elif diagnostic_component == "kd":
                            policy_loss = edge_cfg.get("coef", 1.0) * edge_loss
                        if self._explicit_replica_sync:
                            tokens = response_mask.sum()
                            scale = torch.distributed.get_world_size() * tokens / global_denominators[0]
                            loss = policy_loss * scale
                            if edge_distillation_enabled:
                                active_tokens = (response_mask * (data["edge_distill_weight"] > 0).unsqueeze(-1)).sum()
                                kd_scale = torch.distributed.get_world_size() * active_tokens / global_denominators[1]
                                loss = (policy_loss - edge_cfg.get("coef", 1.0) * edge_loss) * scale + edge_cfg.get("coef", 1.0) * edge_loss * kd_scale
                        elif self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss.backward()



                grad_norm = self._optimizer_step(diagnostic_component) if diagnostic_component else self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
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
        metrics["actor/pg_loss"] = metrics["actor/pg_loss_expanded"]
        if "actor/entropy_expanded" in metrics:
            metrics["actor/entropy_loss"] = metrics["actor/entropy_expanded"]
        metrics["actor/optimizer_steps_per_train_step"] = [self._optimizer_steps_this_update]
        return metrics
