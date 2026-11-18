"""Token/row totals for Actor monitoring; one reduction after local accumulation."""
import torch
import torch.distributed as dist

ACTOR_RATIOS = (
    "actor/entropy_expanded", "actor/pg_loss_expanded", "actor/response_length_expanded", "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower", "actor/ppo_kl", "actor/kl_loss",
    "kd/weighted_loss_all_tokens", "kd/weighted_loss_active_tokens", "kd/active_token_rate",
    "kd/effective_weight_sum_over_N", "edge/distill_loss",
    "edge/actor_active_row_fraction", "edge/teacher_topk_kl", "kd/student_mass", "sft/loss_token_mean",
)
GROUP_RATIOS = ("topk_kl", "student_mass", "logprob_gap", "logprob_gap_abs",
                "mean_weight", "positive_adv_conflict_proxy", "negative_adv_conflict_proxy")
COUNTS = ("actor/expanded_tokens", "kd/active_tokens", "kd/effective_weight_sum",
          "kd/expanded_rows", "kd/active_rows", "perf/edge_teacher_rows", "perf/edge_student_rows",
    "sft/active_tokens", "sft/active_rows")
for_group = tuple(f"kd/{group}/{name}" for group in ("no_T", "with_T") for name in GROUP_RATIOS)
group_counts = tuple(f"kd/{group}/{name}" for group in ("no_T", "with_T") for name in ("tokens", "rows"))
RATIOS = ACTOR_RATIOS + for_group
COUNT_KEYS = COUNTS + group_counts

class ActorMetricTotals:
    def __init__(self, device):
        self.keys = tuple(x + suffix for x in RATIOS for suffix in ("/sum", "/count")) + COUNT_KEYS
        self.indices = {key:i for i,key in enumerate(self.keys)}
        self.values = torch.zeros(len(self.keys), dtype=torch.float64, device=device)

    def add(self, key, value):
        self.values[self.indices[key]] += torch.as_tensor(value, device=self.values.device).detach().double()

    def ratio(self, key, numerator, denominator):
        self.add(key + "/sum", numerator)
        self.add(key + "/count", denominator)

    @torch.no_grad()
    def policy(self, data, mask, old_log_prob, log_prob, advantages, pg_loss,
               entropy, clip_low, clip_high, dual_clip):
        repeat = data.get("w_tok", torch.ones(len(mask),device=mask.device))
        expanded = mask * repeat[:, None]
        total = expanded.sum()
        self.add("actor/expanded_tokens", total)
        self.add("kd/expanded_rows", (repeat * mask.any(-1)).sum())
        self.ratio("actor/response_length_expanded",total,(repeat * mask.any(-1)).sum())
        self.ratio("actor/pg_loss_expanded", pg_loss.detach() * total, total)
        if entropy is not None:
            self.ratio("actor/entropy_expanded", (entropy.detach() * expanded).sum(), total)
        delta = log_prob.detach() - old_log_prob
        ratio = delta.exp()
        if "w_tok" in data:
            # An aggregated node can contain leaves of BOTH signs.
            positive = data["positive_adv_count"][:,None].expand_as(mask)
            negative = data["negative_adv_count"][:,None].expand_as(mask)
        else:
            positive = (advantages > 0).to(mask.dtype)
            negative = (advantages < 0).to(mask.dtype)
        clipped = ((ratio > 1 + clip_high) * positive + (ratio < 1 - clip_low) * negative) * mask
        lower = (ratio > dual_clip) * negative * mask
        self.ratio("actor/pg_clipfrac", clipped.sum(), total)
        self.ratio("actor/pg_clipfrac_lower", lower.sum(), total)
        self.ratio("actor/ppo_kl", (-delta * expanded).sum(), total)
        return expanded, positive, negative

    @torch.no_grad()
    def kd(self, data, mask, expanded, edge_loss, tree_batch, kl=None, mass=None,
           gap=None, gradient=None, topk_ids=None, positive=None, negative=None):
        weights = data["edge_distill_weight"]
        active = mask * (weights > 0)[:,None]
        tokens = active.sum()
        weight_sum = (active * weights[:,None]).sum()
        rows = ((weights > 0) & mask.any(-1)).sum()
        self.add("kd/active_tokens",tokens)
        self.add("kd/effective_weight_sum",weight_sum)
        self.add("kd/active_rows",rows)
        self.ratio("kd/active_token_rate",tokens,expanded.sum())
        self.ratio("kd/effective_weight_sum_over_N",weight_sum,expanded.sum())
        # The scalar KL loss is an active-token mean in both modes.
        numerator = edge_loss.detach() * tokens
        self.ratio("kd/weighted_loss_all_tokens",numerator,expanded.sum())
        self.ratio("kd/weighted_loss_active_tokens",numerator,tokens)
        self.ratio("edge/distill_loss",numerator,tokens)
        self.ratio("edge/actor_active_row_fraction",rows,
                   (data.get("w_tok", torch.ones(len(mask),device=mask.device)) * mask.any(-1)).sum())
        if kl is None:
            return
        self.ratio("edge/teacher_topk_kl",(kl.detach() * active).sum(),tokens)
        if mass is not None:
            self.ratio("kd/student_mass",(mass.detach() * active).sum(),tokens)
        if not tree_batch:
            return
        direction = -(gradient.detach() * (topk_ids == data["responses"][:,:,None])).sum(-1)
        for group, included in (("no_T",~data["kd_path_has_T"]),("with_T",data["kd_path_has_T"])):
            selected = active * included[:,None]
            prefix = "kd/" + group + "/"
            self.add(prefix+"tokens",selected.sum())
            self.add(prefix+"rows",((weights>0) & included & mask.any(-1)).sum())
            for name,value in (("topk_kl",kl),("student_mass",mass),
                               ("logprob_gap",gap),("logprob_gap_abs",gap.abs()),
                               ("mean_weight",weights[:,None])):
                self.ratio(prefix+name,(value.detach() * selected).sum(),selected.sum())
            # Count expanded leaves of each sign on the same canonical KD node.
            for label,count,conflict in (("positive_adv",positive,direction<0),
                                         ("negative_adv",negative,direction>0)):
                eligible = selected * count * (direction != 0)
                self.ratio(prefix+label+"_conflict_proxy",
                           (eligible * conflict).sum(),eligible.sum())

    def finalize(self, coefficient):
        if dist.is_initialized():
            dist.all_reduce(self.values,op=dist.ReduceOp.SUM)
        values = self.values.cpu().tolist()
        raw = dict(zip(self.keys, values))
        metrics = {key:raw[key] for key in COUNT_KEYS}
        for key in RATIOS:
            count = raw[key+"/count"]
            # Zero observed tokens is not evidence of a zero conditional KL.
            if count > 0:
                metrics[key] = raw[key+"/sum"]/count
        for name in ("active_token_rate","effective_weight_sum_over_N","weighted_loss_all_tokens","weighted_loss_active_tokens"):
            metrics.setdefault("kd/"+name,0.)
        metrics.setdefault("edge/distill_loss",0.)
        kd = metrics["kd/weighted_loss_active_tokens"]
        kd_all = metrics["kd/weighted_loss_all_tokens"]
        pg = metrics.get("actor/pg_loss_expanded",0.)
        # Loss ratio is not a gradient norm ratio.
        metrics["kd/to_pg_loss_abs_ratio"] = abs(coefficient*kd)/(abs(pg)+1e-6)
        metrics["kd/weighted_loss_active_denominator"] = coefficient*kd
        metrics["kd/weighted_loss_pg_denominator"] = coefficient*kd_all
        if raw["perf/edge_student_rows"]>0:
            metrics["perf/edge_teacher_row_fraction"]=raw["perf/edge_teacher_rows"]/raw["perf/edge_student_rows"]
        metrics["actor/statistics_global"]=1.
        return metrics
