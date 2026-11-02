from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, List

import torch.cuda
import torch.distributed as dist
from torch import nn

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm, compute_algorithm
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
    format_expert_location_layout,
    format_expert_location_layout_diff,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_exec, get_model, get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        get_model: Callable[[], nn.Module],
        get_expert_location_updater: Callable[[], ExpertLocationUpdater],
        get_expert_backup_client: Callable[[], Any],
        get_weight_updater: Callable[[], Any],
    ):
        super().__init__()
        # These collaborators are set on ModelRunner AFTER EPLBManager is
        # constructed (model load, expert_backup_client, weight_updater), so
        # they are read through getters at rebalance time, not captured here.
        self._model_config = model_config
        self._get_model = get_model
        self._get_expert_location_updater = get_expert_location_updater
        self._get_expert_backup_client = get_expert_backup_client
        self._get_weight_updater = get_weight_updater
        self._rebalance_layers_per_chunk = (
            get_exec().moe.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = get_exec().moe.eplb_rebalance_num_iterations
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            get_exec().moe.eplb_rebalance_num_iterations
            >= get_exec().moe.expert_distribution_recorder_buffer_size
        ), (
            "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"
        )

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        next(self._main_generator)

    def reset_generator(self):
        self._main_generator = self._entrypoint()

    def disable_rebalance(self, reason: str):
        self._rebalance_disabled_reason = reason
        self._rebalance_disabled_logged = False
        self.reset_generator()

    def enable_rebalance(self):
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False
        self.reset_generator()

    # can be more complex if needed
    def _entrypoint(self):
        num_iterations = self._first_rebalance_num_iterations()
        while True:
            for _ in range(num_iterations):
                yield

            yield from self.rebalance()
            num_iterations = self._rebalance_num_iterations

    def _first_rebalance_num_iterations(self) -> int:
        # DeepInfer: the startup layout is trivial, so the first rearrange
        # runs at 1/4 of the interval.  It still waits for a full recorder
        # window so the solver never sees an under-filled buffer.
        if not envs.SGLANG_EPLB_EARLY_FIRST_REBALANCE.get():
            return self._rebalance_num_iterations
        return max(
            self._rebalance_num_iterations // 4,
            get_exec().moe.expert_distribution_recorder_buffer_size,
            1,
        )

    def rebalance(self):
        if self._rebalance_disabled_reason is not None:
            if not self._rebalance_disabled_logged:
                logger.debug(
                    "[EPLBManager] rebalance disabled: %s",
                    self._rebalance_disabled_reason,
                )
                self._rebalance_disabled_logged = True
            return

        elastic_state = ElasticEPStateManager.instance()
        is_post_scale_rebalance = elastic_state is not None and elastic_state.has_scaled
        # A failed later scale leaves the previously committed world serving.
        if is_post_scale_rebalance and (
            elastic_state.pending_ep_size is not None
            or elastic_state.scale_phase not in ("serving_expanded", "failed")
        ):
            return

        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object",
            aggregate_steps_for_rebalance=self._aggregate_steps_for_rebalance(),
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = self._compute_expert_location_metadata(
            logical_count,
            broadcast_over_world=is_post_scale_rebalance,
        )
        self._log_predicted_rank_balancedness(logical_count, expert_location_metadata)

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        all_update_layer_ids = [
            layer_id for chunk in update_layer_ids_chunks for layer_id in chunk
        ]
        self._log_rebalance_layout_before_update(
            expert_location_metadata,
            update_layer_ids=all_update_layer_ids,
        )
        for chunk_layer_ids in update_layer_ids_chunks:
            if len(update_layer_ids_chunks) > 1:
                yield
            update_expert_location_with_recovery(
                expert_location_updater=self._get_expert_location_updater(),
                model=self._get_model(),
                new_expert_location_metadata=expert_location_metadata,
                update_layer_ids=chunk_layer_ids,
                tp_rank=(
                    self._elastic_global_rank()
                    if is_post_scale_rebalance
                    else get_parallel().tp_rank
                ),
                use_flat_topology=is_post_scale_rebalance,
                expert_backup_client=self._get_expert_backup_client(),
                update_weights_from_disk_callable=self._get_weight_updater().update_weights_from_disk,
                ep_dispatch_algorithm=get_exec().moe.ep_dispatch_algorithm,
                init_lplb_solvers_callable=lambda: init_lplb_solvers(
                    model_config=self._model_config
                ),
            )
            if is_post_scale_rebalance:
                # P2P waits only synchronize participating peers. Ranks without
                # moves must also install this chunk before NIXL resumes.
                dist.barrier()

        self._log_rebalance_layout_after_update(update_layer_ids=all_update_layer_ids)

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _aggregate_steps_for_rebalance(self) -> bool:
        """Whether the selected solver consumes only a time aggregate.

        Vector EPLB intentionally needs the per-step tensor.  The scalar
        DeepSeek and elasticity-aware solvers only call ``sum(dim=0)``; let
        the recorder perform that reduction before the expensive logical-map
        expansion and distributed all-reduce.
        """
        metadata = ModelConfigForExpertLocation.from_model_config(self._model_config)
        num_groups = metadata.num_groups if metadata is not None else None
        algorithm = compute_algorithm(
            raw_algorithm=get_exec().moe.eplb_algorithm,
            num_groups=num_groups,
            num_nodes=get_parallel().nnodes,
        )
        return algorithm in {
            EplbAlgorithm.deepseek,
            EplbAlgorithm.deepseek_hierarchical,
            EplbAlgorithm.elasticity_aware,
            EplbAlgorithm.elasticity_aware_hierarchical,
        }

    def _compute_expert_location_metadata(
        self, logical_count, *, broadcast_over_world: bool
    ) -> ExpertLocationMetadata:
        if not broadcast_over_world:
            return ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
            )

        current_metadata = get_global_expert_location_metadata()
        assert current_metadata is not None
        # One owner prevents process-local launch topology from influencing
        # the mapping chosen for the expanded world.
        if dist.get_rank() == 0:
            computed_metadata = ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
                # Arbitrary append topologies may not preserve node divisibility.
                use_flat_topology=True,
            )
            physical_to_logical_map = (
                computed_metadata.physical_to_logical_map.contiguous()
            )
        else:
            physical_to_logical_map = torch.empty_like(
                current_metadata.physical_to_logical_map
            )

        dist.broadcast(physical_to_logical_map, src=0)
        return ExpertLocationMetadata.init_by_mapping(
            self._model_config,
            physical_to_logical_map,
            moe_ep_rank=self._elastic_global_rank(),
        )

    def _elastic_global_rank(self) -> int:
        return get_parallel().tp_rank + get_parallel().ep_join_rank_offset

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > get_exec().moe.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {get_exec().moe.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._get_model().routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _should_log_expert_location_metadata(self) -> bool:
        return (
            get_parallel().tp_rank == 0
            and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get()
        )

    def _log_rebalance_layout_before_update(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if not self._should_log_expert_location_metadata():
            return

        old_expert_location_metadata = get_global_expert_location_metadata()
        logger.info(
            "[EPLBManager] rebalance layout before:\n%s",
            format_expert_location_layout(
                old_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout target:\n%s",
            format_expert_location_layout(
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout diff:\n%s",
            format_expert_location_layout_diff(
                old_expert_location_metadata,
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )

    def _log_predicted_rank_balancedness(
        self, logical_count, new_expert_location_metadata: ExpertLocationMetadata
    ):
        # Rank 0 only and on CPU: a [steps, layers, physical] temporary must
        # not compete with the weight transfer for device memory.
        if get_parallel().tp_rank != 0:
            return
        old_expert_location_metadata = get_global_expert_location_metadata()
        if old_expert_location_metadata is None or new_expert_location_metadata is None:
            return
        try:
            logical_count = logical_count.cpu()
            old = predicted_rank_balancedness(
                logical_count,
                old_expert_location_metadata.physical_to_logical_map,
                old_expert_location_metadata.ep_size,
            )
            new = predicted_rank_balancedness(
                logical_count,
                new_expert_location_metadata.physical_to_logical_map,
                new_expert_location_metadata.ep_size,
            )
        except Exception as e:  # logging only; never block a rebalance
            logger.warning("[EPLBManager] predicted balancedness skipped: %s", e)
            return
        logger.info(
            "[EPLBManager] predicted rank balancedness (mean/max, per-step, "
            "avg over layers) on the recorded window: %.4f -> %.4f",
            old,
            new,
        )

    def _log_rebalance_layout_after_update(self, update_layer_ids: List[int]):
        if not self._should_log_expert_location_metadata():
            return

        logger.info(
            "[EPLBManager] rebalance layout after:\n%s",
            format_expert_location_layout(
                get_global_expert_location_metadata(),
                layer_ids=update_layer_ids,
            ),
        )


def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    tp_rank: int,
    use_flat_topology: bool = False,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=get_parallel().nnodes,
        rank=tp_rank,
        use_flat_topology=use_flat_topology,
    )

    if len(p2p_missing_logical_experts) > 0:
        # Load the missing expert weights from disk
        if callable(getattr(model, "generate_weight_name_filter", None)):
            # Filter and load only missing expert weights
            weight_name_filter = model.generate_weight_name_filter(
                p2p_missing_logical_experts
            )
        else:
            # Do a full reload from disk/DRAM
            logger.info(
                "[Elastic EP] Model does not implement generate_weight_name_filter. "
                "Performing full weight reload."
            )
            weight_name_filter = None

        if expert_backup_client is not None and expert_backup_client.use_backup:
            # Load the missing weights from the DRAM backup
            expert_backup_client.update_weights(weight_name_filter)
        else:
            # Load the missing weights from disk
            update_weights_from_disk_callable(
                get_model().model_path,
                get_model().load_format,
                weight_name_filter=weight_name_filter,
            )

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm == "lp":
        init_lplb_solvers_callable()


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]


def predicted_rank_balancedness(
    logical_count: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    ep_size: int,
) -> float:
    """Mean/max rank load a layout would have had on the recorded window.

    Each logical expert's tokens are split evenly over its replicas (the
    model DeepInfer's placement optimizes).  Computed per step, then averaged
    over steps and layers; 1.0 is perfectly balanced.
    """
    counts = logical_count.float()
    if counts.dim() == 2:
        counts = counts.unsqueeze(0)
    num_steps, num_layers, num_logical = counts.shape
    p2l = physical_to_logical_map.to(device=counts.device, dtype=torch.int64)
    p2l = p2l[:num_layers] % num_logical
    replicas = torch.zeros(
        (num_layers, num_logical), dtype=torch.float32, device=counts.device
    )
    replicas.scatter_add_(1, p2l, torch.ones_like(p2l, dtype=torch.float32))
    per_replica = counts / replicas.clamp_min(1)  # [steps, layers, logical]
    physical_load = per_replica.gather(
        2, p2l.unsqueeze(0).expand(num_steps, -1, -1)
    )  # [steps, layers, physical]
    rank_load = physical_load.view(num_steps, num_layers, ep_size, -1).sum(-1)
    mean = rank_load.mean(-1)
    peak = rank_load.amax(-1)
    active = peak > 0
    if not bool(active.any()):
        return 1.0
    return float((mean[active] / peak[active]).mean())
