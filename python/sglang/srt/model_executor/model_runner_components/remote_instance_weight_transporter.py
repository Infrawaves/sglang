from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import msgspec
import torch

from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    get_ib_devices_for_gpu,
)
from sglang.srt.environ import envs
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    INTRA_NVLINK_PROTOCOL,
    MNNVL_PROTOCOL,
    NVLINK_PROTOCOLS,
    RemoteInstanceWeightLoaderBackend,
    get_remote_instance_transfer_engine_info_per_rank,
    register_memory_region,
)
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    remote_instance_transfer_engine_enabled,
)
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvlink_fabric_utils import (
    NvlinkFabricIdentity,
    get_nvlink_fabric_identity,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class _TransportEndpoint:
    # One engine per transport: initialize() bakes it in.
    protocol: str
    engine: Any
    session_id: str
    fabric_identity: Optional[NvlinkFabricIdentity] = None
    weight_info: Optional[dict[str, tuple[int, int, int]]] = None

    def to_payload(self) -> dict:
        return {
            "session_id": self.session_id,
            "weights_info_dict": self.weight_info,
            "protocol": self.protocol,
            "fabric_identity": (
                msgspec.structs.asdict(self.fabric_identity)
                if self.fabric_identity is not None
                else None
            ),
        }


@dataclass(slots=True, kw_only=True)
class RemoteInstanceWeightTransporter:
    get_model: Callable[[], torch.nn.Module]
    tp_rank: int
    gpu_id: int
    is_draft_worker: bool = False
    endpoints: dict[str, _TransportEndpoint] = field(default_factory=dict)
    fabric_identity: Optional[NvlinkFabricIdentity] = None
    _nixl_manager: Optional[Any] = None

    @property
    def engine(self) -> Optional[Any]:
        endpoint = self._primary_endpoint()
        return endpoint.engine if endpoint is not None else None

    @property
    def session_id(self) -> str:
        endpoint = self._primary_endpoint()
        return endpoint.session_id if endpoint is not None else ""

    @property
    def protocol(self) -> Optional[str]:
        endpoint = self._primary_endpoint()
        return endpoint.protocol if endpoint is not None else None

    @property
    def weight_info(self) -> Optional[dict[str, tuple[int, int, int]]]:
        endpoint = self._primary_endpoint()
        return endpoint.weight_info if endpoint is not None else None

    @weight_info.setter
    def weight_info(self, value) -> None:
        endpoint = self._primary_endpoint()
        if endpoint is not None:
            endpoint.weight_info = value

    def _primary_endpoint(self) -> Optional[_TransportEndpoint]:
        # A seed with several endpoints reports the NIC one, the endpoint every
        # peer can reach; a client has built exactly one.
        if not self.endpoints:
            return None
        for protocol, endpoint in self.endpoints.items():
            if protocol not in NVLINK_PROTOCOLS:
                return endpoint
        return next(iter(self.endpoints.values()))

    @property
    def model(self) -> torch.nn.Module:
        return self.get_model()

    def init_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake-transfer-engine"
            )
            return
        # Assigned by the driver at boot, so it is fixed for the process.
        self.fabric_identity = get_nvlink_fabric_identity(self.gpu_id)
        for protocol in self._protocols_to_serve():
            endpoint = self._build_endpoint(TransferEngine, protocol)
            if endpoint is not None:
                self.endpoints[endpoint.protocol] = endpoint

    def _protocols_to_serve(self) -> tuple[str, ...]:
        # A client needs the one transport that reaches its seed; a seed cannot
        # know which clients will arrive, so it serves every one it can.
        configured = envs.SGLANG_REMOTE_INSTANCE_PROTOCOL.get()
        if configured:
            return (configured,)
        seed_protocol = self._probe_seed_protocol()
        if seed_protocol is not None:
            return (seed_protocol,)
        if self._is_seed():
            return (*self._servable_nvlink_protocols(), envs.MOONCAKE_PROTOCOL.get())
        return (envs.MOONCAKE_PROTOCOL.get(),)

    def _is_seed(self) -> bool:
        # A seed has no seed of its own to point at.
        return get_model().remote_instance_weight_loader_seed_instance_ip is None

    def _servable_nvlink_protocols(self) -> tuple[str, ...]:
        # nvlink_intra needs no fabric -- it is CUDA IPC, and only the client
        # knows whether it landed on this host. MNNVL needs a clique to publish.
        protocols = [INTRA_NVLINK_PROTOCOL]
        if self.fabric_identity is not None:
            protocols.append(MNNVL_PROTOCOL)
        return tuple(protocols)

    @staticmethod
    @contextmanager
    def _mooncake_transport_env(protocol: str):
        """Make mooncake install ``protocol`` for the engine built inside.

        initialize() takes a protocol but picks the transport from these two
        variables, read at each call. Setting them per call is what lets one
        process hold engines on different transports.
        """
        # Mutually exclusive by mooncake's own rule.
        wanted = (
            (envs.MC_INTRANODE_NVLINK, protocol == INTRA_NVLINK_PROTOCOL),
            (envs.MC_FORCE_MNNVL, protocol == MNNVL_PROTOCOL),
        )
        saved = [(env, env.is_set(), env.get()) for env, _ in wanted]
        try:
            for env, enable in wanted:
                # Cleared, not set false: mooncake uses getenv, to which the
                # string "False" is as true as "1".
                env.set(True) if enable else env.clear()
            yield
        finally:
            for env, was_set, value in saved:
                env.set(value) if was_set else env.clear()

    def _build_endpoint(
        self, transfer_engine_cls, protocol: str
    ) -> Optional[_TransportEndpoint]:
        engine = transfer_engine_cls()
        local_ip = get_local_ip_auto()
        # Resolved per rank, not passed through: these accept a
        # {gpu_id: devices} mapping, which mooncake cannot parse -- it would
        # fall back to taking every NIC on every rank.
        configured_device = (
            envs.SGLANG_REMOTE_INSTANCE_IB_DEVICE.get() or envs.MOONCAKE_DEVICE.get()
        )
        try:
            ib_device = get_ib_devices_for_gpu(configured_device, self.gpu_id)
        except ValueError:
            logger.warning(
                "No IB device configured for GPU %s; letting mooncake select "
                "the HCA for remote-instance transfers.",
                self.gpu_id,
            )
            ib_device = None
        try:
            with self._mooncake_transport_env(protocol):
                engine.initialize(local_ip, "P2PHANDSHAKE", protocol, ib_device or "")
        except Exception:
            # A build without this transport compiled in refuses it here.
            logger.warning(
                "Cannot initialize a mooncake TransferEngine on protocol=%s for "
                "GPU %s; that transport will not be offered.",
                protocol,
                self.gpu_id,
                exc_info=True,
            )
            return None
        logger.info(
            "Remote-instance TransferEngine on GPU %s using protocol=%s, "
            "ib_device=%r (empty means mooncake selects), fabric=%s.",
            self.gpu_id,
            protocol,
            ib_device or "",
            self.fabric_identity or "none",
        )
        return _TransportEndpoint(
            protocol=protocol,
            engine=engine,
            session_id=NetworkAddress(
                local_ip, engine.get_rpc_port()
            ).to_host_port_str(),
            fabric_identity=(
                self.fabric_identity if protocol in NVLINK_PROTOCOLS else None
            ),
        )

    def _probe_seed_protocol(self) -> Optional[str]:
        # Runs before any engine exists, so it asks over HTTP, not the fabric.
        seed_ip = get_model().remote_instance_weight_loader_seed_instance_ip
        seed_port = get_model().remote_instance_weight_loader_seed_instance_service_port
        if seed_ip is None or seed_port is None:
            return None

        seed_info = get_remote_instance_transfer_engine_info_per_rank(
            f"http://{seed_ip}:{seed_port}", self.tp_rank
        )
        if seed_info is None or seed_info.protocol is None:
            return None

        offered = seed_info.offered_protocols()
        local_host = get_local_ip_auto()
        # NVLINK_PROTOCOLS is fastest first, so the first reachable one wins.
        for protocol in NVLINK_PROTOCOLS:
            if protocol not in offered:
                continue
            if (
                seed_info.endpoint_for(
                    protocol=protocol,
                    fabric=self.fabric_identity,
                    local_host=local_host,
                )
                is not None
            ):
                logger.info(
                    "GPU %s reaches its seed over %s; loading weights over "
                    "NVLink instead of the NIC.",
                    self.gpu_id,
                    protocol,
                )
                return protocol

        for protocol in offered:
            if protocol not in NVLINK_PROTOCOLS:
                logger.info(
                    "GPU %s cannot reach its seed over NVLink (seed offers %s, "
                    "this host is %s on fabric %s); loading weights over %s.",
                    self.gpu_id,
                    offered,
                    local_host,
                    self.fabric_identity or "none",
                    protocol,
                )
                return protocol

        logger.warning(
            "The seed serves weights only over %s, which GPU %s cannot reach. "
            "Expect a disk load.",
            offered,
            self.gpu_id,
        )
        return None

    def maybe_register_and_publish_weight_info(self) -> None:
        if (
            remote_instance_transfer_engine_enabled()
            # The draft shares the target's tp_rank, the bootstrap server's
            # key, so publishing here would replace the target's manifest.
            and not self.is_draft_worker
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and get_model().remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.engine is not None
            and self.weight_info is None
        ):
            # Off the startup path: ibv_reg_mr runs once per NIC per block,
            # ~90s over 16 NICs, and none of it is needed for this rank to
            # serve. A client arriving before it lands disk-loads.
            threading.Thread(
                target=self._register_and_publish_weight_info,
                name=f"weight-mr-register-tp{self.tp_rank}",
                daemon=True,
            ).start()

    def _register_and_publish_weight_info(self) -> None:
        try:
            # A fresh thread carries no current CUDA context, which the NVLink
            # transports need: they register through cuMemGetAddressRange and
            # would fail with CUDA_ERROR_INVALID_CONTEXT (201).
            torch.cuda.set_device(self.gpu_id)
            for endpoint in self.endpoints.values():
                # Never deregistered: the seed serves handles out of this for the
                # life of the process.
                endpoint.weight_info, _ = register_memory_region(
                    self.model, endpoint.engine
                )
            self._register_to_engine_info_bootstrap()
        except Exception:
            logger.exception(
                "Failed to register weight memory for tp_rank=%s; this instance "
                "will not be usable as a remote-instance seed.",
                self.tp_rank,
            )

    def _register_to_engine_info_bootstrap(self: RemoteInstanceWeightTransporter):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if get_parallel().dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(get_parallel().dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = get_model().engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        # The NIC endpoint is primary, so a client too old to read "alternates"
        # still lands on one it can reach.
        primary = self._primary_endpoint()
        if primary is None:
            logger.error("No transfer engine to publish for tp_rank=%s.", self.tp_rank)
            return
        info = primary.to_payload()
        info["alternates"] = [
            endpoint.to_payload()
            for protocol, endpoint in self.endpoints.items()
            if protocol != primary.protocol
        ]
        payload = {"tp_rank": self.tp_rank, "transfer_engine_info": info}

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )
