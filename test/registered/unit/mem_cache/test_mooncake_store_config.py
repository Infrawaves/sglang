"""Unit tests for MooncakeStoreConfig local_hostname and protocol resolution.

Regression for sgl-project/sglang#23457: runtime attach must resolve
per-node local_hostname from process environment instead of a shared default.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStoreConfig,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestMooncakeStoreConfigLocalHostname(CustomTestCase):
    _BASE_EXTRA_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_EXTRA_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_mooncake_env_when_omitted(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.2"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.local_hostname, "10.0.0.2")

    def test_load_from_extra_config_prefers_env_over_broadcast_override(self):
        with temp_set_env(MOONCAKE_LOCAL_HOSTNAME="10.0.0.4"):
            cfg = self._load_from_extra_config({"local_hostname": "10.0.0.1"})
        self.assertEqual(cfg.local_hostname, "10.0.0.4")


class TestMooncakeStoreConfigProtocol(CustomTestCase):
    _BASE_CONFIG = {"master_server_address": "127.0.0.1:50051"}

    def _load_from_extra_config(self, extra_config=None):
        config = {**self._BASE_CONFIG, **(extra_config or {})}
        return MooncakeStoreConfig.load_from_extra_config(config)

    def test_load_from_extra_config_uses_protocol_env_when_omitted(self):
        with temp_set_env(MOONCAKE_PROTOCOL="efa"):
            cfg = self._load_from_extra_config()
        self.assertEqual(cfg.protocol, "efa")

    def test_load_from_extra_config_prefers_explicit_protocol_over_env(self):
        with temp_set_env(MOONCAKE_PROTOCOL="efa"):
            cfg = self._load_from_extra_config({"protocol": "rdma"})
        self.assertEqual(cfg.protocol, "rdma")


class TestMooncakeConfigSourceSelection(CustomTestCase):
    def test_standalone_follows_the_selected_source(self):
        """Extra config wins only when it names a server address, else the file,
        else env; no field merge."""
        # (extra_config, file_config, env standalone_storage, expected standalone)
        cases = (
            (
                {"master_server_address": "extra", "standalone_storage": False},
                {"standalone_storage": True},
                True,
                False,
            ),
            (
                {"master_server_address": "extra"},
                {"standalone_storage": True},
                True,
                False,
            ),
            (
                {"client_server_address": "extra", "standalone_storage": True},
                {"standalone_storage": False},
                False,
                True,
            ),
            ({"standalone_storage": False}, {"standalone_storage": True}, False, True),
            ({"standalone_storage": True}, None, False, False),
            (
                {"master_server_address": None, "standalone_storage": False},
                {"standalone_storage": True},
                False,
                True,
            ),
            (None, {}, True, False),
            (None, None, True, True),
            (None, None, False, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mooncake.json")
            for extra, file_config, env_standalone, expected in cases:
                with (
                    self.subTest(extra=extra, file=file_config, env=env_standalone),
                    envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(path),
                    envs.MOONCAKE_MASTER.override("environment"),
                    envs.MOONCAKE_STANDALONE_STORAGE.override(env_standalone),
                ):
                    if file_config is None:
                        envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.clear()
                    else:
                        with open(path, "w") as handle:
                            json.dump(
                                {"master_server_address": "file"} | file_config, handle
                            )
                    self.assertEqual(
                        MooncakeStoreConfig.load(extra).standalone_storage, expected
                    )

    def test_invalid_selected_file_does_not_fall_back(self):
        """An unreadable configured file must raise instead of falling back to env."""
        with (
            tempfile.TemporaryDirectory() as directory,
            envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(
                os.path.join(directory, "missing.json")
            ),
            envs.MOONCAKE_MASTER.override("environment"),
            envs.MOONCAKE_STANDALONE_STORAGE.override(False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to load config"):
                MooncakeStoreConfig.load({"standalone_storage": False})


if __name__ == "__main__":
    unittest.main()
