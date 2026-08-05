import unittest

from funasr.bin.train_ds import _resolve_distributed_config


class TestDistributedConfig(unittest.TestCase):
    def test_nested_train_conf_is_supported(self):
        use_ddp, use_fsdp, use_deepspeed, config, trainer_conf = (
            _resolve_distributed_config(
                {
                    "train_conf": {
                        "use_deepspeed": True,
                        "deepspeed_config": "nested_ds.json",
                        "log_interval": 10,
                    }
                },
                world_size=2,
            )
        )

        self.assertFalse(use_ddp)
        self.assertFalse(use_fsdp)
        self.assertTrue(use_deepspeed)
        self.assertEqual(config, "nested_ds.json")
        self.assertEqual(trainer_conf, {"log_interval": 10})

    def test_top_level_values_override_train_conf(self):
        use_ddp, use_fsdp, use_deepspeed, config, trainer_conf = (
            _resolve_distributed_config(
                {
                    "use_deepspeed": False,
                    "deepspeed_config": "top_level_ds.json",
                    "train_conf": {
                        "use_deepspeed": True,
                        "deepspeed_config": "nested_ds.json",
                        "use_ddp": True,
                        "log_interval": 10,
                    },
                },
                world_size=2,
            )
        )

        self.assertTrue(use_ddp)
        self.assertFalse(use_fsdp)
        self.assertFalse(use_deepspeed)
        self.assertEqual(config, "top_level_ds.json")
        self.assertEqual(trainer_conf, {"log_interval": 10})

    def test_deepspeed_and_fsdp_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be enabled"):
            _resolve_distributed_config(
                {
                    "use_deepspeed": True,
                    "train_conf": {"use_fsdp": True},
                },
                world_size=2,
            )


if __name__ == "__main__":
    unittest.main()
