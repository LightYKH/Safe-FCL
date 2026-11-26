"""CPU checks for reading artifacts in their original schema namespace."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/core"))

from lora_classifier import probe_collection_runtime, probe_delta_core, probe_delta_store
from utils import artifact_schema, frozen_continual_schedule, model_artifact_identity


ARCHIVED_NAMESPACE = "archived-release"


def archived_schema(current):
    return ARCHIVED_NAMESPACE + "." + current.removeprefix(artifact_schema.CURRENT_NAMESPACE)


class ArtifactCompatibilityTests(unittest.TestCase):
    def setUp(self):
        digest = hashlib.sha256(ARCHIVED_NAMESPACE.encode("ascii")).hexdigest()
        namespace = patch.object(artifact_schema, "_ARCHIVED_NAMESPACE_SHA256", digest)
        namespace.start()
        self.addCleanup(namespace.stop)

    def test_probe_shards_validate_the_original_record_and_tensor_hashes(self):
        key = "layers.0.q_proj.lora_B.weight"
        tensors = {key: torch.tensor([[0.25, -0.5]], dtype=torch.float32)}
        spec = probe_delta_core.build_first_layer_lora_b_spec(tensors, expected_rank=2)
        current = probe_delta_core.RECORD_SCHEMA_VERSION
        legacy = archived_schema(current)
        round_schema = probe_delta_store.ROUND_SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as folder:
            for schema in (legacy, current):
                with self.subTest(schema=schema):
                    digest = probe_delta_core.compute_delta_tensor_payload_sha256(
                        tensors, spec, schema_version=schema,
                    )
                    metadata = probe_delta_core.build_probe_delta_record_metadata(
                        manifest_sha256="1" * 64, code_commit="2" * 40,
                        model_artifact_sha256="3" * 64, tokenizer_artifact_sha256="4" * 64,
                        lora_spec=spec, round_index=0, client_id=0, step_index=0,
                        role="benign", label=0, previous_record_id=None,
                        previous_history_digest=None, tensor_payload_sha256=digest,
                        collection_r0_tensor_sha256="5" * 64, delta_base_round=0,
                        delta_base_tensor_sha256="5" * 64,
                        lr_policy="task1_main_100_round_cosine_first_10", learning_rate=5e-5,
                        schema_version=schema,
                        step_samples=[dict(
                            sample_id=0, source_id="synthetic-row", content_hash="6" * 64,
                            raw_row_index=0, split_rank=0, probe_role="benign", probe_label=0,
                        )],
                    )
                    if schema == legacy:
                        self.assertNotEqual(
                            digest, probe_delta_core.compute_delta_tensor_payload_sha256(tensors, spec),
                        )
                    else:
                        self.assertEqual(
                            digest, probe_delta_core.compute_delta_tensor_payload_sha256(tensors, spec),
                        )
                    root = Path(folder) / schema
                    root.mkdir()
                    on_disk_schema = archived_schema(round_schema) if schema == legacy else round_schema
                    with patch.object(probe_delta_store, "ROUND_SCHEMA_VERSION", on_disk_schema):
                        probe_delta_store.write_probe_delta_round_shard(
                            root, 0, [{"metadata": metadata, "delta_tensors": tensors}],
                        )
                    loaded = probe_delta_store.load_and_validate_probe_delta_round_shard(
                        root / "round-000",
                    )
                    self.assertEqual(loaded["index"]["schema_version"], on_disk_schema)
                    self.assertEqual(loaded["records"][0]["metadata"], metadata)
                    self.assertTrue(torch.equal(loaded["records"][0]["delta_tensors"][key], tensors[key]))

    def test_legacy_plan_matches_local_model_and_tokenizer_identities(self):
        identity = model_artifact_identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (root / name).write_text("{}", encoding="utf-8")
            weights = root / "model.safetensors"
            weights.write_bytes(b"synthetic model identity")
            with patch.object(identity, "MODEL_SCHEMA_VERSION", archived_schema(identity.MODEL_SCHEMA_VERSION)), \
                 patch.object(identity, "TOKENIZER_SCHEMA_VERSION", archived_schema(identity.TOKENIZER_SCHEMA_VERSION)):
                plan = {
                    "model_artifact": identity.build_model_artifact_identity(root),
                    "tokenizer_artifact": identity.build_tokenizer_artifact_identity(root),
                }

            with patch.object(probe_collection_runtime, "_validate_plan", return_value=plan), \
                 patch.object(probe_collection_runtime, "_require_offline_environment"):
                verified = probe_collection_runtime.validate_local_artifacts_against_plan(plan, root)
                self.assertEqual(verified["model_identity"], plan["model_artifact"])
                self.assertEqual(verified["tokenizer_identity"], plan["tokenizer_artifact"])
                weights.write_bytes(b"changed synthetic model")
                with self.assertRaisesRegex(ValueError, "model artifact SHA"):
                    probe_collection_runtime.validate_local_artifacts_against_plan(plan, root)

    def test_frozen_schedule_keeps_its_original_schema_and_checksum(self):
        schedule = frozen_continual_schedule
        positions = list(range(160))
        payload = {
            "schema_version": archived_schema(schedule.SCHEMA_VERSION),
            "task_id": 1,
            "task_name": schedule.TASK_NAMES[1],
            "continual_round_offset": 0,
            "frozen_parameters": {
                "batch_size": 2, "client_pool_size": 500, "clients_per_round": 3,
                "gradient_accumulation_steps": 8, "local_optimizer_steps": 10,
                "num_clients": 10, "num_rounds": 100, "samples_per_selected_client": 160,
            },
            "rounds": [
                dict(stage_round=step + 1, global_round=step + 1,
                     learning_rate=schedule.frozen_cosine_lr(step),
                     clients=[0, 1, 2], sample_positions=positions,
                     batches=[positions[index:index + 16] for index in range(0, 160, 16)])
                for step in range(100)
            ],
        }
        raw = schedule.canonical_bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "schedule.json"
            path.write_bytes(raw)
            loaded = schedule.load_frozen_schedule(
                path, digest, task_id=1, continual_round_offset=0,
            )
            self.assertEqual(loaded["schema_version"], payload["schema_version"])
            self.assertEqual(loaded["_sha256"], digest)
            self.assertEqual(path.read_bytes(), raw)
