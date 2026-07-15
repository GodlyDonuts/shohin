import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import torch


TRAIN = Path(__file__).resolve().parent
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

import eval_counterfactual_cursor_action as evaluator  # noqa: E402
import score_counterfactual_cursor_action as scorer  # noqa: E402
from counterfactual_cursor_action_training import (  # noqa: E402
    adapter_state_payload,
    build_adapter,
)
from model import GPT, GPTConfig  # noqa: E402


LABELS = ["add", "subtract", "multiply", "remainder", "DONE"]
TOKEN_IDS = [10, 11, 12, 13, 14]


class FakeRuntime:
    arm_name = "orbit_interchange"
    adapter_contract = {
        "arm": "orbit_interchange",
        "adapter_type": "centered_three_bit_q_sidecar",
        "parameters": 192,
    }

    def forward_logits(self, examples, effective_cursors):
        logits = torch.full((len(examples), 32), -5.0)
        for row, cursor in enumerate(effective_cursors):
            logits[row, TOKEN_IDS[cursor]] = 5.0
        return logits


def inference_result(token_id, *, tie=False):
    logits = [-4.0] * 5
    index = TOKEN_IDS.index(token_id)
    logits[index] = 3.0
    if tie:
        other = (index + 1) % 5
        logits[other] = 3.0
        top_count = 2
        unique = False
        prediction = None
    else:
        top_count = 1
        unique = True
        prediction = token_id
    argmax_index = logits.index(max(logits))
    return {
        "full_vocab_argmax_token_id": token_id,
        "full_vocab_argmax_logit": 3.0,
        "full_vocab_top_count": top_count,
        "full_vocab_unique_top1": unique,
        "full_vocab_prediction_token_id": prediction,
        "restricted_argmax_index": argmax_index,
        "restricted_argmax_token_id": TOKEN_IDS[argmax_index],
        "restricted_top_count": top_count,
        "restricted_unique_top1": unique,
        "restricted_prediction_token_id": prediction,
        "restricted_logits": logits,
    }


def synthetic_gold():
    orders = {
        "confirmation-r08-k00-p00": TOKEN_IDS,
        "confirmation-r09-k00-p00": TOKEN_IDS,
        "confirmation-r08-k00-p01": [11, 10, 12, 13, 14],
        "confirmation-r09-k00-p01": [11, 10, 12, 13, 14],
    }
    sources = {}
    cells = {}
    source_cells = {}
    for source_id, targets in orders.items():
        renderer = int(source_id.split("-r", 1)[1][:2])
        sources[source_id] = {"source_id": source_id, "renderer_id": renderer}
        ids = []
        for cursor, token_id in enumerate(targets):
            cell_id = f"{source_id}-c{cursor}"
            cells[cell_id] = {
                "cell_id": cell_id,
                "source_id": source_id,
                "cursor": cursor,
                "target_token_id": token_id,
            }
            ids.append(cell_id)
        source_cells[source_id] = tuple(ids)
    adjacent = (
        {
            "left_source_id": "confirmation-r08-k00-p00",
            "right_source_id": "confirmation-r08-k00-p01",
            "swap_index": 0,
        },
        {
            "left_source_id": "confirmation-r09-k00-p00",
            "right_source_id": "confirmation-r09-k00-p01",
            "swap_index": 0,
        },
    )
    groups = (
        {
            "source_ids": [
                "confirmation-r08-k00-p00", "confirmation-r09-k00-p00"
            ]
        },
        {
            "source_ids": [
                "confirmation-r08-k00-p01", "confirmation-r09-k00-p01"
            ]
        },
    )
    canary = {"label_order": LABELS, "label_token_ids": TOKEN_IDS}
    return scorer.GoldIndex(
        canary=canary,
        audit={},
        canary_sha256="a" * 64,
        audit_sha256="b" * 64,
        sources=sources,
        cells=cells,
        source_cells=source_cells,
        adjacent_pairs=adjacent,
        content_groups=groups,
    )


def synthetic_artifact(gold, *, perfect=True, tie_cell=None, arm="orbit_interchange"):
    records = {}
    rows = []
    for cell_id, cell in gold.cells.items():
        source_cells = gold.source_cells[cell["source_id"]]
        semantic = cell["cursor"]
        if perfect:
            canonical = cell["target_token_id"]
        else:
            canonical = gold.cells[source_cells[(semantic + 1) % 5]]["target_token_id"]
        clamped = gold.cells[source_cells[0]]["target_token_id"]
        deranged = gold.cells[source_cells[(semantic + 1) % 5]]["target_token_id"]
        row = {
            "cell_id": cell_id,
            "conditions": {
                "canonical": inference_result(canonical, tie=cell_id == tie_cell),
                "clamped_zero": inference_result(clamped),
                "deranged_cycle": inference_result(deranged),
            },
        }
        rows.append(row)
        records[cell_id] = row
    raw = {
        "condition_contract": [
            {"name": name, "cursor_map": list(cursor_map)}
            for name, cursor_map in evaluator.CONDITION_LIBRARY.items()
        ],
        "inference_records": rows,
    }
    return scorer.ChainArtifact(
        arm_name=arm,
        raw=raw,
        receipt={},
        receipt_sha256="c" * 64,
        records=records,
    )


class CounterfactualCursorEvaluatorTest(unittest.TestCase):
    def test_prediction_ties_have_no_prediction(self):
        logits = torch.zeros(20)
        record = evaluator.prediction_record(logits, TOKEN_IDS)
        self.assertEqual(record["full_vocab_top_count"], 20)
        self.assertIsNone(record["full_vocab_prediction_token_id"])
        self.assertEqual(record["restricted_top_count"], 5)
        self.assertIsNone(record["restricted_prediction_token_id"])

    def test_gold_blind_batched_inference_and_conditions(self):
        examples = [
            evaluator.InferenceExample(
                cell_id=f"source-c{cursor}",
                source_id="source",
                semantic_cursor=cursor,
                prompt_token_ids=(1, 2, 3),
                text_prompt_token_ids_by_cursor=tuple(
                    (1, 2, 3, index + 4) for index in range(5)
                ),
            )
            for cursor in range(5)
        ]
        conditions = evaluator.parse_conditions(
            "canonical,clamped_zero,deranged_cycle"
        )
        records, forwards = evaluator.evaluate_examples(
            examples, FakeRuntime(), TOKEN_IDS, conditions, batch_size=2
        )
        self.assertEqual(forwards, 9)
        self.assertEqual(
            [row["conditions"]["canonical"]["restricted_prediction_token_id"]
             for row in records],
            TOKEN_IDS,
        )
        self.assertEqual(
            {row["conditions"]["clamped_zero"]["restricted_prediction_token_id"]
             for row in records},
            {TOKEN_IDS[0]},
        )
        serialized = json.dumps(records)
        for forbidden in ("target", "gold", "correct", "accuracy", "score"):
            self.assertNotIn(forbidden, serialized.lower())

    def test_receipt_has_no_inference_values(self):
        bindings = {
            "canary_file_sha256": "a" * 64,
            "canary_payload_sha256": "b" * 64,
            "canary_contract_sha256": "c" * 64,
            "canary_audit_file_sha256": "d" * 64,
            "base_checkpoint_sha256": "e" * 64,
            "base_checkpoint_step": 260000,
            "adapter_sha256": "f" * 64,
            "adapter_implementation_commit": "1" * 40,
            "tokenizer_sha256": "2" * 64,
            "confirmation_sources_sha256": "3" * 64,
            "confirmation_cells_sha256": "4" * 64,
            "training_manifest_sha256": "7" * 64,
            "code_sha256": {"evaluator": "5" * 64},
        }
        receipt = evaluator.build_receipt(
            job_identity={
                "scheduler": "slurm", "job_id": "7", "array_task_id": "none",
                "attempt_id": "1",
            },
            arm_name="ordinary_loss",
            bindings=bindings,
            row_count=20,
            condition_count=3,
            forward_count=3,
            raw_sha256="6" * 64,
            raw_bytes=1234,
        )
        serialized = json.dumps(receipt).lower()
        for forbidden in ("prediction", "argmax", "logit", "correct", "score"):
            self.assertNotIn(forbidden, serialized)

    def test_exact_adapter_payload_uses_shared_factory_and_full_model(self):
        cfg = GPTConfig(
            vocab_size=32, n_layer=1, n_head=9, n_kv_head=3,
            d_model=576, d_ff=32, seq_len=16,
        )
        base = GPT(cfg).eval()
        step = 260000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.pt"
            torch.save({"cfg": cfg.__dict__, "model": base.state_dict(), "step": step}, base_path)
            base_path.chmod(0o444)
            base_sha = evaluator.hash_regular_file(base_path)
            bound = evaluator.BoundCanary(
                document={
                    "payload_sha256": "a" * 64,
                    "contract_sha256": "b" * 64,
                    "tokenizer_sha256": "c" * 64,
                    "implementation_identity": {
                        "git_commit": "2" * 40, "file_sha256": {},
                    },
                    "splits": {"confirmation": {
                        "sources_sha256": "d" * 64, "cells_sha256": "e" * 64,
                    }},
                },
                audit={},
                canary_sha256="f" * 64,
                audit_sha256="1" * 64,
            )
            seed = 19
            adapter, spec = build_adapter("ordinary_loss", cfg, seed)
            state_payload = adapter_state_payload(adapter, spec)
            commit = "2" * 40
            payload = {
                "schema": "counterfactual_cursor_action_adapter_v1",
                "arm": "ordinary_loss",
                **state_payload,
                "bindings": {
                    "base_sha256": base_sha,
                    "base_step": step,
                    "canary_sha256": bound.canary_sha256,
                    "canary_payload_sha256": bound.document["payload_sha256"],
                    "audit_sha256": bound.audit_sha256,
                    "tokenizer_sha256": bound.document["tokenizer_sha256"],
                    "implementation_commit": commit,
                },
                "training": {"seed": seed},
                "resource_ledger": {"trainable_parameters": 192},
            }
            adapter_path = root / "adapter.pt"
            torch.save(payload, adapter_path)
            adapter_path.chmod(0o444)
            adapter_sha = evaluator.hash_regular_file(adapter_path)
            training_manifest = evaluator.TrainingManifest(
                document={},
                sha256="7" * 64,
                path=root / "training_manifest.json",
                entries={"ordinary_loss": {"artifact_sha256": adapter_sha}},
                artifact_paths={"ordinary_loss": evaluator._absolute(adapter_path)},
            )
            runtime = evaluator.load_model_and_adapter(
                base_path, adapter_path, device="cpu", bound=bound,
                base_sha256=base_sha, adapter_sha256=adapter_sha,
                training_manifest=training_manifest,
            )
            self.assertEqual(runtime.arm_name, "ordinary_loss")
            self.assertEqual(runtime.adapter_contract, state_payload["adapter_spec"])
            example = evaluator.InferenceExample(
                "source-c0", "source", 0, (1, 2, 3),
                tuple((1, 2, 3, cursor + 4) for cursor in range(5)),
            )
            observed = runtime.forward_logits([example], [0])
            self.assertEqual(tuple(observed.shape), (1, 32))

    def test_training_manifest_binds_all_arms_and_matched_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = "9" * 40
            bound = evaluator.BoundCanary(
                document={
                    "payload_sha256": "1" * 64,
                    "contract_sha256": "2" * 64,
                    "tokenizer_sha256": "3" * 64,
                    "implementation_identity": {
                        "git_commit": commit, "file_sha256": {},
                    },
                    "splits": {"confirmation": {
                        "sources_sha256": "4" * 64,
                        "cells_sha256": "5" * 64,
                    }},
                },
                audit={},
                canary_sha256="6" * 64,
                audit_sha256="7" * 64,
            )
            entries = []
            for index, arm in enumerate(evaluator.ARMS):
                arm_dir = root / arm
                arm_dir.mkdir()
                artifact = arm_dir / "adapter.pt"
                artifact.write_bytes(f"adapter-{arm}".encode("ascii"))
                artifact.chmod(0o444)
                entries.append({
                    "arm": arm,
                    "artifact": f"{arm}/adapter.pt",
                    "artifact_sha256": evaluator.hash_regular_file(artifact),
                    "initial_adapter_sha256": (
                        "8" * 64 if arm in evaluator.MATCHED_ARMS else f"{index:x}" * 64
                    ),
                    "final_adapter_sha256": f"{index + 1:x}" * 64,
                    "trainable_scalars": evaluator.EXPECTED_PARAMETERS[arm],
                    "updates": evaluator.FROZEN_UPDATES,
                    "relation_coefficient": (
                        evaluator.EXPECTED_RELATION_COEFFICIENTS[arm]
                    ),
                    "fixed_training_compute_proxy": (
                        {"positions": 1}
                        if arm in evaluator.MATCHED_ARMS else {"positions": index + 1}
                    ),
                })
            base_sha = "a" * 64
            document = {
                "schema": evaluator.MANIFEST_SCHEMA,
                "arms": entries,
                "arm_order": list(evaluator.ARMS),
                "bindings": {
                    "base_sha256": base_sha,
                    "base_step": evaluator.FROZEN_BASE_STEP,
                    "canary_sha256": bound.canary_sha256,
                    "canary_payload_sha256": bound.document["payload_sha256"],
                    "audit_sha256": bound.audit_sha256,
                    "tokenizer_sha256": bound.document["tokenizer_sha256"],
                    "implementation_commit": commit,
                },
                "all_arms_complete": True,
                "score_bearing_evaluation_performed": False,
            }
            manifest = root / "training_manifest.json"
            manifest.write_text(json.dumps(document), encoding="ascii")
            manifest.chmod(0o444)
            loaded = evaluator.load_training_manifest(
                manifest,
                expected_sha256=evaluator.hash_regular_file(manifest),
                bound=bound,
                base_sha256=base_sha,
            )
            self.assertEqual(set(loaded.entries), set(evaluator.ARMS))

            bad = json.loads(json.dumps(document))
            bad["arms"][1]["fixed_training_compute_proxy"] = {"positions": 2}
            bad_manifest = root / "bad_manifest.json"
            bad_manifest.write_text(json.dumps(bad), encoding="ascii")
            bad_manifest.chmod(0o444)
            with self.assertRaisesRegex(ValueError, "compute ledgers differ"):
                evaluator.load_training_manifest(
                    bad_manifest,
                    expected_sha256=evaluator.hash_regular_file(bad_manifest),
                    bound=bound,
                    base_sha256=base_sha,
                )


class CounterfactualCursorScorerTest(unittest.TestCase):
    def test_independent_chain_requires_exact_receipt_and_rehashes_bindings(self):
        original = synthetic_gold()
        canary = {
            **original.canary,
            "payload_sha256": "1" * 64,
            "contract_sha256": "2" * 64,
            "tokenizer_sha256": "3" * 64,
            "splits": {"confirmation": {
                "sources_sha256": "4" * 64,
                "cells_sha256": "5" * 64,
            }},
        }
        gold = scorer.GoldIndex(
            canary=canary,
            audit=original.audit,
            canary_sha256=original.canary_sha256,
            audit_sha256=original.audit_sha256,
            sources=original.sources,
            cells=original.cells,
            source_cells=original.source_cells,
            adjacent_pairs=original.adjacent_pairs,
            content_groups=original.content_groups,
        )
        synthetic = synthetic_artifact(gold)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pt"
            adapter = root / "adapter.pt"
            base.write_bytes(b"frozen base")
            adapter.write_bytes(b"frozen adapter")
            base.chmod(0o444)
            adapter.chmod(0o444)
            bindings = {
                "canary_file_sha256": gold.canary_sha256,
                "canary_payload_sha256": canary["payload_sha256"],
                "canary_contract_sha256": canary["contract_sha256"],
                "canary_audit_file_sha256": gold.audit_sha256,
                "base_checkpoint_sha256": scorer.hash_regular_file(base),
                "base_checkpoint_step": 260000,
                "adapter_sha256": scorer.hash_regular_file(adapter),
                "adapter_implementation_commit": "6" * 40,
                "tokenizer_sha256": canary["tokenizer_sha256"],
                "confirmation_sources_sha256": "4" * 64,
                "confirmation_cells_sha256": "5" * 64,
                "training_manifest_sha256": "7" * 64,
                "code_sha256": {
                    name: scorer.hash_regular_file(path, require_read_only=False)
                    for name, path in scorer.LIVE_CODE_PATHS.items()
                },
            }
            raw = evaluator.build_raw_artifact(
                runtime=FakeRuntime(),
                bindings=bindings,
                restricted_token_ids=TOKEN_IDS,
                conditions=synthetic.raw["condition_contract"],
                records=synthetic.raw["inference_records"],
                forward_count=3,
            )
            raw_path = root / "raw.json"
            receipt_path = root / "receipt.json"
            raw_sha256, raw_bytes = evaluator.write_exclusive_read_only_json(
                raw_path, raw
            )
            receipt = evaluator.build_receipt(
                job_identity={
                    "scheduler": "slurm",
                    "job_id": "99",
                    "array_task_id": "none",
                    "attempt_id": "1",
                },
                arm_name="orbit_interchange",
                bindings=bindings,
                row_count=len(gold.cells),
                condition_count=3,
                forward_count=3,
                raw_sha256=raw_sha256,
                raw_bytes=raw_bytes,
            )
            receipt_sha256, _ = evaluator.write_exclusive_read_only_json(
                receipt_path, receipt
            )
            training_manifest = scorer.TrainingManifest(
                sha256="7" * 64,
                base_sha256=scorer.hash_regular_file(base),
                implementation_commit="6" * 40,
                entries={
                    "orbit_interchange": {
                        "artifact_sha256": scorer.hash_regular_file(adapter),
                    },
                },
                artifact_paths={
                    "orbit_interchange": scorer._absolute(adapter),
                },
            )
            with self.assertRaisesRegex(ValueError, "exact receipt"):
                scorer.validate_chain(
                    gold=gold,
                    training_manifest=training_manifest,
                    base_path=base,
                    adapter_path=adapter,
                    raw_path=raw_path,
                    receipt_path=receipt_path,
                    expected_receipt_sha256="0" * 64,
                )
            artifact = scorer.validate_chain(
                gold=gold,
                training_manifest=training_manifest,
                base_path=base,
                adapter_path=adapter,
                raw_path=raw_path,
                receipt_path=receipt_path,
                expected_receipt_sha256=receipt_sha256,
            )
            self.assertEqual(artifact.arm_name, "orbit_interchange")
            self.assertEqual(len(artifact.records), len(gold.cells))

    def test_perfect_scores_all_relations_and_ablations(self):
        gold = synthetic_gold()
        artifact = synthetic_artifact(gold)
        result = scorer.score_mode(gold, artifact, "restricted")
        canonical = result["canonical"]
        self.assertEqual(canonical["accuracy"]["numerator"], 20)
        self.assertEqual(canonical["exact_five_action_groups"]["numerator"], 4)
        self.assertEqual(
            canonical["directed_cursor_switch"]["exact_source_to_donor_switch"],
            scorer.ratio(80, 80),
        )
        self.assertEqual(
            canonical["adjacent_equivariance"][
                "affected_cross_position_relation_and_correct"
            ],
            scorer.ratio(4, 4),
        )
        self.assertEqual(
            canonical["adjacent_equivariance"][
                "unaffected_same_position_invariance_and_correct"
            ],
            scorer.ratio(6, 6),
        )
        self.assertEqual(
            canonical["renderer_invariance"]["prediction_invariance_and_correct"],
            scorer.ratio(10, 10),
        )
        clamped = result["ablations"]["clamped_zero"]
        self.assertAlmostEqual(
            clamped["conditioned_on_canonical_exact_groups"]["accuracy"]["proportion"],
            0.2,
        )
        deranged = result["ablations"]["deranged_cycle"]
        self.assertEqual(
            deranged["conditioned_on_canonical_exact_groups"]["accuracy"]["numerator"],
            0,
        )

    def test_unique_top1_tie_fails_cell_and_group(self):
        gold = synthetic_gold()
        tied_cell = next(iter(gold.cells))
        artifact = synthetic_artifact(gold, tie_cell=tied_cell)
        result = scorer.score_mode(gold, artifact, "restricted")["canonical"]
        self.assertEqual(result["accuracy"]["numerator"], 19)
        self.assertEqual(result["unique_top1_ties"]["numerator"], 1)
        self.assertEqual(result["exact_five_action_groups"]["numerator"], 3)

    def test_paired_bootstrap_is_content_clustered_and_deterministic(self):
        gold = synthetic_gold()
        treatment = synthetic_artifact(gold)
        control = synthetic_artifact(gold, perfect=False, arm="ordinary_loss")
        first = scorer.paired_bootstrap_comparisons(
            gold, treatment, {"ordinary_loss": control}, "restricted",
            replicates=200, seed=17,
        )
        second = scorer.paired_bootstrap_comparisons(
            gold, treatment, {"ordinary_loss": control}, "restricted",
            replicates=200, seed=17,
        )
        self.assertEqual(first, second)
        comparison = first["controls"]["ordinary_loss"]
        self.assertEqual(first["content_cluster_count"], 2)
        self.assertTrue(first["paired_within_content_cluster"])
        self.assertEqual(comparison["observed_difference"], 1.0)
        self.assertEqual(comparison["one_sided_95_percent_lower_bound"], 1.0)

    def test_selector_gate_uses_full_vocab_and_never_authorizes_reasoning(self):
        gold = synthetic_gold()
        treatment = synthetic_artifact(gold)
        full_vocab = scorer.score_mode(gold, treatment, "full_vocab")
        full_vocab["canonical"]["per_renderer"] = {
            str(renderer): {"accuracy": scorer.ratio(1, 1)}
            for renderer in range(5)
        }
        treatment_score = {"full_vocab": full_vocab}
        comparison = {
            "controls": {
                arm: {
                    "observed_difference": 0.20,
                    "bonferroni_simultaneous_95_percent_lower_bound": 0.10,
                }
                for arm in ("ordinary_loss", "relation_sham")
            }
        }
        decision = scorer.selector_gate_decision(treatment_score, comparison)
        self.assertTrue(decision["selector_checks_passed"])
        self.assertEqual(decision["decision"], "selector_go_executor_pending")
        self.assertTrue(decision["atomic_executor_gate_pending"])
        self.assertTrue(decision["one_call_done_eos_gate_pending"])
        self.assertFalse(decision["reasoning_claim_authorized"])

    def test_duplicate_json_keys_and_symlinks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            scorer.strict_json_loads('{"x":1,"x":2}')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}", encoding="ascii")
            target.chmod(0o444)
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                scorer.read_regular_file(link)
            real_directory = root / "real"
            real_directory.mkdir()
            nested = real_directory / "nested.json"
            nested.write_text("{}", encoding="ascii")
            nested.chmod(0o444)
            directory_link = root / "directory-link"
            directory_link.symlink_to(real_directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                scorer.read_regular_file(directory_link / "nested.json")

    def test_exclusive_writer_is_fsynced_read_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            scorer.write_exclusive_read_only_json(path, {"schema": "test"})
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            self.assertTrue(stat.S_ISREG(path.stat().st_mode))
            with self.assertRaisesRegex(ValueError, "existing output"):
                scorer.write_exclusive_read_only_json(path, {"schema": "test"})


if __name__ == "__main__":
    unittest.main()
