from __future__ import annotations

import builtins
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from pipeline import generate_dws_single_completion_v1 as dws_sc  # noqa: E402
from tokenizers import (  # noqa: E402
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

apply_microstep = dws_sc.apply_microstep
canonical_state = dws_sc.canonical_state
parse_state = dws_sc.parse_state


REPLICATION_SOURCE = ROOT / "artifacts/evals/digitwise_recurrent_v2_heldout.jsonl"
GENERATOR_PATH = ROOT / "pipeline/generate_dws_single_completion_v1.py"
PARENT_CHECKPOINT_SHA256 = "a" * 64
ISOLATED_PYTHON = (sys.executable, "-I", "-S", "-B")


def run_isolated_python(
    script: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*ISOLATED_PYTHON, "-c", script, *arguments],
        cwd=ROOT,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )


def run_generator_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*ISOLATED_PYTHON, "-W", "error", str(GENERATOR_PATH), *arguments],
        cwd=ROOT,
        env={},
        capture_output=True,
        text=True,
        check=False,
    )


def install_runtime_phase_action(
    publication: Path, phase: str, action: Callable[[], None]
) -> dict[str, int]:
    state = {"calls": 0, "armed": 1}

    def hook(event: str, arguments: tuple[object, ...]) -> None:
        if (
            state["armed"]
            and event == dws_sc._RUNTIME_PHASE_AUDIT_EVENT
            and arguments == (phase, str(publication))
        ):
            state["armed"] = 0
            state["calls"] += 1
            action()

    sys.addaudithook(hook)
    return state


@dataclass(frozen=True)
class BundleFixture:
    root: Path
    publication: Path
    bundle: Path
    tokenizer_path: Path
    tokenizer_sha256: str
    source_bindings_sha256: str
    runtime_bindings_sha256: str
    receipt: dict


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def make_lossless_byte_tokenizer(path: Path) -> tuple[Path, str]:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=258,
        min_frequency=10**9,
        special_tokens=["<unk>", dws_sc.EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(["ASCII"], trainer=trainer)
    tokenizer.save(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def make_tree_writable(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            os.chmod(Path(root) / name, 0o600)
        for name in directories:
            os.chmod(Path(root) / name, 0o700)
    os.chmod(path, 0o700)


def remove_tree(path: Path) -> None:
    make_tree_writable(path)
    if path.exists():
        shutil.rmtree(path)


def seal_release(path: Path) -> None:
    bundle = path / dws_sc.BUNDLE_DIRECTORY_NAME
    for artifact in bundle.iterdir():
        os.chmod(artifact, 0o444)
    os.chmod(path / dws_sc.SEALED_ROOT_NAME, 0o444)
    os.chmod(bundle, 0o555)
    os.chmod(path, 0o555)


def external_receipt(publication: Path) -> dict[str, int | str]:
    payload = (publication / dws_sc.SEALED_ROOT_NAME).read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def verification_kwargs(
    fixture: BundleFixture,
    publication: Path | None = None,
    receipt: dict | None = None,
    *,
    source_bindings_sha256: str | None = None,
    runtime_bindings_sha256: str | None = None,
    parent_checkpoint_sha256: str = PARENT_CHECKPOINT_SHA256,
) -> dict:
    publication = publication or fixture.publication
    receipt = receipt or fixture.receipt
    return {
        "external_manifest_path": publication / dws_sc.SEALED_ROOT_NAME,
        "expected_external_manifest_sha256": receipt["external_manifest_sha256"]
        if "external_manifest_sha256" in receipt
        else receipt["sha256"],
        "expected_external_manifest_bytes": receipt["external_manifest_bytes"]
        if "external_manifest_bytes" in receipt
        else receipt["bytes"],
        "tokenizer_path": fixture.tokenizer_path,
        "expected_tokenizer_sha256": fixture.tokenizer_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "replication_source": REPLICATION_SOURCE,
        "expected_replication_source_sha256": dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        "expected_source_bindings_sha256": source_bindings_sha256
        or fixture.source_bindings_sha256,
        "expected_runtime_bindings_sha256": runtime_bindings_sha256
        or fixture.runtime_bindings_sha256,
        "mode": "test",
        "seed": dws_sc.GENERATION_SEED,
        "train_per_cell": 8,
        "development_per_cell": 1,
        "lane_length": dws_sc.LANE_LENGTH,
    }


def mutable_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    make_tree_writable(destination)
    return destination


def replace_artifact_with_byte_identical_inode(
    publication: Path, artifact_name: str, aside: Path
) -> tuple[tuple[int, int], tuple[int, int]]:
    bundle = publication / dws_sc.BUNDLE_DIRECTORY_NAME
    artifact = bundle / artifact_name
    payload = artifact.read_bytes()
    original_metadata = artifact.stat()
    bundle.chmod(0o700)
    artifact.rename(aside)
    artifact.write_bytes(payload)
    artifact.chmod(0o444)
    bundle.chmod(0o555)
    replacement_metadata = artifact.stat()
    return (
        (original_metadata.st_dev, original_metadata.st_ino),
        (replacement_metadata.st_dev, replacement_metadata.st_ino),
    )


def restore_replaced_artifact(
    publication: Path, artifact_name: str, aside: Path
) -> None:
    bundle = publication / dws_sc.BUNDLE_DIRECTORY_NAME
    artifact = bundle / artifact_name
    bundle.chmod(0o700)
    if artifact.exists():
        artifact.unlink()
    if aside.exists():
        aside.rename(artifact)
    artifact.chmod(0o444)
    bundle.chmod(0o555)


def reseal_artifact(publication: Path, artifact_name: str | None = None) -> dict:
    root_path = publication / dws_sc.SEALED_ROOT_NAME
    root = read_json(root_path)
    if artifact_name is not None:
        payload = (
            publication / dws_sc.BUNDLE_DIRECTORY_NAME / artifact_name
        ).read_bytes()
        root["artifacts"][artifact_name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    root_path.write_bytes(dws_sc.pretty_json_bytes(root))
    seal_release(publication)
    return external_receipt(publication)


@pytest.fixture(scope="module")
def small_bundle(tmp_path_factory: pytest.TempPathFactory) -> BundleFixture:
    root = tmp_path_factory.mktemp("dws-single-completion")
    tokenizer_path, tokenizer_sha256 = make_lossless_byte_tokenizer(
        root / "tokenizer.json"
    )
    source_sha256 = dws_sc.source_bindings_sha256()
    runtime_sha256 = dws_sc.runtime_bindings_sha256()
    publication = root / "publication"
    receipt = dws_sc.build_bundle(
        out_dir=publication,
        tokenizer_path=tokenizer_path,
        expected_tokenizer_sha256=tokenizer_sha256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        replication_source=REPLICATION_SOURCE,
        expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        expected_source_bindings_sha256=source_sha256,
        expected_runtime_bindings_sha256=runtime_sha256,
        mode="test",
        seed=dws_sc.GENERATION_SEED,
        train_per_cell=8,
        development_per_cell=1,
        lane_length=dws_sc.LANE_LENGTH,
    )
    fixture = BundleFixture(
        root=root,
        publication=publication,
        bundle=publication / dws_sc.BUNDLE_DIRECTORY_NAME,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        source_bindings_sha256=source_sha256,
        runtime_bindings_sha256=runtime_sha256,
        receipt=receipt,
    )
    yield fixture
    remove_tree(root)


def test_frozen_counts_optimizer_and_external_verification(
    small_bundle: BundleFixture,
) -> None:
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert ISOLATED_PYTHON[1:] == ("-I", "-S", "-B")
    assert sys.flags.isolated == 1
    assert sys.flags.no_site == 1
    assert sys.flags.dont_write_bytecode == 1
    assert not {"site", "sitecustomize", "usercustomize"}.intersection(sys.modules)
    cells = len(dws_sc.OPERATIONS) * len(dws_sc.INTERMEDIATE_PATTERNS)
    assert cells * dws_sc.TRAIN_PER_CELL == dws_sc.TRAIN_EPISODES == 2_048
    assert cells * dws_sc.DEVELOPMENT_PER_CELL == dws_sc.DEVELOPMENT_EPISODES == 256
    assert dws_sc.UPDATES_PER_ARM == 1_024
    assert dws_sc.LANES_PER_PACK == len(dws_sc.LANE_ROLES) == 7
    assert dws_sc.OPTIMIZER_CONTRACT["adamw"]["eps"] == 1e-8
    assert dws_sc.OPTIMIZER_CONTRACT["precision"]["tf32"] is False
    assert dws_sc.SOURCE_RETIREMENT_GATES == {
        "paired_carry_target_switch": {
            "metric": "exact_success_rate",
            "overall": {"minimum_successes": 9, "cases": 12},
            "each_width": {"minimum_successes": 3, "cases": 4},
        },
        "counterfactual_full_target_exactness": {
            "metric": "exact_success_rate",
            "overall": {"minimum_successes": 9, "cases": 12},
            "each_width": {"minimum_successes": 3, "cases": 4},
        },
        "output_switch": {
            "metric": "exact_success_rate",
            "overall": {"minimum_successes": 11, "cases": 12},
            "each_width": {"minimum_successes": 4, "cases": 4},
        },
        "recovery_vs_full_history": {
            "metric": "paired_exactness_rate_difference",
            "overall": {
                "minimum_rate": {"numerator": 2, "denominator": 5},
                "cases": 12,
            },
            "each_width": {
                "minimum_rate": {"numerator": 1, "denominator": 2},
                "cases": 4,
            },
        },
    }
    result = dws_sc.verify_bundle(
        small_bundle.publication, **verification_kwargs(small_bundle)
    )
    assert result["verified"] is True
    assert result["semantic_replay"] is True
    assert result["training_authorized"] is False


def test_exact_2048_episode_reconstruction_and_reviewed_row_builder() -> None:
    replication_board = dws_sc.load_replication_board(
        REPLICATION_SOURCE, dws_sc.KNOWN_REPLICATION_SOURCE_SHA256
    )
    cross_inventory = dws_sc._aggregate_episode_inventory(replication_board)
    reserved_signatures = {dws_sc.episode_signature(row) for row in replication_board}
    rng = dws_sc._new_bound_random(
        dws_sc.sha256_bytes((str(dws_sc.GENERATION_SEED) + "\0train").encode("ascii"))
    )
    episodes = dws_sc.generate_balanced_episodes(
        rng=rng,
        split="train",
        per_cell=dws_sc.TRAIN_PER_CELL,
        reserved_signatures=reserved_signatures,
        forbidden_inventory=cross_inventory,
        reserve_inventory_within_split=False,
        require_interventions=False,
    )
    assert len(episodes) == 2_048
    assert dws_sc.hash_json(episodes) == dws_sc.KNOWN_TRAIN_EPISODES_SHA256
    assert Counter(
        (row["operation"], tuple(row["intermediate_carry_pattern"])) for row in episodes
    ) == {
        (operation, pattern): dws_sc.TRAIN_PER_CELL
        for operation in dws_sc.OPERATIONS
        for pattern in dws_sc.INTERMEDIATE_PATTERNS
    }
    assert sum(len(dws_sc.rows_from_episode(row)) for row in episodes) == 18_432


def test_durable_publication_modes_links_inventory_and_external_receipt(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    assert stat.S_IMODE(small_bundle.publication.lstat().st_mode) == 0o555
    assert stat.S_IMODE(small_bundle.bundle.lstat().st_mode) == 0o555
    assert sorted(path.name for path in small_bundle.publication.iterdir()) == sorted(
        (dws_sc.BUNDLE_DIRECTORY_NAME, dws_sc.SEALED_ROOT_NAME)
    )
    assert sorted(path.name for path in small_bundle.bundle.iterdir()) == sorted(
        dws_sc.ARTIFACT_NAMES
    )
    for path in [
        small_bundle.publication / dws_sc.SEALED_ROOT_NAME,
        *small_bundle.bundle.iterdir(),
    ]:
        metadata = path.lstat()
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1
    root = read_json(small_bundle.publication / dws_sc.SEALED_ROOT_NAME)
    assert "payload_sha256" not in root
    assert root["publication_layout"]["manifest_self_authentication_allowed"] is False
    assert root["inputs"]["location_contract"] == {
        "policy": dws_sc.INPUT_LOCATION_POLICY,
        "path_is_identity": False,
        "sha256_is_identity": True,
        "regular_non_symlink_file_required": True,
        "build_and_verify_locations_may_differ": True,
    }
    assert (
        root["inputs"]["runtime_bindings_sha256"]
        == small_bundle.runtime_bindings_sha256
    )
    assert root["inputs"]["runtime_bindings"] == dws_sc.runtime_bindings()
    runtime = root["inputs"]["runtime_bindings"]
    for module_name, expected_path in (
        ("digitwise_protocol", ROOT / "train/digitwise_protocol.py"),
        (
            "pipeline.generate_digitwise_recurrent_v1",
            ROOT / "pipeline/generate_digitwise_recurrent_v1.py",
        ),
    ):
        module_receipt = runtime["reviewed_modules"][module_name]
        source_receipt = root["inputs"]["source_bindings"][
            module_receipt["source_binding_path"]
        ]
        assert Path(module_receipt["resolved_path"]) == expected_path.resolve()
        assert {
            "bytes": module_receipt["bytes"],
            "sha256": module_receipt["sha256"],
        } == source_receipt
    assert runtime["python"]["executable"]["sha256"]
    assert runtime["python"]["version"] == sys.version
    assert runtime["schema"] == "shohin-dws-single-completion-runtime-bindings-v6"
    assert runtime["python"]["startup"]["required_invocation_flags"] == [
        "-I",
        "-S",
        "-B",
    ]
    assert runtime["python"]["startup"]["flags"]["values"]["isolated"] == 1
    generator_builtins = runtime["generator_builtins"]
    assert generator_builtins["sealed"] is True
    assert generator_builtins["exact_key_and_value_identity_required"] is True
    assert generator_builtins["entry_count"] == len(builtins.__dict__)
    assert dws_sc.runtime_bindings.__builtins__ is dws_sc._FROZEN_GENERATOR_BUILTINS
    assert type(dws_sc.__dict__).__name__ == "mappingproxy"
    with pytest.raises(TypeError):
        dws_sc.__dict__["_pack_payload"] = object()
    live_classes = runtime["generator_live_implementations"]["class_methods"]
    assert all(
        receipt["private_frozen_builtins"] is True
        for receipt in runtime["generator_live_implementations"][
            "module_functions"
        ].values()
    )
    assert live_classes["FrozenTokenizer.__init__"]["private_frozen_builtins"] is True
    assert (
        live_classes["_PinnedDirectory.descriptor.fget"]["private_frozen_builtins"]
        is True
    )
    generator = runtime["executing_generator"]
    assert (
        Path(generator["resolved_path"])
        == (ROOT / "pipeline/generate_dws_single_completion_v1.py").resolve()
    )
    assert {"bytes": generator["bytes"], "sha256": generator["sha256"]} == root[
        "inputs"
    ]["source_bindings"][generator["source_binding_path"]]
    assert {
        "struct",
        "_struct",
        "_hashlib",
        "posix",
        "_json",
        "_sre",
        "_stat",
        "_ctypes",
        "json.decoder",
        "json.encoder",
        "json.scanner",
        "importlib.metadata._adapters",
        "importlib.metadata._collections",
        "importlib.metadata._functools",
        "importlib.metadata._itertools",
        "importlib.metadata._meta",
        "importlib.metadata._text",
    } <= set(runtime["direct_runtime_modules"])
    assert runtime["direct_runtime_modules"]["_struct"]["sha256"]
    assert runtime["direct_runtime_modules"]["json.encoder"]["sha256"]
    assert runtime["direct_runtime_modules"]["importlib.metadata._meta"]["sha256"]
    assert (
        runtime["consumed_runtime_exports"]["json.dumps"][
            "captured_object_identity_required"
        ]
        is True
    )
    assert runtime["consumed_runtime_exports"]["importlib.metadata.version"][
        "code_sha256"
    ]
    assert {
        "collections.namedtuple",
        "math.isfinite",
        "platform.python_build",
        "platform.python_compiler",
        "platform.python_implementation",
        "re.compile",
        "sysconfig.get_path",
    } <= set(runtime["consumed_runtime_exports"])
    assert set(runtime["consumed_generator_callable_aliases"]) == {
        "collections.Counter",
        "collections.defaultdict",
        "collections.deque",
        "pathlib.Path",
    }
    assert (
        runtime["packing_semantics"]["struct_pack"]["captured_object_identity_required"]
        is True
    )
    assert runtime["packing_semantics"]["pack_payload"]["code_sha256"]
    assert runtime["tokenizers"]["distribution_version"]
    assert runtime["tokenizers"]["native_module"]["sha256"]
    assert set(runtime["tokenizers"]["consumed_method_descriptors"]) == {
        "decode",
        "encode",
        "get_vocab_size",
        "token_to_id",
    }
    assert runtime["tokenizers"]["encoding_ids_descriptor"]["name"] == "ids"
    assert runtime["reviewed_callables"][
        "pipeline.generate_digitwise_recurrent_v1.rows_from_episode"
    ]["code_sha256"]
    consumed_rows = runtime["consumed_reviewed_callables"][
        "pipeline.generate_digitwise_recurrent_v1.rows_from_episode"
    ]
    assert consumed_rows["code_sha256"]
    assert consumed_rows["private_frozen_globals"] is True
    mutation_guard = runtime["callable_mutation_guards"]["reviewed_filesystem"]
    assert mutation_guard["append_only_runtime_hook"] is True
    assert mutation_guard["self_tested"] is True
    mutation_boundary = runtime["runtime_mutation_boundary"]
    assert mutation_boundary["protected_global_assignment_rejected"] is True
    assert mutation_boundary["all_existing_generator_globals_protected"] is True
    assert mutation_boundary["production_class_descriptor_assignment_rejected"] is True
    assert "_BOUND_OS_FSYNC" in mutation_boundary["protected_global_names"]
    assert "rows_from_episode" in mutation_boundary["protected_global_names"]
    assert mutation_boundary["mutation_guard"]["self_tested"] is True
    frozen_globals = runtime["frozen_reviewed_globals"]
    assert set(frozen_globals) == {
        "digitwise_protocol.builtins",
        "digitwise_protocol.globals",
        "pipeline.generate_digitwise_recurrent_v1.builtins",
        "pipeline.generate_digitwise_recurrent_v1.globals",
    }
    assert all(
        receipt["exact_key_and_value_identity_required"] is True
        and receipt["ordinary_mutation_methods_rejected"] is True
        for receipt in frozen_globals.values()
    )
    fsync_binding = runtime["filesystem_semantics"]["callables"]["os.fsync"]
    assert fsync_binding["implementation_kind"] == "bound_native_runtime"
    assert fsync_binding["bound_owner_module"] == "posix"
    atomic_no_replace = runtime["filesystem_semantics"]["atomic_no_replace"]
    assert atomic_no_replace["captured_process_image_symbol_required"] is True
    assert atomic_no_replace["errcheck"] is None
    assert atomic_no_replace["errcheck_required_null_at_snapshot"] is True
    assert atomic_no_replace["errcheck_required_null_immediately_before_call"] is True
    descriptor_cleanup = runtime["filesystem_semantics"]["descriptor_relative_cleanup"]
    assert descriptor_cleanup["python_path_unlink_audit_window_absent"] is True
    assert descriptor_cleanup["errcheck"] is None
    assert descriptor_cleanup["errcheck_required_null_at_snapshot"] is True
    assert descriptor_cleanup["errcheck_required_null_immediately_before_call"] is True

    bad = verification_kwargs(small_bundle)
    bad["expected_external_manifest_sha256"] = "0" * 64
    with pytest.raises(dws_sc.ContractError, match="receipt mismatch"):
        dws_sc.verify_bundle(small_bundle.publication, **bad)
    bad = verification_kwargs(small_bundle)
    bad["external_manifest_path"] = tmp_path / "not-the-root.json"
    with pytest.raises(dws_sc.ContractError, match="path"):
        dws_sc.verify_bundle(small_bundle.publication, **bad)


def test_direct_cli_build_and_verify_accept_hash_authenticated_relocation(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    bindings_result = run_generator_cli("--print-bindings")
    assert bindings_result.returncode == 0, bindings_result.stderr
    bindings = json.loads(bindings_result.stdout)

    relocated = tmp_path / "relocated-inputs"
    relocated.mkdir()
    build_tokenizer = relocated / "build-tokenizer.json"
    verify_tokenizer = relocated / "verify-tokenizer.json"
    build_replication = relocated / "build-replication.jsonl"
    verify_replication = relocated / "verify-replication.jsonl"
    shutil.copyfile(small_bundle.tokenizer_path, build_tokenizer)
    shutil.copyfile(small_bundle.tokenizer_path, verify_tokenizer)
    shutil.copyfile(REPLICATION_SOURCE, build_replication)
    shutil.copyfile(REPLICATION_SOURCE, verify_replication)
    publication = tmp_path / "cli-publication"

    common = [
        "--expected-tokenizer-sha256",
        small_bundle.tokenizer_sha256,
        "--parent-checkpoint-sha256",
        PARENT_CHECKPOINT_SHA256,
        "--expected-replication-source-sha256",
        dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        "--expected-source-bindings-sha256",
        bindings["source_bindings_sha256"],
        "--expected-runtime-bindings-sha256",
        bindings["runtime_bindings_sha256"],
        "--mode",
        "test",
        "--train-per-cell",
        "8",
        "--development-per-cell",
        "1",
        "--lane-length",
        str(dws_sc.LANE_LENGTH),
    ]
    try:
        build_result = run_generator_cli(
            "--out-dir",
            str(publication),
            "--tokenizer",
            str(build_tokenizer),
            "--replication-source",
            str(build_replication),
            *common,
        )
        assert build_result.returncode == 0, build_result.stderr
        build_receipt = json.loads(build_result.stdout)
        assert build_receipt["verified"] is True
        assert build_receipt["publication_disposition"] == "created_new_publication"

        verify_result = run_generator_cli(
            "--verify",
            str(publication),
            "--external-manifest",
            build_receipt["external_manifest_path"],
            "--expected-external-manifest-sha256",
            build_receipt["external_manifest_sha256"],
            "--expected-external-manifest-bytes",
            str(build_receipt["external_manifest_bytes"]),
            "--tokenizer",
            str(verify_tokenizer),
            "--replication-source",
            str(verify_replication),
            *common,
        )
        assert verify_result.returncode == 0, verify_result.stderr
        verification = json.loads(verify_result.stdout)
        assert verification["verified"] is True
        root = read_json(publication / dws_sc.SEALED_ROOT_NAME)
        assert root["inputs"]["location_contract"]["policy"] == (
            dws_sc.INPUT_LOCATION_POLICY
        )
        assert (
            root["inputs"]["runtime_bindings"]["executing_generator"]["origin_binding"]
            == "direct_script_file_without_import_spec"
        )
    finally:
        remove_tree(publication)


def test_train_development_cross_width_replay_and_zero_overlap(
    small_bundle: BundleFixture,
) -> None:
    train = read_jsonl(small_bundle.bundle / "train_episodes.jsonl")
    development = read_jsonl(small_bundle.bundle / "development_board.jsonl")
    cross_width = read_jsonl(
        small_bundle.bundle / "cross_width_replication_board.jsonl"
    )
    assert len(train) == 2 * 8 * 8
    assert len(development) == 2 * 8 * 1
    assert len(cross_width) == 12
    assert len({dws_sc.episode_signature(row) for row in development}) == len(
        development
    )
    train_inventory = dws_sc._aggregate_episode_inventory(train)
    development_inventory = dws_sc._aggregate_episode_inventory(development)
    cross_inventory = dws_sc._aggregate_episode_inventory(cross_width)
    assert dws_sc.overlap_inventory_is_disjoint(train_inventory, development_inventory)
    assert dws_sc.overlap_inventory_is_disjoint(train_inventory, cross_inventory)
    assert Counter((row["width"], row["operation"]) for row in cross_width) == {
        (width, operation): 2 for width in (4, 6, 8) for operation in dws_sc.OPERATIONS
    }
    for row in development + cross_width:
        intervention = row["generated_history_interventions"]
        for branch_name in ("nominal", "carry_flip", "written_result_r0_flip"):
            branch = intervention[branch_name]
            assert branch["full_history_prefix"].endswith(branch["prefix_state"] + "\n")
            assert branch["fresh_latest_state_prompt"].find(branch["prefix_state"]) >= 0
            state = parse_state(branch["prefix_state"])
            replay = []
            while not state["z"]:
                state = apply_microstep(state)
                replay.append(canonical_state(state))
            assert replay == branch["expected_states"]
            assert branch["target_response"].splitlines()[:-1] == replay


def test_block_diagonal_fillers_exact_context_and_loss_isolation(
    small_bundle: BundleFixture,
) -> None:
    pack_count = len(read_jsonl(small_bundle.bundle / "train_episodes.jsonl"))
    pack_bytes = dws_sc.LANES_PER_PACK * dws_sc.LANE_LENGTH * dws_sc.PACK_ELEMENT_BYTES
    binaries = {
        arm: (small_bundle.bundle / "{}_packs.bin".format(arm)).read_bytes()
        for arm in dws_sc.DATA_ARMS
    }
    assert {len(payload) for payload in binaries.values()} == {pack_count * pack_bytes}
    global_supervised = {arm: [] for arm in dws_sc.DATA_ARMS}
    for pack_index in range(pack_count):
        lanes = {
            arm: dws_sc.unpack_pack_payload(
                payload[pack_index * pack_bytes : (pack_index + 1) * pack_bytes],
                dws_sc.LANE_LENGTH,
            )
            for arm, payload in binaries.items()
        }
        for lane_index in range(dws_sc.LANES_PER_PACK):
            reference = lanes["full_trace"][lane_index]
            for arm in dws_sc.DATA_ARMS[1:]:
                assert lanes[arm][lane_index]["token_ids"] == reference["token_ids"]
                assert (
                    lanes[arm][lane_index]["attention_mask"]
                    == reference["attention_mask"]
                )
                assert lanes[arm][lane_index]["epoch_ids"] == reference["epoch_ids"]
                assert (
                    lanes[arm][lane_index]["position_ids"] == reference["position_ids"]
                )
            assert reference["position_ids"] == list(range(dws_sc.LANE_LENGTH))
            active_counts = {
                arm: sum(lanes[arm][lane_index]["attention_mask"])
                for arm in dws_sc.DATA_ARMS
            }
            assert len(set(active_counts.values())) == 1
        supervised_lane_sets = {
            arm: {
                index for index, lane in enumerate(arm_lanes) if sum(lane["loss_mask"])
            }
            for arm, arm_lanes in lanes.items()
        }
        assert supervised_lane_sets == {
            "full_trace": {0},
            "decomposed_one_step": {2, 3, 4, 5, 6},
            "multiline_sham": {1},
        }
        for arm, arm_lanes in lanes.items():
            for lane_index, lane in enumerate(arm_lanes):
                if lane_index not in supervised_lane_sets[arm]:
                    assert sum(lane["loss_mask"]) == 0
                global_supervised[arm].extend(
                    token_id
                    for token_id, loss in zip(
                        lane["token_ids"], lane["loss_mask"], strict=True
                    )
                    if loss
                )
    assert global_supervised["full_trace"] == global_supervised["decomposed_one_step"]
    assert Counter(global_supervised["full_trace"]) == Counter(
        global_supervised["multiline_sham"]
    )
    plan = read_json(small_bundle.bundle / "training_plan.json")
    filler = plan["block_diagonal_context_filler"]
    assert filler["independent_batch_lanes"] is True
    assert filler["cross_lane_attention"] is False
    assert filler["filler_can_affect_supervised_lane"] is False
    assert filler["all_arms_position_ids_identical"] is True
    assert plan["equalization"]["position_ids"] == {
        "integer_encoding": "little-endian uint16",
        "normative_per_lane": "range(lane_length)",
        "padding_restarts_positions": False,
        "serialized": True,
    }


def test_sham_seed_schedules_decision_multiplicity_and_claim_boundary(
    small_bundle: BundleFixture,
) -> None:
    episodes = {
        row["id"]: row
        for row in read_jsonl(small_bundle.bundle / "train_episodes.jsonl")
    }
    sham_rows = read_jsonl(small_bundle.bundle / "multiline_sham_train.jsonl")
    for row in sham_rows:
        source = episodes[row["episode_id"]]
        donors = row["line_donor_episode_ids"] + [row["answer_donor_episode_id"]]
        assert len(donors) == len(set(donors)) == 5
        assert source["id"] not in donors
        assert row["sham_answer"] != row["treatment_answer"]
        lines = row["response"].splitlines()
        previous = parse_state(source["initial_state"])
        for line in lines[:4]:
            assert canonical_state(apply_microstep(previous)) != line
            previous = parse_state(line)

    schedules = read_json(small_bundle.bundle / "seed_schedules.json")["schedules"]
    assert [row["seed"] for row in schedules] == list(dws_sc.PAIRED_TRAINING_SEEDS)
    assert len({row["pack_order_sha256"] for row in schedules}) == 3
    assert (
        len(
            {tuple(row["rng_initialization"]["startup_probe_u64"]) for row in schedules}
        )
        == 3
    )
    for row in schedules:
        assert set(row["run_cell_pack_order_sha256"]) == set(dws_sc.RUN_CELLS)
        assert set(row["run_cell_pack_order_sha256"].values()) == {
            row["pack_order_sha256"]
        }
    plan = read_json(small_bundle.bundle / "training_plan.json")
    decision = plan["decision_contract"]
    assert decision["seed_rule"]["directional_success_required"].startswith("3/3")
    assert decision["seed_rule"]["pooled_favorable_selection_forbidden"] is True
    assert decision["multiplicity"]["method"] == "Holm-Bonferroni"
    assert decision["multiplicity"]["family"].endswith("15 tests")
    assert plan["optimizer_contract"] == dws_sc.OPTIMIZER_CONTRACT
    assert plan["scored_mechanism_name"].startswith("model-triggered external")
    package = plan["package_effect_only"]
    assert package["stale_source_specific_attribution"] is False
    assert "SCERT" in package["component_attribution_deferred_to"]
    assert package["autonomous_base_model_reasoning_claim"] is False
    assert plan["implementation_boundary"] == {
        "evaluator_scoring_present": False,
        "future_evaluator_rational_comparison": (
            "integer cross multiplication only; binary floating point forbidden"
        ),
        "h100_authorized": False,
        "trainer_consumption_present": False,
    }
    assert plan["metadata_serialization"]["serialized_into_binary_packs"] is False
    assert (
        plan["metadata_serialization"]["future_trainer_must_prove_metadata_exclusion"]
        is True
    )
    assert (
        "treatment_answer"
        in plan["metadata_serialization"]["audit_only_fields_include"]
    )


def test_paired_target_switch_requires_nominal_and_counterfactual_exact() -> None:
    confounded = dws_sc.score_paired_target_switch(
        nominal_output="wrong",
        nominal_target="nominal",
        counterfactual_output="counterfactual",
        counterfactual_target="counterfactual",
    )
    assert confounded["counterfactual_target_exact"] is True
    assert confounded["paired_target_switch"] is False
    valid = dws_sc.score_paired_target_switch(
        nominal_output="nominal",
        nominal_target="nominal",
        counterfactual_output="counterfactual",
        counterfactual_target="counterfactual",
    )
    assert valid["paired_target_switch"] is True


@pytest.mark.parametrize(
    "payload",
    (
        b'{"value":1e999}',
        b'{"nested":[{"value":-1e999}]}',
        b'{"nested":{"values":[0.0,1e999]}}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
    ),
)
def test_strict_json_rejects_every_non_finite_number(payload: bytes) -> None:
    with pytest.raises(dws_sc.ContractError, match="non-finite"):
        dws_sc.strict_json_loads(payload, "adversarial JSON")
    assert dws_sc.strict_json_loads(b'{"value":1e308}', "finite JSON") == {
        "value": 1e308
    }


def test_recursive_identity_comparison_rejects_equal_numeric_values_of_wrong_type() -> (
    None
):
    expected = {"outer": [{"train_per_cell": 128}]}
    actual = {"outer": [{"train_per_cell": 128.0}]}
    assert actual == expected
    assert not dws_sc._recursively_type_strict_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", float(dws_sc.GENERATION_SEED)),
        ("seed", True),
        ("train_per_cell", float(dws_sc.TRAIN_PER_CELL)),
        ("train_per_cell", True),
        ("development_per_cell", float(dws_sc.DEVELOPMENT_PER_CELL)),
        ("development_per_cell", True),
        ("lane_length", float(dws_sc.LANE_LENGTH)),
        ("lane_length", True),
    ),
)
def test_production_generation_counts_require_exact_int_types(
    field: str, value: object
) -> None:
    arguments = {
        "mode": "production",
        "seed": dws_sc.GENERATION_SEED,
        "train_per_cell": dws_sc.TRAIN_PER_CELL,
        "development_per_cell": dws_sc.DEVELOPMENT_PER_CELL,
        "lane_length": dws_sc.LANE_LENGTH,
        "expected_tokenizer_sha256": dws_sc.KNOWN_TOKENIZER_SHA256,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "expected_replication_source_sha256": (dws_sc.KNOWN_REPLICATION_SOURCE_SHA256),
        "expected_source_bindings_sha256": "b" * 64,
        "expected_runtime_bindings_sha256": "c" * 64,
    }
    arguments[field] = value
    with pytest.raises(dws_sc.ContractError, match="must be an exact integer"):
        dws_sc._validate_generation_contract(**arguments)


def test_production_count_iterative_matching_handles_old_recursion_depth() -> None:
    source_ids = ["donor-{:04d}".format(index) for index in range(2_048)]
    candidates = {
        source_ids[index]: [source_ids[index], source_ids[index + 1]]
        for index in range(len(source_ids) - 1)
    }
    candidates[source_ids[-1]] = [source_ids[0]]
    started = time.monotonic()
    matching = dws_sc._perfect_matching(
        source_ids, candidates, "production-depth-regression"
    )
    elapsed = time.monotonic() - started

    assert matching[source_ids[-1]] == source_ids[0]
    assert all(
        matching[source_ids[index]] == source_ids[index + 1]
        for index in range(len(source_ids) - 1)
    )
    assert elapsed < 5.0


@pytest.mark.parametrize(
    "mutation",
    ("plan_authorization", "checkpoint", "source_binding", "promotion"),
)
def test_self_rehashed_plan_source_checkpoint_authorization_rejected(
    small_bundle: BundleFixture, tmp_path: Path, mutation: str
) -> None:
    publication = mutable_copy(
        small_bundle.publication, tmp_path / "self-rehashed-{}".format(mutation)
    )
    try:
        artifact_name = None
        root_path = publication / dws_sc.SEALED_ROOT_NAME
        root = read_json(root_path)
        if mutation == "plan_authorization":
            artifact_name = "training_plan.json"
            plan_path = publication / dws_sc.BUNDLE_DIRECTORY_NAME / artifact_name
            plan = read_json(plan_path)
            plan["authorization"]["training"] = True
            plan_path.write_bytes(dws_sc.pretty_json_bytes(plan))
        elif mutation == "checkpoint":
            root["inputs"]["parent_checkpoint_sha256"] = "b" * 64
            root_path.write_bytes(dws_sc.pretty_json_bytes(root))
        elif mutation == "source_binding":
            root["inputs"]["source_bindings_sha256"] = "b" * 64
            root_path.write_bytes(dws_sc.pretty_json_bytes(root))
        else:
            root["authorization"]["promotion"] = True
            root_path.write_bytes(dws_sc.pretty_json_bytes(root))
        receipt = reseal_artifact(publication, artifact_name)
        with pytest.raises(dws_sc.ContractError):
            dws_sc.verify_bundle(
                publication,
                **verification_kwargs(small_bundle, publication, receipt),
            )
    finally:
        remove_tree(publication)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "development_board.jsonl",
        "cross_width_replication_board.jsonl",
        "seed_schedules.json",
    ),
)
def test_collapsed_boards_and_repeated_seed_schedule_rejected_after_rehash(
    small_bundle: BundleFixture, tmp_path: Path, artifact_name: str
) -> None:
    publication = mutable_copy(
        small_bundle.publication, tmp_path / "collapsed-{}".format(artifact_name)
    )
    try:
        artifact = publication / dws_sc.BUNDLE_DIRECTORY_NAME / artifact_name
        if artifact_name.endswith(".jsonl"):
            rows = read_jsonl(artifact)
            rows[1] = dict(rows[0])
            artifact.write_bytes(dws_sc.jsonl_bytes(rows))
        else:
            schedules = read_json(artifact)
            schedules["schedules"][1] = dict(schedules["schedules"][0])
            artifact.write_bytes(dws_sc.pretty_json_bytes(schedules))
        receipt = reseal_artifact(publication, artifact_name)
        with pytest.raises(dws_sc.ContractError, match="semantic bundle replay"):
            dws_sc.verify_bundle(
                publication,
                **verification_kwargs(small_bundle, publication, receipt),
            )
    finally:
        remove_tree(publication)


def test_rehashed_pack_tamper_and_hardlink_rejected(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = mutable_copy(small_bundle.publication, tmp_path / "pack-tamper")
    try:
        artifact_name = "full_trace_packs.bin"
        artifact = publication / dws_sc.BUNDLE_DIRECTORY_NAME / artifact_name
        payload = bytearray(artifact.read_bytes())
        payload[-1] ^= 1
        artifact.write_bytes(payload)
        receipt = reseal_artifact(publication, artifact_name)
        with pytest.raises(dws_sc.ContractError, match="semantic bundle replay"):
            dws_sc.verify_bundle(
                publication,
                **verification_kwargs(small_bundle, publication, receipt),
            )
    finally:
        remove_tree(publication)

    alias = tmp_path / "sealed-root-hardlink"
    os.link(small_bundle.publication / dws_sc.SEALED_ROOT_NAME, alias)
    try:
        with pytest.raises(dws_sc.ContractError, match="hard link"):
            dws_sc.verify_bundle(
                small_bundle.publication, **verification_kwargs(small_bundle)
            )
    finally:
        alias.unlink()


def test_final_revalidation_rejects_byte_identical_artifact_replacement(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    artifact_name = dws_sc.ARTIFACT_NAMES[0]
    artifact = small_bundle.bundle / artifact_name
    aside = tmp_path / (artifact_name + ".verified-aside")
    expected_payload = artifact.read_bytes()
    identities: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def replace_after_semantic_replay() -> None:
        identities.append(
            replace_artifact_with_byte_identical_inode(
                small_bundle.publication, artifact_name, aside
            )
        )

    phase = install_runtime_phase_action(
        small_bundle.publication,
        "after semantic replay before sealed publication revalidation",
        replace_after_semantic_replay,
    )
    try:
        with pytest.raises(
            dws_sc.ContractError, match="path is not the held regular file descriptor"
        ):
            dws_sc.verify_bundle(
                small_bundle.publication, **verification_kwargs(small_bundle)
            )
        assert phase["calls"] == 1
        assert identities and identities[0][0] != identities[0][1]
        assert artifact.read_bytes() == expected_payload
    finally:
        restore_replaced_artifact(small_bundle.publication, artifact_name, aside)


def test_final_revalidation_rejects_byte_identical_publication_directory_replacement(
    small_bundle: BundleFixture,
) -> None:
    publication = small_bundle.publication
    aside = publication.parent / "publication-held-aside"
    original_identity = (publication.stat().st_dev, publication.stat().st_ino)
    replacement_identity: list[tuple[int, int]] = []

    def replace_publication_after_semantic_replay() -> None:
        publication.rename(aside)
        shutil.copytree(aside, publication)
        metadata = publication.stat()
        replacement_identity.append((metadata.st_dev, metadata.st_ino))

    phase = install_runtime_phase_action(
        publication,
        "after semantic replay before sealed publication revalidation",
        replace_publication_after_semantic_replay,
    )
    try:
        with pytest.raises(
            dws_sc.ContractError, match="publication directory path identity changed"
        ):
            dws_sc.verify_bundle(publication, **verification_kwargs(small_bundle))
        assert phase["calls"] == 1
        assert replacement_identity and replacement_identity[0] != original_identity
        assert external_receipt(publication) == {
            "bytes": small_bundle.receipt["external_manifest_bytes"],
            "sha256": small_bundle.receipt["external_manifest_sha256"],
        }
    finally:
        remove_tree(publication)
        if aside.exists():
            aside.rename(publication)


def test_restart_recovery_rejects_replaced_artifact_then_replays_restored_inode(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    artifact_name = dws_sc.ARTIFACT_NAMES[0]
    aside = tmp_path / (artifact_name + ".recovery-aside")
    identities: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def replace_during_recovery_replay() -> None:
        identities.append(
            replace_artifact_with_byte_identical_inode(
                small_bundle.publication, artifact_name, aside
            )
        )

    phase = install_runtime_phase_action(
        small_bundle.publication,
        "after semantic replay before sealed publication revalidation",
        replace_during_recovery_replay,
    )
    try:
        with pytest.raises(
            dws_sc.ContractError, match="path is not the held regular file descriptor"
        ):
            dws_sc.build_bundle(
                out_dir=small_bundle.publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=(
                    dws_sc.KNOWN_REPLICATION_SOURCE_SHA256
                ),
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert phase["calls"] == 1
        assert identities and identities[0][0] != identities[0][1]
    finally:
        restore_replaced_artifact(small_bundle.publication, artifact_name, aside)

    recovered = dws_sc.build_bundle(
        out_dir=small_bundle.publication,
        tokenizer_path=small_bundle.tokenizer_path,
        expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        replication_source=REPLICATION_SOURCE,
        expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
        expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
        mode="test",
        train_per_cell=8,
        development_per_cell=1,
    )
    assert recovered["verified"] is True
    assert recovered["publication_disposition"] == (
        "recovered_existing_exact_publication"
    )


def test_sealed_write_keeps_created_descriptor_through_path_validation(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "sealed-descriptor-race"
    directory.mkdir()
    destination = directory / "payload.bin"
    aside = directory / "payload.bin.created-aside"
    payload = b"descriptor-bound payload\n"
    foreign_payload = b"foreign pathname replacement\n"
    state = {"armed": 1, "calls": 0}

    def replace_path_before_fchmod(event: str, arguments: tuple[object, ...]) -> None:
        if state["armed"] and event == "os.chmod":
            state["armed"] = 0
            state["calls"] += 1
            destination.rename(aside)
            destination.write_bytes(foreign_payload)
            destination.chmod(0o444)

    sys.addaudithook(replace_path_before_fchmod)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(
            dws_sc.ContractError, match="path is not the held regular file descriptor"
        ):
            dws_sc._write_sealed_file_at(directory_fd, destination.name, payload)
    finally:
        os.close(directory_fd)
    assert state == {"armed": 0, "calls": 1}
    assert aside.read_bytes() == payload
    assert destination.read_bytes() == foreign_payload


def test_cleanup_holds_validated_member_descriptor_through_unlink(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    parent = tmp_path / "cleanup-descriptor-parent"
    parent.mkdir()
    publication = parent / "publication"
    stage = dws_sc._partial_stage_path(publication)
    stage.mkdir(mode=0o700)
    pinned_parent = dws_sc._pin_publication_parent(parent, create=False)
    staging_identity = dws_sc._staging_identity(
        out_dir=publication,
        pinned_parent=pinned_parent,
        expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
        expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
        mode="test",
        seed=dws_sc.GENERATION_SEED,
        train_per_cell=8,
        development_per_cell=1,
        lane_length=dws_sc.LANE_LENGTH,
    )
    marker = stage / dws_sc.SEALED_ROOT_NAME
    marker_aside = stage / "sealed_manifest.created-aside"
    owner_payload = dws_sc.pretty_json_bytes(
        {"schema": dws_sc.STAGING_OWNER_SCHEMA, "identity": staging_identity}
    )
    stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    try:
        dws_sc._write_sealed_file_at(stage_fd, dws_sc.SEALED_ROOT_NAME, owner_payload)
    finally:
        os.close(stage_fd)
    marker_identity = (marker.stat().st_dev, marker.stat().st_ino)
    foreign_payload = b"foreign cleanup replacement\n"
    state = {"armed": 1, "calls": 0}

    def replace_marker_during_cleanup_chmod(
        event: str, arguments: tuple[object, ...]
    ) -> None:
        if not state["armed"] or event != "os.chmod" or not arguments:
            return
        target = arguments[0]
        if type(target) is not int:
            return
        try:
            target_metadata = os.fstat(target)
        except OSError:
            return
        if (target_metadata.st_dev, target_metadata.st_ino) != marker_identity:
            return
        state["armed"] = 0
        state["calls"] += 1
        marker.rename(marker_aside)
        marker.write_bytes(foreign_payload)
        marker.chmod(0o444)

    sys.addaudithook(replace_marker_during_cleanup_chmod)
    try:
        with pytest.raises(
            dws_sc.ContractError, match="path is not the held regular file descriptor"
        ):
            dws_sc._remove_protocol_staging_tree(
                pinned_parent, stage.name, staging_identity
            )
        assert state == {"armed": 0, "calls": 1}
        assert marker.read_bytes() == foreign_payload
        assert marker_aside.read_bytes() == owner_payload
    finally:
        pinned_parent.close()
        remove_tree(stage)


def test_unlinkat_errcheck_install_use_remove_is_rejected_before_native_cleanup(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "unlinkat-errcheck"
    directory.mkdir()
    member = directory / "member.bin"
    payload = b"native cleanup errcheck guard\n"
    member.write_bytes(payload)
    member.chmod(0o444)
    member_metadata = member.stat()
    member_identity = (member_metadata.st_dev, member_metadata.st_ino)
    state = {"installs": 0, "uses": 0, "restores": 0, "armed": 1}

    def malicious_errcheck(result: int, function: object, _arguments: object) -> int:
        state["uses"] += 1
        delattr(function, "errcheck")
        return result

    def install_errcheck_during_cleanup(
        event: str, arguments: tuple[object, ...]
    ) -> None:
        if not state["armed"] or event != "os.chmod" or not arguments:
            return
        target = arguments[0]
        if type(target) is not int:
            return
        try:
            metadata = os.fstat(target)
        except OSError:
            return
        if (metadata.st_dev, metadata.st_ino) != member_identity:
            return
        state["armed"] = 0
        state["installs"] += 1
        dws_sc._BOUND_UNLINKAT.errcheck = malicious_errcheck

    assert dws_sc._BOUND_UNLINKAT.errcheck is None
    sys.addaudithook(install_errcheck_during_cleanup)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    member_fd = os.open(member, os.O_RDONLY)
    try:
        with pytest.raises(
            dws_sc.ContractError,
            match="descriptor-relative cleanup errcheck must remain null",
        ):
            dws_sc._remove_held_regular_file_at(
                directory_fd, member.name, member_fd, "errcheck cleanup member"
            )
        assert member.read_bytes() == payload
    finally:
        if dws_sc._BOUND_UNLINKAT.errcheck is not None:
            del dws_sc._BOUND_UNLINKAT.errcheck
            state["restores"] += 1
        os.close(member_fd)
        os.close(directory_fd)
    assert state == {"installs": 1, "uses": 0, "restores": 1, "armed": 0}
    assert (
        dws_sc.runtime_bindings()["filesystem_semantics"][
            "descriptor_relative_cleanup"
        ]["errcheck"]
        is None
    )


def test_exact_existing_publication_is_replayed_without_overwrite_and_failure_cleans(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    recovered = dws_sc.build_bundle(
        out_dir=small_bundle.publication,
        tokenizer_path=small_bundle.tokenizer_path,
        expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
        parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        replication_source=REPLICATION_SOURCE,
        expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
        expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
        mode="test",
        train_per_cell=8,
        development_per_cell=1,
    )
    assert recovered["verified"] is True
    assert recovered["publication_disposition"] == (
        "recovered_existing_exact_publication"
    )

    interrupted = tmp_path / "interrupted-publication"

    def fail_publish() -> None:
        raise dws_sc.ContractError("injected atomic publication failure")

    phase = install_runtime_phase_action(
        interrupted, "during publication after runtime validation", fail_publish
    )
    with pytest.raises(dws_sc.ContractError, match="injected"):
        dws_sc.build_bundle(
            out_dir=interrupted,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
    assert not interrupted.exists()
    assert not dws_sc._partial_stage_path(interrupted).exists()
    assert phase["calls"] == 1


def test_noop_os_fsync_substitution_is_fail_closed_before_publication(
    small_bundle: BundleFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = tmp_path / "noop-fsync-publication"
    intercepted = 0

    def noop_fsync(_descriptor: int) -> None:
        nonlocal intercepted
        intercepted += 1

    assert dws_sc.runtime_bindings_sha256() == small_bundle.runtime_bindings_sha256
    monkeypatch.setattr(dws_sc.os, "fsync", noop_fsync)
    with pytest.raises(
        dws_sc.ContractError, match="filesystem runtime callable changed: os.fsync"
    ):
        dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
    assert intercepted == 0
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_transient_builtin_substitution_is_not_consumed_and_fails_closed(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "builtin-substitution"
    original_sorted = builtins.sorted
    intercepted = 0

    def malicious_sorted(*_args: object, **_kwargs: object) -> list[object]:
        nonlocal intercepted
        intercepted += 1
        raise RuntimeError("mutable public builtin was consumed")

    def substitute_builtin() -> None:
        builtins.sorted = malicious_sorted

    phase = install_runtime_phase_action(
        publication,
        "after initial runtime validation",
        substitute_builtin,
    )
    try:
        with pytest.raises(dws_sc.ContractError, match="generator builtins changed"):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
    finally:
        builtins.sorted = original_sorted
    assert phase["calls"] == 1
    assert intercepted == 0
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_rows_from_episode_code_substitution_is_fail_closed(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "rows-code-substitution"
    original_code = dws_sc.rows_from_episode.__code__
    replacement_code = (lambda _episode: []).__code__
    assert dws_sc.runtime_bindings_sha256() == small_bundle.runtime_bindings_sha256

    with pytest.raises(
        dws_sc.ContractError,
        match="protected callable implementation mutation rejected",
    ):
        dws_sc.rows_from_episode.__code__ = replacement_code
    assert dws_sc.rows_from_episode.__code__ is original_code
    assert dws_sc.runtime_bindings_sha256() == small_bundle.runtime_bindings_sha256
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_transient_reviewed_code_swap_after_validation_cannot_be_consumed_or_restored(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "transient-reviewed-code-swap"
    original_code = dws_sc.rows_from_episode.__code__
    attempted = False
    restored = False

    def malicious_rows(_episode: dict) -> list[dict]:
        raise RuntimeError("transient malicious rows implementation was consumed")

    def transient_swap_after_validation() -> None:
        nonlocal attempted, restored
        attempted = True
        try:
            dws_sc.rows_from_episode.__code__ = malicious_rows.__code__
        finally:
            if dws_sc.rows_from_episode.__code__ is not original_code:
                dws_sc.rows_from_episode.__code__ = original_code
                restored = True

    phase = install_runtime_phase_action(
        publication,
        "after initial runtime validation",
        transient_swap_after_validation,
    )
    with pytest.raises(
        dws_sc.ContractError,
        match="protected callable implementation mutation rejected",
    ):
        dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
    assert attempted is True
    assert restored is False
    assert phase["calls"] == 1
    assert dws_sc.rows_from_episode.__code__ is original_code
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_transient_reviewed_global_swap_after_validation_is_rejected(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "transient-reviewed-global-swap"
    reviewed_globals = dws_sc.rows_from_episode.__globals__
    original_canonical_state = reviewed_globals["canonical_state"]
    attempted = False
    restored = False

    def malicious_canonical_state(_state: object) -> str:
        raise RuntimeError("transient malicious reviewed helper was consumed")

    def transient_swap_after_validation() -> None:
        nonlocal attempted, restored
        attempted = True
        try:
            reviewed_globals["canonical_state"] = malicious_canonical_state
        finally:
            if reviewed_globals["canonical_state"] is not original_canonical_state:
                reviewed_globals["canonical_state"] = original_canonical_state
                restored = True

    phase = install_runtime_phase_action(
        publication,
        "after initial runtime validation",
        transient_swap_after_validation,
    )
    with pytest.raises(
        dws_sc.ContractError,
        match="frozen reviewed globals mutation rejected",
    ):
        dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
    assert attempted is True
    assert restored is False
    assert phase["calls"] == 1
    assert reviewed_globals["canonical_state"] is original_canonical_state
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_fsync_substitution_during_atomic_publication_is_detected_and_cleaned(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "during-publication-fsync-substitution"
    original_fsync = dws_sc.os.fsync
    substituted = False
    intercepted = 0

    def noop_fsync(_descriptor: int) -> None:
        nonlocal intercepted
        intercepted += 1

    def substitute_during_publication() -> None:
        nonlocal substituted
        dws_sc.os.fsync = noop_fsync
        substituted = True

    phase = install_runtime_phase_action(
        publication,
        "during publication after runtime validation",
        substitute_during_publication,
    )
    try:
        with pytest.raises(
            dws_sc.ContractError, match="filesystem runtime callable changed: os.fsync"
        ):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
    finally:
        dws_sc.os.fsync = original_fsync
    assert substituted is True
    assert phase["calls"] == 1
    assert intercepted == 0
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_captured_fsync_substitution_during_publication_is_rejected_and_cleaned(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "captured-fsync-substitution"
    attempted = False
    intercepted = 0

    def noop_fsync(_descriptor: int) -> None:
        nonlocal intercepted
        intercepted += 1

    def substitute_captured_binding() -> None:
        nonlocal attempted
        attempted = True
        dws_sc._BOUND_OS_FSYNC = noop_fsync

    phase = install_runtime_phase_action(
        publication,
        "during publication after runtime validation",
        substitute_captured_binding,
    )
    with pytest.raises(
        dws_sc.ContractError,
        match="protected generator runtime binding mutation rejected: _BOUND_OS_FSYNC",
    ):
        dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
    assert attempted is True
    assert phase["calls"] == 1
    assert intercepted == 0
    assert dws_sc._BOUND_OS_FSYNC is dws_sc.os.fsync
    assert not publication.exists()
    assert not dws_sc._partial_stage_path(publication).exists()


def test_atomic_rename_errcheck_install_use_remove_is_rejected_before_native_call(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "atomic-rename-errcheck"
    state = {"installs": 0, "uses": 0, "restores": 0}

    def malicious_errcheck(_result: int, function: object, _arguments: object) -> int:
        state["uses"] += 1
        delattr(function, "errcheck")
        return -1

    def install_errcheck_before_rename() -> None:
        state["installs"] += 1
        dws_sc._BOUND_ATOMIC_RENAME.errcheck = malicious_errcheck

    assert dws_sc._BOUND_ATOMIC_RENAME.errcheck is None
    phase = install_runtime_phase_action(
        publication,
        "immediately before atomic no-replace rename",
        install_errcheck_before_rename,
    )
    try:
        with pytest.raises(
            dws_sc.ContractError, match="atomic no-replace errcheck must remain null"
        ):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=(
                    dws_sc.KNOWN_REPLICATION_SOURCE_SHA256
                ),
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
    finally:
        if dws_sc._BOUND_ATOMIC_RENAME.errcheck is not None:
            del dws_sc._BOUND_ATOMIC_RENAME.errcheck
            state["restores"] += 1
        remove_tree(publication)
        remove_tree(dws_sc._partial_stage_path(publication))
    assert phase["calls"] == 1
    assert state == {"installs": 1, "uses": 0, "restores": 1}
    assert (
        dws_sc.runtime_bindings()["filesystem_semantics"]["atomic_no_replace"][
            "errcheck"
        ]
        is None
    )


def test_process_death_staging_recovers_without_partial_authority(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "process-death-publication"
    stage = dws_sc._partial_stage_path(publication)
    script = """
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "train"))
from pipeline import generate_dws_single_completion_v1 as dws_sc

def die_before_rename(event, arguments):
    if (
        event == dws_sc._RUNTIME_PHASE_AUDIT_EVENT
        and arguments
        == ("during publication after runtime validation", str(Path(sys.argv[2])))
    ):
        os._exit(73)

sys.addaudithook(die_before_rename)
dws_sc.build_bundle(
    out_dir=Path(sys.argv[2]),
    tokenizer_path=Path(sys.argv[3]),
    expected_tokenizer_sha256=sys.argv[4],
    parent_checkpoint_sha256=sys.argv[5],
    replication_source=Path(sys.argv[6]),
    expected_replication_source_sha256=sys.argv[7],
    expected_source_bindings_sha256=sys.argv[8],
    expected_runtime_bindings_sha256=sys.argv[9],
    mode="test",
    train_per_cell=8,
    development_per_cell=1,
)
"""
    result = run_isolated_python(
        script,
        str(ROOT),
        str(publication),
        str(small_bundle.tokenizer_path),
        small_bundle.tokenizer_sha256,
        PARENT_CHECKPOINT_SHA256,
        str(REPLICATION_SOURCE),
        dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        small_bundle.source_bindings_sha256,
        small_bundle.runtime_bindings_sha256,
    )
    assert result.returncode == 73
    assert not publication.exists()
    assert stage.is_dir()

    stage_receipt = external_receipt(stage)
    with pytest.raises(dws_sc.ContractError, match="partial staging tree"):
        dws_sc.verify_bundle(
            stage,
            **verification_kwargs(
                small_bundle,
                publication=stage,
                receipt=stage_receipt,
            ),
        )

    try:
        receipt = dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
        assert receipt["verified"] is True
        assert publication.is_dir()
        assert not stage.exists()
    finally:
        remove_tree(publication)


def test_process_death_after_destination_rename_recovers_exact_publication(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "post-rename-process-death-publication"
    stage = dws_sc._partial_stage_path(publication)
    script = """
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "train"))
from pipeline import generate_dws_single_completion_v1 as dws_sc

def die_after_rename(event, arguments):
    if (
        event == dws_sc._RUNTIME_PHASE_AUDIT_EVENT
        and arguments
        == ("after destination rename before external receipt", str(Path(sys.argv[2])))
    ):
        os._exit(74)

sys.addaudithook(die_after_rename)
dws_sc.build_bundle(
    out_dir=Path(sys.argv[2]),
    tokenizer_path=Path(sys.argv[3]),
    expected_tokenizer_sha256=sys.argv[4],
    parent_checkpoint_sha256=sys.argv[5],
    replication_source=Path(sys.argv[6]),
    expected_replication_source_sha256=sys.argv[7],
    expected_source_bindings_sha256=sys.argv[8],
    expected_runtime_bindings_sha256=sys.argv[9],
    mode="test",
    train_per_cell=8,
    development_per_cell=1,
)
"""
    result = run_isolated_python(
        script,
        str(ROOT),
        str(publication),
        str(small_bundle.tokenizer_path),
        small_bundle.tokenizer_sha256,
        PARENT_CHECKPOINT_SHA256,
        str(REPLICATION_SOURCE),
        dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        small_bundle.source_bindings_sha256,
        small_bundle.runtime_bindings_sha256,
    )
    assert result.returncode == 74
    assert publication.is_dir()
    assert not stage.exists()

    try:
        receipt = dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
        assert receipt["verified"] is True
        assert receipt["publication_disposition"] == (
            "recovered_existing_exact_publication"
        )
        verification = dws_sc.verify_bundle(
            publication,
            **verification_kwargs(
                small_bundle,
                publication=publication,
                receipt=receipt,
            ),
        )
        assert verification["verified"] is True
        assert not stage.exists()
    finally:
        remove_tree(publication)


def test_ordinary_post_rename_failure_retains_destination_for_restart_replay(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "post-rename-ordinary-failure-publication"
    stage = dws_sc._partial_stage_path(publication)

    def fail_after_rename() -> None:
        raise RuntimeError("injected post-rename verification failure")

    phase = install_runtime_phase_action(
        publication,
        "after destination rename before external receipt",
        fail_after_rename,
    )
    try:
        with pytest.raises(RuntimeError, match="post-rename verification failure"):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=(
                    dws_sc.KNOWN_REPLICATION_SOURCE_SHA256
                ),
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert phase["calls"] == 1
        assert publication.is_dir()
        assert not stage.exists()

        receipt = dws_sc.build_bundle(
            out_dir=publication,
            tokenizer_path=small_bundle.tokenizer_path,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            replication_source=REPLICATION_SOURCE,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            train_per_cell=8,
            development_per_cell=1,
        )
        assert receipt["verified"] is True
        assert receipt["publication_disposition"] == (
            "recovered_existing_exact_publication"
        )
    finally:
        remove_tree(publication)


def test_atomic_no_overwrite_foreign_destination_created_immediately_before_rename(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "last-moment-foreign-destination"
    stage = dws_sc._partial_stage_path(publication)
    foreign_member = publication / "foreign.bin"
    foreign_payload = b"foreign destination must survive byte-for-byte\x00\xff"

    def create_foreign_destination() -> None:
        publication.mkdir(mode=0o700)
        foreign_member.write_bytes(foreign_payload)

    phase = install_runtime_phase_action(
        publication,
        "immediately before atomic no-replace rename",
        create_foreign_destination,
    )
    try:
        with pytest.raises(
            dws_sc.ContractError, match="refusing to overwrite existing publication"
        ):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=(
                    dws_sc.KNOWN_REPLICATION_SOURCE_SHA256
                ),
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert phase["calls"] == 1
        assert sorted(path.name for path in publication.iterdir()) == ["foreign.bin"]
        assert foreign_member.read_bytes() == foreign_payload
        assert not stage.exists()
    finally:
        remove_tree(publication)
        remove_tree(stage)


def test_concurrent_process_publication_has_one_winner_and_preserves_foreign_stage(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "concurrent-publication"
    stage = dws_sc._partial_stage_path(publication)
    ready = tmp_path / "winner-renamed.ready"
    release = tmp_path / "winner-renamed.release"
    foreign_member = stage / "foreign.txt"
    script = """
import json
from pathlib import Path
import sys
import time

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "train"))
from pipeline import generate_dws_single_completion_v1 as dws_sc

role = sys.argv[10]
ready = Path(sys.argv[11])
release = Path(sys.argv[12])
if role == "winner":
    def hold_lock_after_rename(event, arguments):
        if (
            event == dws_sc._RUNTIME_PHASE_AUDIT_EVENT
            and arguments
            == (
                "after destination rename before external receipt",
                str(Path(sys.argv[2])),
            )
        ):
            ready.write_bytes(b"ready\\n")
            deadline = time.monotonic() + 120.0
            while not release.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting to release publication lock")
                time.sleep(0.01)

    sys.addaudithook(hold_lock_after_rename)

receipt = dws_sc.build_bundle(
    out_dir=Path(sys.argv[2]),
    tokenizer_path=Path(sys.argv[3]),
    expected_tokenizer_sha256=sys.argv[4],
    parent_checkpoint_sha256=sys.argv[5],
    replication_source=Path(sys.argv[6]),
    expected_replication_source_sha256=sys.argv[7],
    expected_source_bindings_sha256=sys.argv[8],
    expected_runtime_bindings_sha256=sys.argv[9],
    mode="test",
    train_per_cell=8,
    development_per_cell=1,
)
print(json.dumps(receipt, sort_keys=True))
"""
    process_arguments = (
        str(ROOT),
        str(publication),
        str(small_bundle.tokenizer_path),
        small_bundle.tokenizer_sha256,
        PARENT_CHECKPOINT_SHA256,
        str(REPLICATION_SOURCE),
        dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
        small_bundle.source_bindings_sha256,
        small_bundle.runtime_bindings_sha256,
    )
    winner = subprocess.Popen(
        [
            *ISOLATED_PYTHON,
            "-W",
            "error",
            "-c",
            script,
            *process_arguments,
            "winner",
            str(ready),
            str(release),
        ],
        cwd=ROOT,
        env={},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120.0
        while not ready.exists() and winner.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        if not ready.is_file():
            winner_stdout, winner_stderr = winner.communicate(timeout=30)
            pytest.fail(
                "winner did not reach the post-rename lock window\n"
                + winner_stdout
                + winner_stderr
            )
        stage.mkdir(mode=0o700)
        foreign_member.write_bytes(b"foreign concurrent staging tree\n")

        contender = subprocess.run(
            [
                *ISOLATED_PYTHON,
                "-W",
                "error",
                "-c",
                script,
                *process_arguments,
                "contender",
                str(ready),
                str(release),
            ],
            cwd=ROOT,
            env={},
            capture_output=True,
            text=True,
            check=False,
        )
        assert contender.returncode != 0
        assert "publication target belongs to a live invocation" in contender.stderr
        assert foreign_member.read_bytes() == b"foreign concurrent staging tree\n"

        release.write_bytes(b"release\n")
        winner_stdout, winner_stderr = winner.communicate(timeout=120)
        assert winner.returncode == 0, winner_stderr
        winner_receipt = json.loads(winner_stdout)
        assert winner_receipt["verified"] is True
        assert [winner.returncode, contender.returncode].count(0) == 1
        verification = dws_sc.verify_bundle(
            publication,
            **verification_kwargs(
                small_bundle,
                publication=publication,
                receipt=winner_receipt,
            ),
        )
        assert verification["verified"] is True
        assert foreign_member.read_bytes() == b"foreign concurrent staging tree\n"
    finally:
        if not release.exists():
            release.write_bytes(b"release\n")
        if winner.poll() is None:
            winner.kill()
        winner.communicate(timeout=30)
        remove_tree(publication)
        remove_tree(stage)


def test_foreign_partial_identity_is_fail_closed_and_not_removed(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "foreign-stage-publication"
    stage = dws_sc._partial_stage_path(publication)
    stage.mkdir(mode=0o700)
    marker = stage / dws_sc.SEALED_ROOT_NAME
    marker.write_bytes(
        dws_sc.pretty_json_bytes(
            {
                "schema": dws_sc.STAGING_OWNER_SCHEMA,
                "identity": {"protocol": "not-this-invocation"},
            }
        )
    )
    marker.chmod(0o444)
    try:
        with pytest.raises(dws_sc.ContractError, match="identity mismatch"):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert stage.is_dir()
        assert not publication.exists()
    finally:
        remove_tree(stage)


def test_float_colliding_foreign_marker_is_preserved(
    small_bundle: BundleFixture, tmp_path: Path
) -> None:
    publication = tmp_path / "float-collision-publication"
    stage = dws_sc._partial_stage_path(publication)
    pinned_parent = dws_sc._pin_publication_parent(publication.parent, create=False)
    try:
        identity = dws_sc._staging_identity(
            out_dir=publication,
            pinned_parent=pinned_parent,
            expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
            expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
            expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
            mode="test",
            seed=dws_sc.GENERATION_SEED,
            train_per_cell=8,
            development_per_cell=1,
            lane_length=dws_sc.LANE_LENGTH,
        )
    finally:
        pinned_parent.close()
    identity["train_per_cell"] = 8.0
    stage.mkdir(mode=0o700)
    marker = stage / dws_sc.SEALED_ROOT_NAME
    marker_payload = dws_sc.pretty_json_bytes(
        {"schema": dws_sc.STAGING_OWNER_SCHEMA, "identity": identity}
    )
    marker.write_bytes(marker_payload)
    marker.chmod(0o444)
    try:
        with pytest.raises(dws_sc.ContractError, match="identity mismatch"):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert marker.read_bytes() == marker_payload
        assert stage.is_dir()
        assert not publication.exists()
    finally:
        remove_tree(stage)


def test_retargeted_ancestor_symlink_during_publication_is_rejected_and_cleaned(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    anchor = root / "anchor"
    displaced = root / "anchor-pinned"
    replacement = root / "replacement"
    (anchor / "parent").mkdir(parents=True)
    (replacement / "parent").mkdir(parents=True)
    publication = anchor / "parent/publication"
    retargeted = False

    def retarget_during_rename() -> None:
        nonlocal retargeted
        anchor.rename(displaced)
        anchor.symlink_to(replacement, target_is_directory=True)
        retargeted = True

    phase = install_runtime_phase_action(
        publication,
        "during publication after runtime validation",
        retarget_during_rename,
    )
    try:
        with pytest.raises(dws_sc.ContractError, match="publication ancestor"):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert retargeted is True
        assert phase["calls"] == 1
        assert not dws_sc._partial_stage_path(publication).exists()
        assert not (displaced / "parent/publication").exists()
        assert not (displaced / "parent/.publication.partial").exists()
        assert not (replacement / "parent/publication").exists()
    finally:
        if anchor.is_symlink():
            anchor.unlink()
        if displaced.exists():
            displaced.rename(anchor)


def test_locked_stage_path_substitution_is_rejected_without_publication(
    small_bundle: BundleFixture,
    tmp_path: Path,
) -> None:
    publication = tmp_path / "stage-substitution-publication"
    stage = dws_sc._partial_stage_path(publication)
    aside = publication.parent / (stage.name + ".locked-aside")
    foreign_member = stage / "foreign-tree.txt"
    substituted = False

    def substitute_locked_stage() -> None:
        nonlocal substituted
        stage.rename(aside)
        stage.mkdir(mode=0o700)
        foreign_member.write_bytes(b"foreign staging tree\n")
        substituted = True

    phase = install_runtime_phase_action(
        publication,
        "during publication after runtime validation",
        substitute_locked_stage,
    )
    try:
        with pytest.raises(dws_sc.ContractError, match="source path identity changed"):
            dws_sc.build_bundle(
                out_dir=publication,
                tokenizer_path=small_bundle.tokenizer_path,
                expected_tokenizer_sha256=small_bundle.tokenizer_sha256,
                parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
                replication_source=REPLICATION_SOURCE,
                expected_replication_source_sha256=dws_sc.KNOWN_REPLICATION_SOURCE_SHA256,
                expected_source_bindings_sha256=small_bundle.source_bindings_sha256,
                expected_runtime_bindings_sha256=small_bundle.runtime_bindings_sha256,
                mode="test",
                train_per_cell=8,
                development_per_cell=1,
            )
        assert substituted is True
        assert phase["calls"] == 1
        assert not publication.exists()
        assert foreign_member.read_bytes() == b"foreign staging tree\n"
        assert (aside / dws_sc.SEALED_ROOT_NAME).is_file()
        assert not (stage / dws_sc.SEALED_ROOT_NAME).exists()
    finally:
        remove_tree(stage)
        remove_tree(aside)


def test_source_runtime_and_tokenizer_external_commitments_are_required(
    small_bundle: BundleFixture,
) -> None:
    bad = verification_kwargs(small_bundle, source_bindings_sha256="0" * 64)
    with pytest.raises(dws_sc.ContractError, match="source-binding"):
        dws_sc.verify_bundle(small_bundle.publication, **bad)
    bad = verification_kwargs(small_bundle)
    bad["expected_tokenizer_sha256"] = "0" * 64
    with pytest.raises(dws_sc.ContractError, match="tokenizer SHA-256 mismatch"):
        dws_sc.verify_bundle(small_bundle.publication, **bad)
    bad = verification_kwargs(small_bundle, runtime_bindings_sha256="0" * 64)
    with pytest.raises(dws_sc.ContractError, match="runtime-binding"):
        dws_sc.verify_bundle(small_bundle.publication, **bad)


@pytest.mark.parametrize(
    ("owner", "attribute", "label"),
    (
        (dws_sc.json, "dumps", "json.dumps"),
        (dws_sc.json, "loads", "json.loads"),
        (dws_sc.collections_module, "namedtuple", "collections.namedtuple"),
        (dws_sc._BOUND_JSON_ENCODER_MODULE, "encode_basestring_ascii", "json.encoder"),
        (dws_sc.importlib_metadata, "version", "importlib.metadata.version"),
        (dws_sc.math, "isfinite", "math.isfinite"),
        (dws_sc.platform, "python_build", "platform.python_build"),
        (dws_sc.platform, "python_compiler", "platform.python_compiler"),
        (
            dws_sc.platform,
            "python_implementation",
            "platform.python_implementation",
        ),
        (dws_sc.re, "compile", "re.compile"),
        (dws_sc.sysconfig, "get_path", "sysconfig.get_path"),
    ),
    ids=(
        "json-dumps",
        "json-loads",
        "collections-namedtuple",
        "json-encoder",
        "metadata-version",
        "math-isfinite",
        "platform-build",
        "platform-compiler",
        "platform-implementation",
        "re-compile",
        "sysconfig-get-path",
    ),
)
def test_consumed_runtime_export_substitution_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    owner: object,
    attribute: str,
    label: str,
) -> None:
    assert len(dws_sc.runtime_bindings_sha256()) == 64
    monkeypatch.setattr(owner, attribute, object())
    with pytest.raises(dws_sc.ContractError) as raised:
        dws_sc.runtime_bindings_sha256()
    assert label in str(raised.value)


@pytest.mark.parametrize(
    ("global_name", "label"),
    (
        ("Counter", "collections.Counter"),
        ("defaultdict", "collections.defaultdict"),
        ("deque", "collections.deque"),
        ("Path", "pathlib.Path"),
    ),
)
def test_consumed_generator_callable_alias_rebinding_is_rejected(
    global_name: str, label: str
) -> None:
    baseline = dws_sc.runtime_bindings_sha256()
    original = getattr(dws_sc, global_name)
    with pytest.raises(
        dws_sc.ContractError,
        match="protected generator runtime binding mutation rejected",
    ):
        setattr(dws_sc, global_name, object())
    assert getattr(dws_sc, global_name) is original
    assert label in dws_sc.runtime_bindings()["consumed_generator_callable_aliases"]
    assert dws_sc.runtime_bindings_sha256() == baseline


@pytest.mark.parametrize(
    ("owner", "attribute", "label"),
    (
        (dws_sc._BOUND_TOKENIZER_CLASS, "get_vocab_size", "get_vocab_size"),
        (dws_sc._BOUND_TOKENIZER_CLASS, "token_to_id", "token_to_id"),
        (dws_sc._BOUND_TOKENIZER_CLASS, "encode", "encode"),
        (dws_sc._BOUND_TOKENIZER_CLASS, "decode", "decode"),
        (dws_sc._BOUND_TOKENIZER_ENCODING_CLASS, "ids", "exports"),
    ),
)
def test_consumed_tokenizer_descriptor_substitution_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    owner: type,
    attribute: str,
    label: str,
) -> None:
    assert len(dws_sc.runtime_bindings_sha256()) == 64
    monkeypatch.setattr(owner, attribute, object())
    with pytest.raises(dws_sc.ContractError, match=label):
        dws_sc.runtime_bindings_sha256()


def test_random_method_descriptor_substitution_is_not_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = dws_sc._new_bound_random(123)
    intercepted = 0

    def malicious_randrange(*_arguments: object) -> int:
        nonlocal intercepted
        intercepted += 1
        return 0

    monkeypatch.setattr(dws_sc._BOUND_RANDOM_CLASS, "randrange", malicious_randrange)
    with pytest.raises(dws_sc.ContractError, match="random method changed: randrange"):
        dws_sc._call_bound_random_method(rng, "randrange", 10)
    assert intercepted == 0


def test_generator_production_class_descriptor_rebinding_is_rejected() -> None:
    original = dws_sc.FrozenTokenizer.__dict__["encode"]
    with pytest.raises(
        dws_sc.ContractError,
        match="sealed generator runtime class mutation rejected",
    ):
        dws_sc.FrozenTokenizer.encode = object()
    assert dws_sc.FrozenTokenizer.__dict__["encode"] is original


def test_executed_json_submodule_substitution_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(dws_sc.runtime_bindings_sha256()) == 64
    monkeypatch.setitem(sys.modules, "json.encoder", object())
    with pytest.raises(
        dws_sc.ContractError, match="runtime module changed: json.encoder"
    ):
        dws_sc.runtime_bindings_sha256()


def test_captured_json_callable_code_mutation_is_rejected() -> None:
    original_code = dws_sc._BOUND_JSON_DUMPS.__code__
    with pytest.raises(
        dws_sc.ContractError,
        match="protected callable implementation mutation rejected: runtime-exports-v1",
    ):
        dws_sc._BOUND_JSON_DUMPS.__code__ = (lambda _value: "{}").__code__
    assert dws_sc._BOUND_JSON_DUMPS.__code__ is original_code


@pytest.mark.parametrize(
    "module_name",
    ("digitwise_protocol", "pipeline.generate_digitwise_recurrent_v1"),
)
def test_preloaded_reviewed_modules_are_rejected(module_name: str) -> None:
    script = """
import sys
import types

root, module_name = sys.argv[1:]
sys.path.insert(0, root)
sys.path.insert(0, root + "/train")
sys.modules[module_name] = types.ModuleType(module_name)
from pipeline import generate_dws_single_completion_v1
"""
    result = run_isolated_python(script, str(ROOT), module_name)
    assert result.returncode != 0
    assert "preloaded before byte binding" in result.stderr


def test_preloaded_tokenizers_runtime_is_rejected() -> None:
    script = """
import sys
import sysconfig

root = sys.argv[1]
sys.path.insert(0, root)
sys.path.insert(0, root + "/train")
sys.path.append(sysconfig.get_path("purelib"))
import tokenizers
from pipeline import generate_dws_single_completion_v1
"""
    result = run_isolated_python(script, str(ROOT))
    assert result.returncode != 0
    assert "preloaded before semantic binding" in result.stderr


@pytest.mark.parametrize(
    "module",
    (dws_sc._BOUND_TOKENIZERS_MODULE, dws_sc._BOUND_TOKENIZERS_NATIVE_MODULE),
    ids=("package-export", "native-export"),
)
def test_tokenizer_export_substitution_is_fail_closed(
    small_bundle: BundleFixture,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    baseline_runtime_sha256 = dws_sc.runtime_bindings_sha256()
    assert len(baseline_runtime_sha256) == 64
    monkeypatch.setattr(module, "Tokenizer", object())

    with pytest.raises(dws_sc.ContractError, match="tokenizers runtime exports"):
        dws_sc.runtime_bindings_sha256()
    with pytest.raises(dws_sc.ContractError, match="tokenizers runtime exports"):
        dws_sc.FrozenTokenizer(
            small_bundle.tokenizer_path, small_bundle.tokenizer_sha256
        )


def test_random_class_substitution_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_runtime_sha256 = dws_sc.runtime_bindings_sha256()
    assert len(baseline_runtime_sha256) == 64
    monkeypatch.setattr(dws_sc.random, "Random", object())

    with pytest.raises(dws_sc.ContractError, match="random runtime exports"):
        dws_sc.runtime_bindings_sha256()
    with pytest.raises(dws_sc.ContractError, match="random runtime exports"):
        dws_sc.build_seed_schedules(["episode-0", "episode-1"])


def test_struct_pack_substitution_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(dws_sc.runtime_bindings_sha256()) == 64
    monkeypatch.setattr(dws_sc.struct, "pack", object())
    with pytest.raises(dws_sc.ContractError, match="struct runtime exports"):
        dws_sc.runtime_bindings_sha256()


def test_pack_payload_substitution_is_fail_closed() -> None:
    baseline = dws_sc.runtime_bindings_sha256()
    with pytest.raises(
        dws_sc.ContractError,
        match="protected generator runtime binding mutation rejected: _pack_payload",
    ):
        dws_sc._pack_payload = lambda _lanes, _length: b""
    assert dws_sc.runtime_bindings_sha256() == baseline
