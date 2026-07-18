"""Exhaustive CPU tests for the R12 OCSC preregistration bundle."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import errno
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest


ROOT = Path(os.path.abspath(os.fspath(Path(__file__).parent.parent)))
GENERATOR = ROOT / "pipeline" / "generate_orthogonal_carry_serializer_curriculum.py"
RUNNER = ROOT / "pipeline" / "run_orthogonal_carry_serializer_curriculum.py"
sys.path.insert(0, str(ROOT / "pipeline"))
import generate_orthogonal_carry_serializer_curriculum as ocsc  # noqa: E402
import run_orthogonal_carry_serializer_curriculum as ocsc_runner  # noqa: E402


TEST_PUBLICATION_PRIVATE_KEY_HEX = (
    "73716406cd8101f22d7890c814e328bf348f13ee7317eba26ae989e098e0034b"
)
TEST_INDEPENDENT_REVIEW_PRIVATE_KEY_HEX = (
    "6ed99f67de3e556e1965ed92caa0e0114ef3204ab9eb0442cd7aeb36a581323b"
)
TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX = "1f" * 32


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


_EXTERNAL_BOOTSTRAP_ARTIFACTS: dict[str, Path | str] | None = None


def external_bootstrap_artifacts() -> dict[str, Path | str]:
    global _EXTERNAL_BOOTSTRAP_ARTIFACTS
    if _EXTERNAL_BOOTSTRAP_ARTIFACTS is not None:
        return _EXTERNAL_BOOTSTRAP_ARTIFACTS
    temporary_parent = Path("/private/tmp" if sys.platform == "darwin" else "/tmp")
    build_root = Path(
        tempfile.mkdtemp(prefix="ocsc-external-bootstrap-", dir=temporary_parent)
    )
    bootstrap = build_root / "ocsc-external-bootstrap"
    subprocess.run(
        [
            os.environ.get("CC", "cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-x",
            "c",
            str(RUNNER),
            "-o",
            str(bootstrap),
        ],
        cwd=ROOT,
        env={"LC_ALL": "C", "PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        check=True,
    )
    ocsc._preload_consumed_distributions()
    runtime = ocsc.runtime_closure_contract()
    system_interpreter = Path(runtime["interpreter"]["resolved_path"])
    interpreter = build_root / "python"
    interpreter.write_bytes(system_interpreter.read_bytes())
    interpreter.chmod(0o555)
    held = {
        record["resolved_path"]: record["sha256"]
        for record in runtime["stdlib"]["modules"].values()
        if record.get("kind") == "file"
    }
    inventory = {}
    code_suffixes = {".dll", ".dylib", ".py", ".pyd", ".so"}
    for distribution in runtime["distributions"].values():
        for record in distribution["files"].values():
            destination = (
                held
                if Path(record["resolved_path"]).suffix.lower() in code_suffixes
                else inventory
            )
            destination[record["resolved_path"]] = record["sha256"]
    for path, record in runtime["native_images"]["files"].items():
        held[path] = record["sha256"]
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            continue
        path = Path(os.path.abspath(module_path))
        if path.suffix.lower() in code_suffixes and path.is_file():
            held[str(path)] = sha256_file(path)
    held.pop(str(system_interpreter), None)
    inventory.pop(str(system_interpreter), None)
    lines = [
        "bootstrap\t{}\t{}".format(sha256_file(bootstrap), bootstrap),
        "python\t{}\t{}".format(sha256_file(interpreter), interpreter),
        *("runtime-held\t{}\t{}".format(digest, path) for path, digest in held.items()),
        *(
            "runtime-inventory\t{}\t{}".format(digest, path)
            for path, digest in inventory.items()
            if path not in held
        ),
    ]
    manifest = build_root / "runtime.manifest"
    manifest.write_text(
        "shohin-ocsc-external-runtime-closure-v1\n" + "\n".join(sorted(lines)) + "\n",
        encoding="ascii",
    )
    manifest.chmod(0o444)
    _EXTERNAL_BOOTSTRAP_ARTIFACTS = {
        "bootstrap": bootstrap,
        "bootstrap_sha256": sha256_file(bootstrap),
        "interpreter": interpreter,
        "interpreter_sha256": sha256_file(interpreter),
        "runtime_manifest": manifest,
        "runtime_manifest_sha256": sha256_file(manifest),
        "runtime_fd_count": 3 + len(held),
    }
    return _EXTERNAL_BOOTSTRAP_ARTIFACTS


def source_bound_command(
    generator_argv: list[str],
    *,
    profile: str = "qualification",
) -> list[str]:
    external = external_bootstrap_artifacts()
    return [
        str(external["bootstrap"]),
        "--bootstrap-sha256",
        str(external["bootstrap_sha256"]),
        "--runtime-manifest",
        str(external["runtime_manifest"]),
        "--runtime-manifest-sha256",
        str(external["runtime_manifest_sha256"]),
        "--python",
        str(external["interpreter"]),
        "--runner",
        str(RUNNER),
        "--checkout-root",
        str(ROOT),
        "--runner-sha256",
        sha256_file(RUNNER),
        "--prereg-sha256",
        sha256_file(ROOT / "R12_ORTHOGONAL_CARRY_SERIALIZER_CURRICULUM_PREREG.md"),
        "--generator-sha256",
        sha256_file(GENERATOR),
        "--tests-sha256",
        sha256_file(Path(__file__)),
        "--oracle-sha256",
        ocsc.DIGITWISE_PROTOCOL_SHA256,
        "--python-sha256",
        str(external["interpreter_sha256"]),
        "--profile",
        profile,
        "--",
        *generator_argv,
    ]


def run_source_bound(
    generator_argv: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        source_bound_command(generator_argv),
        cwd=ROOT,
        env={"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=check,
    )


def make_fake_source_checkout(root: Path) -> Path:
    """Create a non-scientific source tree for bootstrap mechanics tests."""

    for relative in ocsc_runner.SOURCE_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == ocsc_runner.RUNNER_RELATIVE_PATH:
            path.write_bytes(RUNNER.read_bytes())
        elif relative == ocsc_runner.GENERATOR_RELATIVE_PATH:
            path.write_text(
                """import os
CONTEXT = _OCSC_BOOTSTRAP_EXECUTION_CONTEXT
def bootstrap_cli(argv):
    if argv == [\"--replace-tests-path\"]:
        victim = CONTEXT[\"contract\"][\"sources\"][\"pipeline/test_generate_orthogonal_carry_serializer_curriculum.py\"][\"resolved_path\"]
        replacement = victim + \".replacement\"
        with open(replacement, \"wb\") as stream:
            stream.write(b\"substituted after pin\\n\")
        os.replace(replacement, victim)
    if argv == [\"--replace-pipeline-component\"]:
        pipeline = os.path.join(CONTEXT[\"checkout_root_path\"], \"pipeline\")
        os.rename(pipeline, pipeline + \".retained\")
        os.mkdir(pipeline, 0o700)
    return {\"argv\": argv, \"source_count\": len(CONTEXT[\"source_fds\"]), \"runtime_count\": len(CONTEXT[\"runtime_fds\"])}, {\"schema\": \"fake-source-manifest\"}
def bootstrap_execution_contract(argv, required=False):
    if list(argv) != CONTEXT[\"contract\"][\"generator_argv\"] or required is not True:
        raise ValueError(\"fake execution contract mismatch\")
    return CONTEXT[\"contract\"]
def validate_source_manifest_contract(contract):
    if contract != {\"schema\": \"fake-source-manifest\"}:
        raise ValueError(\"fake source manifest mismatch\")
    return contract
""",
                encoding="ascii",
            )
        else:
            path.write_text(
                "mechanics-only fixture: {}\n".format(relative), encoding="ascii"
            )
    return root


def fake_source_bound_command(
    checkout: Path,
    generator_argv: list[str],
    *,
    runner_sha256: str | None = None,
) -> list[str]:
    runner = checkout / ocsc_runner.RUNNER_RELATIVE_PATH
    external = external_bootstrap_artifacts()
    return [
        str(external["bootstrap"]),
        "--bootstrap-sha256",
        str(external["bootstrap_sha256"]),
        "--runtime-manifest",
        str(external["runtime_manifest"]),
        "--runtime-manifest-sha256",
        str(external["runtime_manifest_sha256"]),
        "--python",
        str(external["interpreter"]),
        "--runner",
        str(runner),
        "--checkout-root",
        str(checkout),
        "--runner-sha256",
        runner_sha256 or sha256_file(runner),
        "--prereg-sha256",
        sha256_file(checkout / ocsc_runner.SOURCE_PATHS[0]),
        "--generator-sha256",
        sha256_file(checkout / ocsc_runner.GENERATOR_RELATIVE_PATH),
        "--tests-sha256",
        sha256_file(checkout / ocsc_runner.SOURCE_PATHS[2]),
        "--oracle-sha256",
        sha256_file(checkout / ocsc_runner.SOURCE_PATHS[4]),
        "--python-sha256",
        str(external["interpreter_sha256"]),
        "--profile",
        "test",
        "--",
        *generator_argv,
    ]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(ocsc.jsonl_bytes(rows))


def make_byte_tokenizer(path: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: index for index, token in enumerate(alphabet)},
            merges=[],
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(path))
    return path


def registry_row(use: str, family: str, index: int) -> dict:
    prompt_id = "{}-{}-{:04d}".format(
        use.replace("_", "-"), family.replace("_", "-"), index
    )
    source_commitment = sha256_bytes(
        "source|{}|{}|{}".format(use, family, index).encode("ascii")
    )
    semantic_signature = sha256_bytes(
        "semantic|{}|{}|{}".format(use, family, index).encode("ascii")
    )
    if use == "replay":
        prompt = "Retention {} context {:04d}.".format(family, index)
        return {
            "prompt_id": prompt_id,
            "family": family,
            "use": use,
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("ascii")),
            "normalized_prompt_sha256": ocsc.normalized_prompt_sha256(prompt),
            "semantic_signature_sha256": semantic_signature,
            "source_commitment": source_commitment,
        }
    opaque = "opaque|{}|{}|{}".format(use, family, index)
    return {
        "prompt_id": prompt_id,
        "family": family,
        "use": use,
        "prompt_sha256": sha256_bytes(("content|" + opaque).encode("ascii")),
        "normalized_prompt_sha256": sha256_bytes(
            ("normalized|" + opaque).encode("ascii")
        ),
        "semantic_signature_sha256": semantic_signature,
        "source_commitment": source_commitment,
    }


def make_prompt_registry(path: Path) -> Path:
    rows = [
        registry_row(use, family, index)
        for (use, family), count in ocsc.REGISTRY_COUNTS.items()
        for index in range(count)
    ]
    write_jsonl(path, rows)
    return path


def hidden_opening_row(
    tokenizer: ocsc.FrozenTokenizer,
    board_id: str,
    ordinal: int,
    row_id: str,
    kind: str,
    state: dict,
    *,
    site_id: str,
    role: str | None,
    reachability: str,
    serializer_slice: str | None,
    pair_id: str | None,
    carry_pair_id: str | None,
    prefix_pair_id: str | None,
    endpoint: str,
    prefix_variant: str | None,
    intervention_field: str | None,
    intervention_position: int | None,
    orientation: str | None,
) -> dict:
    prompt = (
        ocsc.transition_prompt(state)
        if kind == "transition"
        else ocsc.serializer_prompt(state)
    )
    response = (
        ocsc.canonical_state(ocsc.apply_microstep(state))
        if kind == "transition"
        else "answer={}".format(ocsc.state_answer(state))
    )
    local_target = ocsc._local_target(state) if kind == "transition" else None
    return {
        "schema": "shohin-ocsc-hidden-opening-row-v2",
        "board_id": board_id,
        "ordinal": ordinal,
        "row_id": row_id,
        "kind": kind,
        "width": int(state["w"]),
        "role": role,
        "position": int(state["p"]),
        "operation": state["op"],
        "incoming_carry": int(state["c"]),
        "reachability": reachability,
        "serializer_slice": serializer_slice,
        "site_id": site_id,
        "pair_id": pair_id,
        "carry_pair_id": carry_pair_id,
        "prefix_pair_id": prefix_pair_id,
        "endpoint": endpoint,
        "prefix_variant": prefix_variant,
        "intervention_field": intervention_field,
        "intervention_position": intervention_position,
        "orientation": orientation,
        "state": ocsc.canonical_state(state),
        "prompt": prompt,
        "response": response,
        "prompt_sha256": sha256_bytes(prompt.encode("ascii")),
        "normalized_prompt_sha256": ocsc.normalized_prompt_sha256(prompt),
        "semantic_signature_sha256": ocsc.semantic_signature(state, kind),
        "local_target": local_target,
        "scoring_contract": ocsc.hidden_scoring_contract(tokenizer, state, kind),
    }


def select_semantically_covered_cells(
    width: int, role: str, candidates: list[ocsc.Cell], count: int
) -> list[ocsc.Cell]:
    contract = ocsc.hidden_transition_semantic_contract()
    unique = {}
    for cell in candidates:
        if role == "terminal_sub" and cell.left_digit <= cell.right_digit:
            continue
        key = (cell.operation, cell.position, cell.left_digit, cell.right_digit)
        unique.setdefault(key, cell)
    pool = list(unique.values())
    if role in {"terminal_add", "terminal_sub"}:
        required_pairs = (
            [(digit, digit) for digit in range(10)]
            if role == "terminal_add"
            else [(digit + 1, digit) for digit in range(9)]
        )
        selected = []
        used = set()
        for left_digit, right_digit in required_pairs:
            cell = next(
                cell
                for cell in pool
                if cell.left_digit == left_digit and cell.right_digit == right_digit
            )
            selected.append(cell)
            used.add((cell.operation, cell.position, left_digit, right_digit))
        for cell in pool:
            key = (cell.operation, cell.position, cell.left_digit, cell.right_digit)
            if key not in used:
                selected.append(cell)
                used.add(key)
            if len(selected) == count:
                return selected
        raise AssertionError("insufficient terminal semantic coverage cells")

    positions = sorted({cell.position for cell in pool})
    base, remainder = divmod(count, len(positions))
    quotas = {
        position: base + int(index < remainder)
        for index, position in enumerate(positions)
    }
    rng = random.Random(
        ocsc.stable_seed("hidden-semantic-cells|{}|{}".format(width, role))
    )
    for _ in range(100_000):
        selected = []
        for position in positions:
            position_pool = [cell for cell in pool if cell.position == position]
            rng.shuffle(position_pool)
            selected.extend(position_pool[: quotas[position]])
        operation_counts = Counter(cell.operation for cell in selected)
        left_counts = Counter(cell.left_digit for cell in selected)
        right_counts = Counter(cell.right_digit for cell in selected)
        if (
            len(selected) == count
            and all(
                operation_counts[operation]
                >= contract["minimum_sites_per_required_operation"][role]
                for operation in contract["required_operations_by_role"][role]
            )
            and all(
                left_counts[digit]
                >= contract["minimum_sites_per_required_active_digit"][role]
                for digit in contract["required_active_left_digits"][role]
            )
            and all(
                right_counts[digit]
                >= contract["minimum_sites_per_required_active_digit"][role]
                for digit in contract["required_active_right_digits"][role]
            )
        ):
            return selected
    raise AssertionError("failed to select semantic coverage cells")


def make_hidden_opening(path: Path, tokenizer_path: Path) -> tuple[Path, list[dict]]:
    board_id = "ocsc-hidden-confirmation-v2"
    train_transitions, _ = ocsc.generate_ocsc_transition_rows()
    train_serializers, _ = ocsc.generate_serializer_rows()
    tokenizer = ocsc.FrozenTokenizer(tokenizer_path, "test")
    iid_transitions = ocsc.generate_iid_transition_rows(train_transitions, tokenizer)
    all_training_rows = train_transitions + iid_transitions + train_serializers
    blocked_prompts = {row["normalized_prompt_sha256"] for row in all_training_rows}
    blocked_signatures = {row["semantic_signature_sha256"] for row in all_training_rows}
    seen_prompts = set(blocked_prompts)
    seen_signatures = set(blocked_signatures)
    rows: list[dict] = []

    for width in ocsc.WIDTHS:
        cells = ocsc.cells_for_width(width)
        by_stratum = defaultdict(list)
        for cell in cells:
            by_stratum[cell.role].append(cell)

        initial_cells = select_semantically_covered_cells(
            width,
            "initial",
            by_stratum["initial"],
            ocsc.HIDDEN_INITIAL_SITES_PER_WIDTH,
        )
        for site_index in range(ocsc.HIDDEN_INITIAL_SITES_PER_WIDTH):
            cell = initial_cells[site_index % len(initial_cells)]
            site_id = "hidden-init-w{}-s{:02d}".format(width, site_index)
            pair_id = site_id + "-suffix-pair"
            for attempt in range(100_000):
                nonce = 50_000 + width * 100_000 + site_index * 101 + attempt
                anchor = ocsc.reachable_context_state(cell, site_index % 3, nonce)
                variant, field, position = ocsc.initial_suffix_variant(
                    anchor, site_index % 2
                )
                try:
                    natural_variant = ocsc.state_at(
                        variant["op"],
                        ocsc.value_lsf(variant["a"]),
                        ocsc.value_lsf(variant["b"]),
                        variant["w"],
                        0,
                    )
                except (AssertionError, ValueError, ocsc.ContractError):
                    continue
                if ocsc.canonical_state(natural_variant) != ocsc.canonical_state(
                    variant
                ):
                    continue
                candidate_rows = []
                for endpoint, state in (("anchor", anchor), ("variant", variant)):
                    candidate_rows.append(
                        hidden_opening_row(
                            tokenizer,
                            board_id,
                            len(rows) + len(candidate_rows),
                            "hidden-transition-{:04d}".format(
                                len(rows) + len(candidate_rows)
                            ),
                            "transition",
                            state,
                            site_id=site_id,
                            role="initial",
                            reachability="reachable",
                            serializer_slice=None,
                            pair_id=pair_id,
                            carry_pair_id=None,
                            prefix_pair_id=pair_id,
                            endpoint=endpoint,
                            prefix_variant=None,
                            intervention_field=field,
                            intervention_position=position,
                            orientation=None,
                        )
                    )
                if any(
                    row["normalized_prompt_sha256"] in seen_prompts
                    or row["semantic_signature_sha256"] in seen_signatures
                    for row in candidate_rows
                ):
                    continue
                for row in candidate_rows:
                    seen_prompts.add(row["normalized_prompt_sha256"])
                    seen_signatures.add(row["semantic_signature_sha256"])
                rows.extend(candidate_rows)
                break
            else:
                raise AssertionError("failed to construct hidden initial site")

        for role, site_count in ocsc.HIDDEN_NONINITIAL_SITE_COUNTS.items():
            candidates = select_semantically_covered_cells(
                width, role, by_stratum[role], site_count
            )
            for site_index in range(site_count):
                cell = candidates[site_index % len(candidates)]
                site_id = "hidden-carry-w{}-{}-s{:02d}".format(
                    width, role.replace("_", "-"), site_index
                )
                for attempt in range(100_000):
                    nonce = 100_000 + len(rows) * 101 + attempt
                    anchor = ocsc.reachable_context_state(cell, site_index % 3, nonce)
                    prefix, position = ocsc.intervention_state(anchor, site_index % 2)
                    natural_carry = int(anchor["c"])
                    candidate_rows = []
                    for prefix_variant, source in (
                        ("anchor", anchor),
                        ("intervention", prefix),
                    ):
                        for carry in (0, 1):
                            state = dict(source)
                            state["c"] = carry
                            if prefix_variant == "anchor" and carry == natural_carry:
                                reachability = "reachable"
                            elif prefix_variant == "anchor":
                                reachability = "carry_interventional"
                            elif carry == natural_carry:
                                reachability = "prefix_interventional"
                            else:
                                reachability = "prefix_and_carry_interventional"
                            candidate_rows.append(
                                hidden_opening_row(
                                    tokenizer,
                                    board_id,
                                    len(rows) + len(candidate_rows),
                                    "hidden-transition-{:04d}".format(
                                        len(rows) + len(candidate_rows)
                                    ),
                                    "transition",
                                    state,
                                    site_id=site_id,
                                    role=role,
                                    reachability=reachability,
                                    serializer_slice=None,
                                    pair_id=None,
                                    carry_pair_id="{}-{}-carry-pair".format(
                                        site_id, prefix_variant
                                    ),
                                    prefix_pair_id="{}-c{}-prefix-pair".format(
                                        site_id, carry
                                    ),
                                    endpoint="c{}".format(carry),
                                    prefix_variant=prefix_variant,
                                    intervention_field="r",
                                    intervention_position=position,
                                    orientation=None,
                                )
                            )
                    if any(
                        row["normalized_prompt_sha256"] in seen_prompts
                        or row["semantic_signature_sha256"] in seen_signatures
                        for row in candidate_rows
                    ):
                        continue
                    for row in candidate_rows:
                        seen_prompts.add(row["normalized_prompt_sha256"])
                        seen_signatures.add(row["semantic_signature_sha256"])
                    rows.extend(candidate_rows)
                    break
                else:
                    raise AssertionError("failed to construct hidden carry site")

    train_tapes = {row["tape"] for row in train_serializers}
    for width in ocsc.WIDTHS:
        used_tapes = set(train_tapes)
        pair_index = 0
        for pattern in ocsc.hidden_serializer_patterns(width):
            for translation in range(10):
                forward = "".join(str((digit + translation) % 10) for digit in pattern)
                reverse = forward[::-1]
                assert forward != reverse
                assert not {forward, reverse} & used_tapes
                used_tapes.update((forward, reverse))
                left, right = ocsc.serializer_operands(
                    width, 1_000 + pair_index, (forward, reverse)
                )
                site_id = "hidden-ser-w{}-site{:02d}".format(width, pair_index)
                for slice_name, (operation, carry) in {
                    "add_c0": ("add", 0),
                    "add_c1": ("add", 1),
                    "sub_c0": ("sub", 0),
                }.items():
                    pair_id = "hidden-ser-w{}-p{:02d}-{}".format(
                        width, pair_index, slice_name.replace("_", "-")
                    )
                    for orientation, tape in (
                        ("forward", forward),
                        ("reverse", reverse),
                    ):
                        state = ocsc.serializer_state(
                            width, operation, carry, left, right, tape
                        )
                        candidate = hidden_opening_row(
                            tokenizer,
                            board_id,
                            len(rows),
                            "hidden-serializer-{:04d}".format(len(rows)),
                            "serializer",
                            state,
                            site_id=site_id,
                            role=None,
                            reachability="interventional",
                            serializer_slice=slice_name,
                            pair_id=pair_id,
                            carry_pair_id=None,
                            prefix_pair_id=None,
                            endpoint=orientation,
                            prefix_variant=None,
                            intervention_field=None,
                            intervention_position=None,
                            orientation=orientation,
                        )
                        assert candidate["normalized_prompt_sha256"] not in seen_prompts
                        assert (
                            candidate["semantic_signature_sha256"]
                            not in seen_signatures
                        )
                        seen_prompts.add(candidate["normalized_prompt_sha256"])
                        seen_signatures.add(candidate["semantic_signature_sha256"])
                        rows.append(candidate)
                pair_index += 1
        assert pair_index == 50
    assert len(rows) == 3_600
    write_jsonl(path, rows)
    return path, rows


def make_custodian_opening(path: Path, board_id: str) -> tuple[Path, dict]:
    document = {
        "schema": "shohin-ocsc-custodian-opening-v1",
        "board_id": board_id,
        "custodian_id": "test-custodian",
        "nonce_hex": sha256_bytes(b"test-hidden-custodian-nonce"),
    }
    path.write_bytes(ocsc.canonical_json_bytes(document, newline=True))
    return path, document


def make_confirmation_commitment(
    path: Path,
    rows: list[dict],
    custodian_opening: dict,
    root: str | None = None,
) -> Path:
    canonical_rows = [ocsc.canonical_json_bytes(row) for row in rows]
    document = {
        "schema": "shohin-ocsc-hidden-merkle-commitment-v2",
        "board_id": rows[0]["board_id"],
        "leaf_count": 3_600,
        "merkle_root": root or ocsc.hidden_merkle_root(canonical_rows),
        "geometry": ocsc.hidden_geometry_contract(),
        "geometry_sha256": ocsc.hash_json(ocsc.hidden_geometry_contract()),
        "merkle_algorithm": ocsc.hidden_merkle_algorithm(),
        "custodian_commitment_algorithm": ocsc.custodian_commitment_algorithm(),
        "custodian_commitment": ocsc.custodian_opening_commitment(custodian_opening),
        "secret_rows_in_document": False,
    }
    path.write_bytes(ocsc.canonical_json_bytes(document, newline=True))
    return path


def freeze_custody_root(path: Path) -> Path:
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)
    return path


def make_publication_commitment(
    path: Path,
    output: Path,
    tokenizer: Path,
    registry: Path,
    confirmation: Path,
) -> Path:
    request = ocsc.publication_commitment_request(
        "test", output, tokenizer, registry, confirmation, 0
    )
    return write_signed_publication_commitment(path, request)


def write_signed_publication_commitment(path: Path, request: dict) -> Path:
    unsigned = {
        "schema": "shohin-ocsc-prepublication-commitment-v1",
        "custodian_id": "independent-test-custodian",
        "sequence": 1,
        "nonce_hex": sha256_bytes(b"ocsc-test-prepublication-nonce-v1"),
        "request": request,
        "request_sha256": ocsc.hash_json(request),
        "signature_algorithm": "ed25519",
        "signer_public_key_hex": ocsc.TRUSTED_PUBLICATION_KEYS["test"],
    }
    private_key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(TEST_PUBLICATION_PRIVATE_KEY_HEX)
    )
    document = dict(unsigned)
    document["signature_hex"] = private_key.sign(
        ocsc.publication_signing_payload(unsigned)
    ).hex()
    path.write_bytes(ocsc.canonical_json_bytes(document, newline=True))
    return path


def write_signed_independent_review_receipt(path: Path, request: dict) -> Path:
    unsigned = {
        "schema": "shohin-ocsc-independent-review-receipt-v1",
        "reviewer_id": "independent-test-reviewer",
        "sequence": 1,
        "nonce_hex": sha256_bytes(b"ocsc-test-independent-review-nonce-v1"),
        "decision": "approve-cpu-publication-contract-only",
        "review_request": request,
        "review_request_sha256": ocsc.hash_json(request),
        "signature_algorithm": "ed25519",
        "signer_public_key_hex": ocsc.TRUSTED_INDEPENDENT_REVIEW_KEYS["test"],
    }
    private_key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(TEST_INDEPENDENT_REVIEW_PRIVATE_KEY_HEX)
    )
    document = dict(unsigned)
    document["signature_hex"] = private_key.sign(
        ocsc.independent_review_signing_payload(unsigned)
    ).hex()
    path.write_bytes(ocsc.canonical_json_bytes(document, newline=True))
    return path


def write_signed_linux_lustre_qualification_receipt(
    path: Path,
    unsigned: dict,
) -> Path:
    private_key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX)
    )
    document = dict(unsigned)
    document["signature_hex"] = private_key.sign(
        ocsc.linux_lustre_qualification_signing_payload(unsigned)
    ).hex()
    path.write_bytes(ocsc.canonical_json_bytes(document, newline=True))
    return path


def make_qualification_evidence(
    root: Path,
    source_manifest_sha256: str,
) -> dict:
    root.mkdir(mode=0o700)
    mountpoint = str(Path(os.path.abspath(os.fspath(root))))
    mount_metadata = root.stat()
    retained_evidence = []
    for crash_point in ocsc.QUALIFICATION_CRASH_POINTS:
        canonical = crash_point == "canonical-before-parent-fsync"
        journal_absent = crash_point == "stage-created-before-journal"
        retained_evidence.append(
            {
                "crash_point": crash_point,
                "custody_path": str(root / ("evidence-" + crash_point)),
                "tree_device": mount_metadata.st_dev,
                "tree_inode": mount_metadata.st_ino,
                "tree_inventory_sha256": sha256_bytes(
                    ("tree|" + crash_point).encode("ascii")
                ),
                "journal_sha256": (
                    None
                    if journal_absent
                    else sha256_bytes(("journal|" + crash_point).encode("ascii"))
                ),
                "lease_sha256": sha256_bytes(("lease|" + crash_point).encode("ascii")),
                "stage_state": "renamed-canonical" if canonical else "retained",
                "canonical_state": "retained" if canonical else "absent",
                "journal_state": "absent" if journal_absent else "retained",
                "lease_state": "retained",
            }
        )
    events = []
    previous_by_host = {"host-a": "0" * 64, "host-b": "0" * 64}
    sequence_by_host = {"host-a": 0, "host-b": 0}
    for index, check_name in enumerate(ocsc.LINUX_LUSTRE_QUALIFICATION_CHECKS):
        host_id = "host-a" if index % 2 == 0 else "host-b"
        sequence_by_host[host_id] += 1
        details = {
            "operation": check_name,
            "outcome": "observed-from-raw-evidence",
        }
        if check_name == "publication_path_complete":
            details.update(
                {
                    "path_steps": [
                        "production-broker-transfer",
                        "publish_bundle-stage-no-replace",
                        "file-fsync-after-chmod",
                        "stage-fsync",
                        "rename-noreplace",
                        "parent-fsync",
                        "descriptor-inode-readback",
                    ],
                    "publication_receipt_sha256": sha256_bytes(b"publication-receipt"),
                    "bundle_manifest_sha256": sha256_bytes(b"bundle-manifest"),
                    "output_device": mount_metadata.st_dev,
                    "output_inode": mount_metadata.st_ino,
                }
            )
        if check_name in {
            "all_crash_evidence_permanently_retained",
            "permanent_evidence_inventory_recorded",
        }:
            details["retained_evidence"] = copy.deepcopy(retained_evidence)
        details["evidence_sha256"] = ocsc.hash_json(details)
        unsigned = ocsc.qualification_event_unsigned(
            qualification_id="qualification-source-test",
            host_id=host_id,
            host_fqdn=host_id + ".example.invalid",
            host_kernel_identity_sha256=sha256_bytes(
                ("kernel|" + host_id).encode("ascii")
            ),
            sequence=sequence_by_host[host_id],
            previous_event_sha256=previous_by_host[host_id],
            event_type=check_name,
            nonce_hex=sha256_bytes(("nonce|" + check_name).encode("ascii")),
            source_manifest_sha256=source_manifest_sha256,
            lustre_mount_source="lustre-test-source",
            lustre_mountpoint=mountpoint,
            output_parent_device=mount_metadata.st_dev,
            output_parent_inode=mount_metadata.st_ino,
            details=details,
            mode="test",
        )
        event = ocsc.sign_qualification_event(
            unsigned,
            TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX,
            "test",
        )
        events.append(event)
        previous_by_host[host_id] = ocsc.hash_json(event)
    broker_event = next(
        event
        for event in events
        if event["event_type"] == "production_broker_transfer_complete"
    )
    source_dir = root / "broker-source"
    broker_dir = root / "broker-records"
    publication_dir = root / "publication-events"
    for directory in (source_dir, broker_dir, publication_dir):
        directory.mkdir(mode=0o700)
    source_event_path = source_dir / (broker_event["event_id"] + ".event.json")
    source_event_path.write_bytes(ocsc.canonical_json_bytes(broker_event, newline=True))
    source_event_path.chmod(0o444)
    transfer = ocsc.execute_qualification_broker_transfer(
        source_event_path,
        broker_dir,
        publication_dir,
        broker_id="production-broker",
        sequence=1,
        previous_request_sha256="0" * 64,
        previous_receipt_sha256="0" * 64,
        expected_source_manifest_sha256=source_manifest_sha256,
        private_key_hex=TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX,
        mode="test",
    )
    report = ocsc.derive_qualification_report(
        events,
        [transfer["request"]],
        [transfer["receipt"]],
        source_manifest_sha256,
        "test",
    )
    return {
        "events": events,
        "broker_requests": [transfer["request"]],
        "broker_receipts": [transfer["receipt"]],
        "report": report,
        "marker": ocsc.qualification_marker(report),
        "transfer": transfer,
    }


def generate_bundle(
    output: Path,
    tokenizer: Path,
    registry: Path,
    confirmation: Path,
    publication_commitment: Path,
    independent_review_receipt: Path,
) -> subprocess.CompletedProcess:
    return run_source_bound(
        [
            "--output-dir",
            str(output),
            "--mode",
            "test",
            "--tokenizer",
            str(tokenizer),
            "--prompt-registry",
            str(registry),
            "--secret-confirmation-commitment",
            str(confirmation),
            "--publication-commitment",
            str(publication_commitment),
            "--independent-review-receipt",
            str(independent_review_receipt),
            "--pad-token-id",
            "0",
        ],
        check=True,
    )


@pytest.fixture(scope="session")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path | list[dict]]:
    root = tmp_path_factory.mktemp("ocsc-v2")
    tokenizer_root = root / "tokenizer-root"
    tokenizer_root.mkdir()
    tokenizer = freeze_custody_root(
        make_byte_tokenizer(tokenizer_root / "tokenizer.json")
    )
    registry_root = root / "registry-root"
    registry_root.mkdir()
    registry = freeze_custody_root(
        make_prompt_registry(registry_root / "prompt_registry.jsonl")
    )
    opening_root = root / "opening-root"
    opening_root.mkdir()
    opening, hidden_rows = make_hidden_opening(
        opening_root / "hidden_opening.jsonl", tokenizer
    )
    freeze_custody_root(opening)
    custodian_root = root / "custodian-root"
    custodian_root.mkdir()
    custodian_opening, custodian_document = make_custodian_opening(
        custodian_root / "custodian_opening.json", hidden_rows[0]["board_id"]
    )
    freeze_custody_root(custodian_opening)
    confirmation_root = root / "confirmation-root"
    confirmation_root.mkdir()
    confirmation = make_confirmation_commitment(
        confirmation_root / "confirmation.json",
        hidden_rows,
        custodian_document,
    )
    freeze_custody_root(confirmation)
    bundle = root / "bundle"
    publication_root = root / "publication-root"
    publication_root.mkdir()
    publication_commitment = freeze_custody_root(
        make_publication_commitment(
            publication_root / "prepublication_commitment.json",
            bundle,
            tokenizer,
            registry,
            confirmation,
        )
    )
    request = ocsc.publication_commitment_request(
        "test", bundle, tokenizer, registry, confirmation, 0
    )
    publication_receipt = ocsc.load_publication_commitment(
        publication_commitment,
        request,
        "test",
        require_unpublished=True,
    )
    artifacts = ocsc.build_artifacts(
        "test",
        tokenizer,
        registry,
        confirmation,
        0,
        publication_receipt,
        expected_output_dir=bundle,
        require_unpublished=True,
    )
    review_root = root / "independent-review-root"
    review_root.mkdir()
    independent_review_receipt_path = freeze_custody_root(
        write_signed_independent_review_receipt(
            review_root / "independent_review_receipt.json",
            ocsc.independent_review_request(request, artifacts),
        )
    )
    independent_review_receipt = ocsc.load_independent_review_receipt(
        independent_review_receipt_path,
        ocsc.independent_review_request(request, artifacts),
        "test",
    )
    assert not bundle.exists()
    ocsc.publish_bundle(
        bundle,
        artifacts,
        mode="test",
        tokenizer_path=tokenizer,
        prompt_registry_path=registry,
        confirmation_path=confirmation,
        pad_token_id=0,
        publication_receipt=publication_receipt,
        independent_review_receipt=independent_review_receipt,
    )
    return {
        "root": root,
        "tokenizer": tokenizer,
        "registry": registry,
        "opening": opening,
        "hidden_rows": hidden_rows,
        "confirmation": confirmation,
        "custodian_opening": custodian_opening,
        "publication_commitment": publication_commitment,
        "independent_review_receipt": independent_review_receipt_path,
        "bundle": bundle,
    }


def rebind_test_artifacts(
    generated: dict[str, Path | list[dict]], publication_receipt: dict
) -> dict[str, bytes]:
    bundle = Path(generated["bundle"])
    artifacts = {name: (bundle / name).read_bytes() for name in ocsc.ARTIFACT_NAMES}
    audit = json.loads(artifacts["audit_report.json"])
    audit["prepublication_custody"] = {
        "receipt_sha256": publication_receipt["physical_sha256"],
        "request_sha256": publication_receipt["request_sha256"],
        "custodian_id": publication_receipt["custodian_id"],
        "sequence": publication_receipt["sequence"],
        "signer_public_key_hex": publication_receipt["signer_public_key_hex"],
        "signature_verified_before_publication": True,
        "postpublication_self_attestation_accepted": False,
    }
    audit.pop("payload_sha256")
    audit = ocsc.with_payload_hash(audit, "payload_sha256")
    artifacts["audit_report.json"] = ocsc.pretty_json_bytes(audit)

    manifest = json.loads(artifacts["manifest.json"])
    manifest["files"] = {
        name: {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in sorted(artifacts.items())
        if name != "manifest.json"
    }
    manifest["inputs"].update(
        {
            "prepublication_commitment_sha256": publication_receipt["physical_sha256"],
            "prepublication_commitment_bytes": publication_receipt["physical_bytes"],
            "prepublication_commitment_path": publication_receipt["resolved_path"],
            "prepublication_commitment_file_device": publication_receipt[
                "physical_file_device"
            ],
            "prepublication_commitment_file_inode": publication_receipt[
                "physical_file_inode"
            ],
            "prepublication_commitment_custody_root_path": publication_receipt[
                "custody_root_path"
            ],
            "prepublication_commitment_custody_root_device": publication_receipt[
                "custody_root_device"
            ],
            "prepublication_commitment_custody_root_inode": publication_receipt[
                "custody_root_inode"
            ],
            "prepublication_request_sha256": publication_receipt["request_sha256"],
            "prepublication_custodian_id": publication_receipt["custodian_id"],
            "prepublication_sequence": publication_receipt["sequence"],
            "prepublication_signer_public_key_hex": publication_receipt[
                "signer_public_key_hex"
            ],
        }
    )
    manifest.pop("payload_sha256")
    manifest = ocsc.with_payload_hash(manifest, "payload_sha256")
    artifacts["manifest.json"] = ocsc.pretty_json_bytes(manifest)
    return artifacts


def authorized_publication_case(
    generated: dict[str, Path | list[dict]],
    root: Path,
    output: Path,
) -> dict:
    publication_root = root / "publication-root"
    publication_root.mkdir()
    publication_commitment = freeze_custody_root(
        make_publication_commitment(
            publication_root / "prepublication_commitment.json",
            output,
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["confirmation"]),
        )
    )
    request = ocsc.publication_commitment_request(
        "test",
        output,
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        0,
    )
    publication_receipt = ocsc.load_publication_commitment(
        publication_commitment,
        request,
        "test",
        require_unpublished=True,
    )
    artifacts = rebind_test_artifacts(generated, publication_receipt)
    review_root = root / "independent-review-root"
    review_root.mkdir()
    independent_review_path = freeze_custody_root(
        write_signed_independent_review_receipt(
            review_root / "independent_review_receipt.json",
            ocsc.independent_review_request(request, artifacts),
        )
    )
    independent_review_receipt = ocsc.load_independent_review_receipt(
        independent_review_path,
        ocsc.independent_review_request(request, artifacts),
        "test",
    )
    return {
        "artifacts": artifacts,
        "publication_commitment": publication_commitment,
        "publication_receipt": publication_receipt,
        "independent_review_path": independent_review_path,
        "independent_review_receipt": independent_review_receipt,
        "request": request,
    }


def lightweight_publication_verifier(path: Path, directory_fd: int, **kwargs) -> dict:
    with ocsc.PinnedBundle(path, directory_fd=directory_fd) as bundle_snapshot:
        finalizer = kwargs.get("pinned_finalizer")
        if finalizer is not None:
            finalizer(bundle_snapshot)
        bundle_snapshot.assert_unchanged()
    return {}


def leave_stale_publication_lease(
    parent_fd: int,
    publication_receipt: dict,
    independent_review_receipt: dict,
) -> tuple[str, dict]:
    lease_name = ocsc.publication_lease_name(
        publication_receipt, independent_review_receipt
    )
    lease = ocsc._acquire_publication_lease(parent_fd, lease_name)
    record = copy.deepcopy(lease.record)
    ocsc._release_publication_lease(lease)
    return lease_name, record


def test_complete_basis_and_matched_local_relations(
    generated: dict[str, Path | list[dict]],
) -> None:
    bundle = Path(generated["bundle"])
    rows = read_jsonl(bundle / "ocsc_train.jsonl")
    transitions = [row for row in rows if row["kind"] == "transition"]
    relations = read_jsonl(bundle / "relational_pairs.jsonl")
    local = [
        pair
        for pair in relations
        if pair["relation"]
        in {"local_prefix_intervention", "initial_suffix_context_invariance"}
    ]
    assert len(transitions) == 22_500
    assert Counter(pair["relation"] for pair in local) == {
        "local_prefix_intervention": 7_000,
        "initial_suffix_context_invariance": 2_000,
    }
    assert Counter(
        pair["incoming_carry"]
        for pair in local
        if pair["relation"] == "local_prefix_intervention" and pair["factorial_active"]
    ) == {0: 3_450, 1: 3_450}
    by_id = {row["row_id"]: row for row in transitions}
    for pair in local:
        left, right = by_id[pair["left_row_id"]], by_id[pair["right_row_id"]]
        ocsc.assert_matched_local_pair(left, right, pair["relation"])
        if pair["relation"] == "local_prefix_intervention":
            assert left["reachability"] == "reachable"
            assert right["reachability"] == "interventional"
            assert right["perturbation_field"] == "r"
        else:
            assert left["reachability"] == right["reachability"] == "reachable"
            assert right["intervention"] == "none"
            assert right["perturbation_field"] in {"a", "b"}
            assert pair["factorial_active"] is False

    for width in ocsc.WIDTHS:
        cells = ocsc.cells_for_width(width)
        assert len(cells) == 900
        assert Counter(cell.role for cell in cells) == Counter(ocsc.ROLE_CELL_COUNTS)
        terminal_sub = [cell for cell in cells if cell.role == "terminal_sub"]
        assert Counter(cell.incoming_carry for cell in terminal_sub) == {0: 55, 1: 45}
        assert all(
            cell.left_digit >= cell.right_digit
            if cell.incoming_carry == 0
            else cell.left_digit > cell.right_digit
            for cell in terminal_sub
        )
        positions = Counter(cell.position for cell in cells if cell.role == "interior")
        assert set(positions) == set(range(1, width - 1))
        assert sum(positions.values()) == 400
        assert max(positions.values()) - min(positions.values()) <= 1


def test_independent_arithmetic_replay_rejects_oracle_and_row_mutation(
    generated: dict[str, Path | list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = read_jsonl(Path(generated["bundle"]) / "ocsc_train.jsonl")
    replay = ocsc.validate_independent_arithmetic_rows(rows)
    assert replay == {
        "schema": "shohin-ocsc-independent-arithmetic-replay-v1",
        "transition_rows": 22_500,
        "serializer_rows": 1_500,
        "reviewed_oracle_sha256": ocsc.DIGITWISE_PROTOCOL_SHA256,
        "reviewed_oracle_bytes": ocsc.DIGITWISE_PROTOCOL_BYTES,
        "all_rows_exact": True,
    }

    mutated = dict(next(row for row in rows if row["kind"] == "transition"))
    mutated["local_target"] = dict(mutated["local_target"])
    mutated["local_target"]["digit"] = (mutated["local_target"]["digit"] + 1) % 10
    with pytest.raises(ocsc.ContractError, match="independent transition replay"):
        ocsc.validate_independent_arithmetic_rows([mutated])

    reviewed_snapshot = ocsc._REVIEWED_DIGITWISE_PROTOCOL_SNAPSHOT
    monkeypatch.setattr(
        ocsc,
        "_REVIEWED_DIGITWISE_PROTOCOL_SNAPSHOT",
        ocsc.FileSnapshot(
            payload=b"coherently wrong reviewed oracle",
            resolved_path=reviewed_snapshot.resolved_path,
            metadata=reviewed_snapshot.metadata,
            parent_resolved_path=reviewed_snapshot.parent_resolved_path,
            parent_metadata=reviewed_snapshot.parent_metadata,
        ),
    )
    with pytest.raises(ocsc.ContractError, match="source identity mismatch"):
        ocsc.source_manifest()
    monkeypatch.setattr(
        ocsc,
        "_REVIEWED_DIGITWISE_PROTOCOL_SNAPSHOT",
        reviewed_snapshot,
    )

    reviewed_apply = ocsc.apply_microstep

    def coherently_wrong_apply(state: dict) -> dict:
        result = reviewed_apply(state)
        digits = list(result["r"])
        digits[state["p"]] = str((int(digits[state["p"]]) + 1) % 10)
        result["r"] = "".join(digits)
        return result

    monkeypatch.setattr(ocsc, "apply_microstep", coherently_wrong_apply)
    with pytest.raises(ocsc.ContractError, match="oracle disagrees"):
        ocsc.validate_independent_arithmetic_rows([rows[0]])


def test_reviewed_oracle_is_authenticated_before_any_byte_executes(
    tmp_path: Path,
) -> None:
    hostile_root = tmp_path / "hostile-import"
    (hostile_root / "pipeline").mkdir(parents=True)
    (hostile_root / "train").mkdir()
    copied_generator = hostile_root / "pipeline" / GENERATOR.name
    shutil.copyfile(GENERATOR, copied_generator)
    marker = hostile_root / "oracle-import-side-effect"
    malicious_oracle = (ROOT / "train" / "digitwise_protocol.py").read_bytes() + (
        "\nopen({!r}, 'w', encoding='ascii').write('executed')\n".format(str(marker))
    ).encode("ascii")
    (hostile_root / "train" / "digitwise_protocol.py").write_bytes(malicious_oracle)
    result = subprocess.run(
        [sys.executable, str(copied_generator), "--help"],
        cwd=hostile_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "direct generator execution is forbidden" in result.stderr
    assert not marker.exists()


def test_exact_corpora_iid_slot_geometry_hashes_and_leakage(
    generated: dict[str, Path | list[dict]],
) -> None:
    bundle = Path(generated["bundle"])
    ocsc_rows = read_jsonl(bundle / "ocsc_train.jsonl")
    iid_rows = read_jsonl(bundle / "iid_control_train.jsonl")
    assert len(ocsc_rows) == len(iid_rows) == 24_000
    assert Counter(row["kind"] for row in ocsc_rows) == {
        "transition": 22_500,
        "serializer": 1_500,
    }
    structural = ("width", "role", "position", "operation")
    ocsc_transitions = [row for row in ocsc_rows if row["kind"] == "transition"]
    iid_transitions = [row for row in iid_rows if row["kind"] == "transition"]
    assert Counter(
        tuple(row[field] for field in structural) for row in ocsc_transitions
    ) == Counter(tuple(row[field] for field in structural) for row in iid_transitions)
    assert all(row["reachability"] == "reachable" for row in iid_transitions)
    assert {row["row_id"]: row for row in ocsc_rows if row["kind"] == "serializer"} == {
        row["row_id"]: row for row in iid_rows if row["kind"] == "serializer"
    }
    for row in ocsc_rows + iid_rows:
        payload = dict(row)
        claimed = payload.pop("row_sha256")
        assert ocsc.hash_json(payload) == claimed
        if row["kind"] == "transition":
            ocsc.audit_reachability(row)
    registry, summary = ocsc.load_prompt_registry(Path(generated["registry"]))
    assert summary["rows"] == 4_096 and summary["evaluation_rows"] == 2_816
    assert summary["resolved_path"] == str(Path(generated["registry"]).resolve())
    assert summary["physical_bytes"] == Path(generated["registry"]).stat().st_size
    replay_registry = {
        row["prompt_id"]: row for row in registry if row["use"] == "replay"
    }
    replay_rows = read_jsonl(bundle / "replay_prompts.jsonl")
    assert len(replay_rows) == 1_280
    assert all(
        row["registry_row_sha256"] == ocsc.hash_json(replay_registry[row["replay_id"]])
        for row in replay_rows
    )
    train_prompts = {row["normalized_prompt_sha256"] for row in ocsc_rows + iid_rows}
    assert not train_prompts & {row["normalized_prompt_sha256"] for row in registry}


def test_serializer_diversity_orientation_marginals_and_counterfactuals(
    generated: dict[str, Path | list[dict]],
) -> None:
    bundle = Path(generated["bundle"])
    rows = [
        row
        for row in read_jsonl(bundle / "ocsc_train.jsonl")
        if row["kind"] == "serializer"
    ]
    relations = read_jsonl(bundle / "relational_pairs.jsonl")
    assert Counter(pair["relation"] for pair in relations) == {
        "local_prefix_intervention": 7_000,
        "initial_suffix_context_invariance": 2_000,
        "serializer_reversal": 750,
        "serializer_counterfactual_mismatch": 750,
    }
    for width in ocsc.WIDTHS:
        patterns = ocsc.serializer_patterns(width)
        metrics = [ocsc.serializer_pattern_metrics(pattern) for pattern in patterns]
        assert all(not metric["constant_except_one"] for metric in metrics)
        assert all(metric["non_affine"] for metric in metrics)
        assert all(metric["distinct_digits"] >= min(width, 3) for metric in metrics)
        for index, left in enumerate(patterns):
            for right in patterns[index + 1 :]:
                assert min(
                    ocsc.hamming(left, variant)
                    for variant in ocsc.translated_orbit(right)
                ) >= ocsc.serializer_min_hamming(width)
        width_rows = [row for row in rows if row["width"] == width]
        tapes = {row["tape"] for row in width_rows}
        assert len(width_rows) == 300 and len(tapes) == 100
        assert all(tape != tape[::-1] for tape in tapes)
        for position in range(width):
            assert Counter(tape[position] for tape in tapes) == {
                str(digit): 10 for digit in range(10)
            }
        assert sum(tape[0] == "0" for tape in tapes) == 10
        assert sum(tape[-1] == "0" for tape in tapes) == 10
        by_pair = defaultdict(list)
        for row in width_rows:
            by_pair[row["pair_base_id"]].append(row)
        assert len(by_pair) == 50
        for pair_rows in by_pair.values():
            assert len({row["operand_signature_sha256"] for row in pair_rows}) == 1
            pair_tapes = {row["tape"] for row in pair_rows}
            assert len(pair_tapes) == 2
            first = next(iter(pair_tapes))
            assert first[::-1] in pair_tapes


def test_independent_slot_payloads_and_activation_maps(
    generated: dict[str, Path | list[dict]],
) -> None:
    bundle = Path(generated["bundle"])
    receipts = read_jsonl(bundle / "tokenization_receipt.jsonl")
    packs = read_jsonl(bundle / "packs.jsonl")
    assert Counter(row["record_kind"] for row in receipts) == {
        "training_row": 46_500,
        "replay_prompt": 1_280,
        "dummy_slot": 1,
    }
    receipt_by_hash = {row["receipt_sha256"]: row for row in receipts}
    payload_by_hash = {}
    for receipt in receipts:
        if receipt["record_kind"] == "replay_prompt":
            assert receipt["token_ids_sha256"] == ocsc.int_array_sha256(
                receipt["token_ids"]
            )
            continue
        vectors = ocsc.decode_slot_payload(receipt)
        assert len(vectors["token_ids"]) == 256
        assert sum(vectors["attention_mask"]) == receipt["token_count"]
        assert sum(vectors["completion_mask"]) == receipt["supervised_tokens"]
        assert sum(vectors["raw_weight_units"]) == receipt["raw_weight_units"]
        assert all(
            vectors["position_ids"][index] == index
            for index in range(receipt["token_count"])
        )
        payload_by_hash[receipt["slot_payload"]["uncompressed_sha256"]] = vectors

    assert len(packs) == 9_375
    assert Counter(pack["variant"] for pack in packs) == {
        "A": 4_500,
        "OCSC": 4_500,
        "SHARED": 375,
    }
    for pack in packs:
        expected_real = 5 if pack["geometry"]["kind"] == "transition" else 4
        assert pack["batch_shape"] == [8, 256]
        assert (pack["real_slots"], pack["dummy_slots"]) == (
            expected_real,
            8 - expected_real,
        )
        assert pack["attention"] == {
            "topology": "eight-independent-block-diagonal-causal-lanes",
            "cross_slot_attention": False,
            "causal_mask_reset_per_slot": True,
            "position_ids_reset_per_slot": True,
            "kv_cache_shared_across_slots": False,
        }
        assert [slot["slot_index"] for slot in pack["slots"]] == list(range(8))
        for index, slot in enumerate(pack["slots"]):
            assert slot["real"] is (index < expected_real)
            receipt = receipt_by_hash[slot["receipt_sha256"]]
            assert (
                receipt["slot_payload"]["uncompressed_sha256"]
                == slot["slot_payload_sha256"]
            )
            vectors = payload_by_hash[slot["slot_payload_sha256"]]
            if not slot["real"]:
                assert not any(vectors["attention_mask"])
                assert not any(vectors["completion_mask"])
        for pair_map in pack["pair_activation_maps"]:
            assert pair_map["left_slot"] < expected_real
            assert pair_map["right_slot"] < expected_real
            if pair_map["package"] == "initial_invariance":
                assert pair_map["factorial_active"] is False
            if pair_map["package"] == "serializer":
                assert pair_map["aligned_positions"]
                if pair_map["polarity"] == "counterfactual":
                    assert all(
                        item["left_target_digit"] != item["right_target_digit"]
                        for item in pair_map["aligned_positions"]
                    )


def test_shared_schedule_factorial_resources_and_exact_rational_weights(
    generated: dict[str, Path | list[dict]],
) -> None:
    bundle = Path(generated["bundle"])
    schedule = read_jsonl(bundle / "schedule.jsonl")
    packs = {row["pack_id"]: row for row in read_jsonl(bundle / "packs.jsonl")}
    audit = json.loads((bundle / "audit_report.json").read_text(encoding="ascii"))
    assert len(schedule) == 3 * 6 * 5_120
    by_run = defaultdict(list)
    for row in schedule:
        by_run[(row["paired_seed_index"], row["run_cell"])].append(row)
    replicate_cycles = []
    for seed_index, seed in enumerate(ocsc.PAIRED_SEEDS):
        runs = {cell: by_run[(seed_index, cell)] for cell in ocsc.RUN_CELLS}
        skeletons = [row["skeleton_id"] for row in runs["A"]]
        replicate_cycles.append(tuple(skeletons))
        assert len({row["replicate_cycle_sha256"] for row in runs["A"]}) == 1
        assert all(
            [row["skeleton_id"] for row in runs[cell]] == skeletons
            for cell in ocsc.RUN_CELLS
        )
        for cell, rows in runs.items():
            assert [row["update"] for row in rows] == list(range(5_120))
            first_repeat = next(
                index for index, row in enumerate(rows) if row["pack_occurrence"] == 1
            )
            assert first_repeat == 4_875
            assert Counter(Counter(row["pack_id"] for row in rows).values()) == {
                1: 4_630,
                2: 245,
            }
            assert len({row["replay_id"] for row in rows if row["replay_id"]}) == 1_280
            supervised = 0
            raw_units = 0
            for row in rows:
                pack = packs[row["pack_id"]]
                assert row["update_rng_seed"] == ocsc.paired_update_rng_seed(
                    seed, row["update"]
                )
                supervised += pack["supervised_positions"]
                raw_units += (
                    pack["field_raw_weight_units"]
                    if row["field_weight_profile"] == "carry_serializer_v1"
                    else pack["supervised_positions"]
                )
            scale = Fraction(supervised, raw_units)
            stats = audit["schedules"]["seed{}:{}".format(seed_index, cell)]
            normalized = stats["normalized_token_weight"]
            assert (normalized["scale_numerator"], normalized["scale_denominator"]) == (
                scale.numerator,
                scale.denominator,
            )
            assert Fraction(raw_units) * scale == supervised
            assert (
                Fraction(
                    normalized["scheduled_mean_numerator"],
                    normalized["scheduled_mean_denominator"],
                )
                == 1
            )
            expected_carry_dose = (
                {"0": 3_622, "1": 3_622} if cell in {"M10", "M11"} else {}
            )
            assert (
                stats["active_noninitial_local_pair_presentations_by_incoming_carry"]
                == expected_carry_dose
            )
        for left, right in zip(runs["A"], runs["B"]):
            assert all(
                left[field] == right[field]
                for field in (
                    "row_presentations",
                    "nonpadding_tokens",
                    "supervised_positions",
                    "main_positions",
                )
            )
            if packs[left["pack_id"]]["geometry"]["kind"] == "serializer":
                assert left["pack_id"] == right["pack_id"]
        for update in range(5_120):
            reference = runs["M00"][update]
            for cell in ("M10", "M01", "M11"):
                candidate = runs[cell][update]
                assert all(
                    candidate[field] == reference[field]
                    for field in (
                        "pack_id",
                        "skeleton_id",
                        "update_rng_seed",
                        "replay_id",
                        "nonpadding_tokens",
                        "supervised_positions",
                        "raw_weight_units",
                        "main_positions",
                    )
                )
        assert not any(row["active_pair_ids"] for row in runs["M00"])
        assert all(
            not row["serializer_relation_active"] and row["local_relation_active"]
            for row in runs["M10"]
        )
        assert all(
            row["serializer_relation_active"] and not row["local_relation_active"]
            for row in runs["M01"]
        )
        assert all(
            row["serializer_relation_active"] and row["local_relation_active"]
            for row in runs["M11"]
        )
    assert len(set(replicate_cycles)) == len(ocsc.PAIRED_SEEDS)
    assert audit["shared_skeleton"]["replicate_batch_permutations_distinct"] is True
    assert len(
        set(audit["shared_skeleton"]["replicate_cycle_sha256_by_seed"].values())
    ) == len(ocsc.PAIRED_SEEDS)
    assert len(
        set(audit["shared_skeleton"]["replicate_rng_stream_sha256_by_seed"].values())
    ) == len(ocsc.PAIRED_SEEDS)
    repeat_hashes = audit["shared_skeleton"]["replicate_repeat_set_sha256_by_seed"]
    assert audit["shared_skeleton"]["replicate_repeat_sets_distinct"] is True
    assert len(set(repeat_hashes.values())) == len(ocsc.PAIRED_SEEDS)


def test_repeat_set_identity_rejects_same_members_in_different_orders() -> None:
    base = {"skeleton-{:04d}".format(index) for index in range(ocsc.REPEATED_PACKS)}
    with pytest.raises(ocsc.ContractError, match="reuse one repeat-set identity"):
        ocsc.assert_distinct_repeat_sets(
            {str(seed): set(reversed(sorted(base))) for seed in ocsc.PAIRED_SEEDS}
        )


def test_execution_contract_integer_gates_and_cpu_only_status(
    generated: dict[str, Path | list[dict]],
) -> None:
    bundle = Path(generated["bundle"])
    audit = json.loads((bundle / "audit_report.json").read_text(encoding="ascii"))
    commitments = json.loads((bundle / "commitments.json").read_text(encoding="ascii"))
    contract = commitments["future_execution_contract"]
    assert contract["parent_checkpoint"]["sha256"] == ocsc.PARENT_CHECKPOINT_SHA256
    assert contract["model"]["batch_shape"] == [8, 256]
    assert contract["model"]["parameter_and_activation_dtype"] == "bfloat16"
    assert contract["optimizer"]["Muon"]["learning_rate"] == {
        "numerator": 1,
        "denominator": 50,
    }
    assert contract["optimizer"]["AdamW"]["betas"] == [
        {"numerator": 9, "denominator": 10},
        {"numerator": 19, "denominator": 20},
    ]
    assert contract["replay_kl"]["direction"] == "KL(parent||run_cell)"
    assert contract["replay_kl"]["vocabulary"] == "full tokenizer vocabulary"
    assert contract["replay_kl"]["coefficient"] == {
        "numerator": 1,
        "denominator": 10,
    }
    assert contract["relational_loss"]["coefficient"] == {
        "numerator": 1,
        "denominator": 4,
    }
    assert "M10" in contract["relational_loss"]["factorial_cells"]
    assert contract["relational_loss"]["initial_suffix_invariance_activation"] == (
        "always zero and excluded"
    )
    assert contract["randomness"]["batch_permutation_must_differ_across_seeds"] is True
    assert contract["randomness"]["pooled_pseudoreplication_forbidden"] is True
    assert contract["execution_authorized"] is False
    consumer = contract["consumer_interface"]
    assert consumer["status"] == "interface-only-consumers-unimplemented"
    assert consumer["execution_authorized"] is False
    assert consumer["consumer_compatibility_claimed"] is False
    assert (
        consumer["source_binding"]["ocsc_source_manifest_sha256"]
        == audit["source_manifest"]["payload_sha256"]
    )
    assert consumer["source_binding"]["trainer_source"]["sha256"] is None
    assert consumer["source_binding"]["evaluator_source"]["sha256"] is None
    assert consumer["trainer_request"]["consumed_bundle_files"] == list(
        ocsc.ARTIFACT_NAMES
    )
    assert consumer["evaluation_report"]["pooled_rows_authorized"] is False
    assert consumer["evaluation_report"]["missing_slice_authorized"] is False
    assert consumer["enablement"]["current_gate"] == "NO-GO"
    gates = audit["evaluation_gates"]
    assert gates["development"]["replication_policy"] == {
        "unit": "paired seed",
        "seed_count": 3,
        "distinct_stratified_batch_permutation_required": True,
        "row_level_pooling_as_independent_replication_forbidden": True,
        "report_each_seed_before_any_descriptive_summary": True,
    }
    primary = gates["development"]["primary_local_effect_each_seed"]
    assert primary["denominator"] == 400
    assert primary["M11_minus_M01_minimum"] == 8
    assert primary["M11_minus_M00_minimum"] == 8
    assert primary["initial_suffix_invariance_rows_in_numerator"] == 0
    assert primary["pooled_row_numerator_authorized"] is False
    carry_gate = gates["development"]["carry_canonicalization_package_board"]
    assert carry_gate["base_sites"] == 170
    assert carry_gate["carry_pairs_per_package_condition"] == 170
    assert carry_gate["prompt_presentations"] == 680
    assert set(carry_gate["slice_geometry"].values()) == {17}
    package = carry_gate["compound_canonicalization_package"]
    assert package["four_prompts_per_base_site"] is True
    assert package["source_only_effect_identified"] is False
    assert "SCERT" in package["source_specific_attribution_deferred_to"]
    assert carry_gate["noncompensatory_gates_each_M11_seed"] == {
        "history_retained_package_target_switch": {
            "minimum": 166,
            "denominator": 170,
        },
        "fresh_current_state_package_target_switch": {
            "minimum": 166,
            "denominator": 170,
        },
        "canonicalization_package_joint_exact": {
            "minimum": 163,
            "denominator": 170,
        },
        "each_width_true_carry_slice": {"minimum": 16, "denominator": 17},
    }
    assert carry_gate["counterfactual_accuracy_can_satisfy_target_switch_gate"] is False
    assert carry_gate["raw_counterfactual_accuracy_causal_authority"] is False
    assert carry_gate["source_specific_claim_authorized"] is False
    assert gates["hidden"]["direct_overall"] == {
        "minimum": 3_564,
        "denominator": 3_600,
    }
    assert gates["hidden"]["noninitial_paired_carry_target_switch_each_M11_seed"][
        "overall"
    ] == {"minimum": 392, "denominator": 400}
    assert (
        sum(
            gate["denominator"]
            for gate in gates["hidden"]["transition_slices"].values()
        )
        == 2_100
    )
    assert len(gates["hidden"]["serializer_slices"]) == 15
    assert audit["cpu_preregistration_eligible"] is False
    assert audit["cpu_preregistration_review_status"] == (
        "NO-GO-pending-independent-hostile-review"
    )
    assert audit["production_training_eligible"] is False
    assert audit["gpu_execution_authorized"] is False
    assert audit["promotion_authorized"] is False


def contract_leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from contract_leaf_paths(child, prefix + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from contract_leaf_paths(child, prefix + (index,))
    else:
        yield prefix


def mutate_contract_leaf(value):
    if isinstance(value, bool):
        return not value
    if type(value) is int:
        return value + 1
    if isinstance(value, str):
        return value + "-mutated"
    raise AssertionError("unsupported contract leaf type: {!r}".format(value))


def test_evaluation_contract_rejects_every_rehashed_leaf_mutation() -> None:
    contract = ocsc.evaluation_gate_contract()
    diagnostics = contract["frozen_diagnostics"]
    assert diagnostics["replay_v5"]["artifact"] == {
        "path": ocsc.REPLAY_V5_PATH,
        "sha256": ocsc.REPLAY_V5_SHA256,
    }
    assert diagnostics["width_sweep_v2"]["artifact"] == {
        "path": ocsc.WIDTH_SWEEP_V2_PATH,
        "sha256": ocsc.WIDTH_SWEEP_V2_SHA256,
    }
    assert diagnostics["residual_swap"]["artifact"] == {
        "path": ocsc.RESIDUAL_SWAP_PATH,
        "sha256": ocsc.RESIDUAL_SWAP_SHA256,
    }
    assert diagnostics["width_sweep_v2"]["positive_raw_failure_fields_by_width"] == {
        "w2": ["z"],
        "w3": ["z"],
        "w4": ["c"],
        "w5": ["c"],
        "w6": ["c", "r"],
        "w7": ["c"],
        "w8": ["c", "r"],
        "w9": ["c", "r"],
        "w10": ["p", "c", "r", "z"],
    }
    serializer = diagnostics["width_sweep_v2"]["noncompensatory_gates"]
    assert serializer["negative_raw_carry_exact"] == {
        "minimum": 9,
        "denominator": 9,
    }
    assert serializer["negative_serializer_preservation"]["widths"] == list(range(2, 7))
    assert serializer["negative_serializer_transfer"]["widths"] == list(range(7, 11))
    assert diagnostics["residual_swap"]["frozen_parent_observation"][
        "inverted_separation_widths"
    ] == [6]
    assert diagnostics["decision_rule"] == {
        "all_required": True,
        "per_seed_required": True,
        "per_width_required": True,
        "per_polarity_required": True,
        "row_pooling_authorized": False,
        "slice_pooling_authorized": False,
        "cross_gate_compensation_authorized": False,
        "aggregate_score_override_authorized": False,
    }

    unhashed_contract = {
        key: value for key, value in contract.items() if key != "payload_sha256"
    }
    for path in contract_leaf_paths(unhashed_contract):
        mutated = copy.deepcopy(contract)
        parent = mutated
        for component in path[:-1]:
            parent = parent[component]
        parent[path[-1]] = mutate_contract_leaf(parent[path[-1]])
        mutated.pop("payload_sha256")
        mutated = ocsc.with_payload_hash(mutated, "payload_sha256")
        with pytest.raises(ocsc.ContractError, match="evaluation gate contract"):
            ocsc.validate_evaluation_gate_contract(mutated)
    mutated_hash = copy.deepcopy(contract)
    mutated_hash["payload_sha256"] = "0" * 64
    with pytest.raises(ocsc.ContractError, match="evaluation gate contract"):
        ocsc.validate_evaluation_gate_contract(mutated_hash)


def hidden_validation_context(
    generated: dict[str, Path | list[dict]],
) -> tuple[ocsc.FrozenTokenizer, dict]:
    tokenizer = ocsc.FrozenTokenizer(Path(generated["tokenizer"]), "test")
    commitment, _ = ocsc.load_confirmation_commitment(Path(generated["confirmation"]))
    return tokenizer, commitment


def rebuild_hidden_row(tokenizer: ocsc.FrozenTokenizer, row: dict, state: dict) -> dict:
    return hidden_opening_row(
        tokenizer,
        row["board_id"],
        row["ordinal"],
        row["row_id"],
        row["kind"],
        state,
        site_id=row["site_id"],
        role=row["role"],
        reachability=row["reachability"],
        serializer_slice=row["serializer_slice"],
        pair_id=row["pair_id"],
        carry_pair_id=row["carry_pair_id"],
        prefix_pair_id=row["prefix_pair_id"],
        endpoint=row["endpoint"],
        prefix_variant=row["prefix_variant"],
        intervention_field=row["intervention_field"],
        intervention_position=row["intervention_position"],
        orientation=row["orientation"],
    )


def verify_hidden(generated: dict[str, Path | list[dict]]) -> dict:
    return ocsc.verify_hidden_opening(
        Path(generated["opening"]),
        Path(generated["confirmation"]),
        Path(generated["bundle"]),
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["custodian_opening"]),
        Path(generated["publication_commitment"]),
        Path(generated["independent_review_receipt"]),
    )


def test_hidden_authenticated_opening_and_exact_site_contract(
    generated: dict[str, Path | list[dict]],
) -> None:
    result = verify_hidden(generated)
    assert result["verified"] is True
    assert result["full_bundle_verified"] is True
    assert result["rows"] == 3_600
    assert result["transition_sites"] == 650
    assert result["serializer_sites"] == 250
    rows = list(generated["hidden_rows"])
    transitions = [row for row in rows if row["kind"] == "transition"]
    serializers = [row for row in rows if row["kind"] == "serializer"]
    assert len(transitions) == 2_100 and len(serializers) == 1_500
    assert len({row["semantic_signature_sha256"] for row in rows}) == 3_600
    assert all(row["scoring_contract"]["endpoint_token_ids"] for row in rows)
    assert all(row["scoring_contract"]["prediction_positions"] for row in rows)


def hidden_semantic_record_map(rows: list[dict]) -> dict:
    by_site = defaultdict(list)
    for row in rows:
        if row["kind"] == "transition":
            by_site[row["site_id"]].append(row)
    result = defaultdict(list)
    for site_id, site_rows in by_site.items():
        natural_row = next(
            row for row in site_rows if row["reachability"] == "reachable"
        )
        state = ocsc.independent_parse_state(natural_row["state"])
        role = natural_row["role"]
        endpoint_tuples = []
        for row in site_rows:
            if not (
                (role == "initial" and row["endpoint"] == "anchor")
                or (role != "initial" and row["prefix_variant"] == "anchor")
            ):
                continue
            endpoint = ocsc.independent_parse_state(row["state"])
            target = ocsc.independent_apply_microstep(endpoint)
            endpoint_tuples.append(
                (
                    endpoint["op"],
                    endpoint["p"],
                    endpoint["c"],
                    int(endpoint["a"][endpoint["p"]]),
                    int(endpoint["b"][endpoint["p"]]),
                    target["r"][endpoint["p"]],
                    target["c"],
                )
            )
        result[(state["w"], role)].append(
            {
                "site_id": site_id,
                "operation": state["op"],
                "position": state["p"],
                "left_digit": int(state["a"][state["p"]]),
                "right_digit": int(state["b"][state["p"]]),
                "natural_tuple": (
                    state["op"],
                    state["p"],
                    state["c"],
                    int(state["a"][state["p"]]),
                    int(state["b"][state["p"]]),
                ),
                "anchor_endpoint_tuples": endpoint_tuples,
            }
        )
    return result


def test_hidden_semantic_coverage_rejects_operation_position_digit_and_tuple_collapse(
    generated: dict[str, Path | list[dict]],
) -> None:
    records = hidden_semantic_record_map(list(generated["hidden_rows"]))
    summary = ocsc._validate_hidden_transition_semantic_record_map(records)
    assert summary["w7:interior"]["sites"] == 40

    operation_collapse = copy.deepcopy(records)
    for row in operation_collapse[(3, "interior")]:
        row["operation"] = "add"
    with pytest.raises(ocsc.ContractError, match="operation coverage collapsed"):
        ocsc._validate_hidden_transition_semantic_record_map(operation_collapse)

    position_collapse = copy.deepcopy(records)
    for row in position_collapse[(7, "interior")]:
        row["position"] = 1
    with pytest.raises(ocsc.ContractError, match="position coverage collapsed"):
        ocsc._validate_hidden_transition_semantic_record_map(position_collapse)

    digit_collapse = copy.deepcopy(records)
    for row in digit_collapse[(4, "initial")]:
        row["left_digit"] = 0
    with pytest.raises(ocsc.ContractError, match="active-operand coverage collapsed"):
        ocsc._validate_hidden_transition_semantic_record_map(digit_collapse)

    tuple_collapse = copy.deepcopy(records)
    first, second = tuple_collapse[(5, "terminal_add")][:2]
    second["natural_tuple"] = first["natural_tuple"]
    second["anchor_endpoint_tuples"] = first["anchor_endpoint_tuples"]
    with pytest.raises(ocsc.ContractError, match="local-transition tuples collapsed"):
        ocsc._validate_hidden_transition_semantic_record_map(tuple_collapse)


def test_hidden_transition_two_position_mutation_and_leaf_rejections(
    generated: dict[str, Path | list[dict]],
) -> None:
    tokenizer, commitment = hidden_validation_context(generated)
    rows = [dict(row) for row in list(generated["hidden_rows"])]

    reordered = [dict(row) for row in rows]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ocsc.ContractError, match="identity/type"):
        ocsc.validate_hidden_opening_rows(reordered, tokenizer, commitment)

    duplicate = [dict(row) for row in rows]
    duplicate[1] = dict(duplicate[0])
    with pytest.raises(ocsc.ContractError, match="identity/type"):
        ocsc.validate_hidden_opening_rows(duplicate, tokenizer, commitment)

    malformed = [dict(row) for row in rows]
    malformed[0]["extra"] = 1
    with pytest.raises(ocsc.ContractError, match="row key mismatch"):
        ocsc.validate_hidden_opening_rows(malformed, tokenizer, commitment)

    noncanonical_state = [dict(row) for row in rows]
    noncanonical_state[0]["state"] = " " + noncanonical_state[0]["state"]
    with pytest.raises(ocsc.ContractError, match="invalid or noncanonical"):
        ocsc.validate_hidden_opening_rows(noncanonical_state, tokenizer, commitment)

    leading_zero_state = [dict(row) for row in rows]
    leading_zero_state[0]["state"] = leading_zero_state[0]["state"].replace(
        ";w=", ";w=0", 1
    )
    with pytest.raises(ocsc.ContractError, match="invalid or noncanonical"):
        ocsc.validate_hidden_opening_rows(leading_zero_state, tokenizer, commitment)

    unreachable_anchor = [dict(row) for row in rows]
    unreachable_site = next(
        row["site_id"]
        for row in unreachable_anchor
        if row["kind"] == "transition"
        and row["role"] != "initial"
        and row["position"] >= 2
    )
    site_rows = [
        row for row in unreachable_anchor if row["site_id"] == unreachable_site
    ]
    corruption_position = next(
        position
        for position in range(site_rows[0]["position"])
        if position != site_rows[0]["intervention_position"]
    )
    for index, row in enumerate(unreachable_anchor):
        if row["site_id"] != unreachable_site:
            continue
        state = ocsc.parse_state(row["state"])
        tape = list(state["r"])
        tape[corruption_position] = str((int(tape[corruption_position]) + 3) % 10)
        state["r"] = "".join(tape)
        unreachable_anchor[index] = rebuild_hidden_row(tokenizer, row, state)
    with pytest.raises(ocsc.ContractError, match="anchor is not solver-reachable"):
        ocsc.validate_hidden_opening_rows(unreachable_anchor, tokenizer, commitment)

    target_index = next(
        index
        for index, row in enumerate(rows)
        if row["kind"] == "transition"
        and row["prefix_variant"] == "intervention"
        and row["endpoint"] == "c0"
        and row["position"] >= 2
    )
    target = rows[target_index]
    anchor = next(
        row
        for row in rows
        if row["site_id"] == target["site_id"]
        and row["prefix_variant"] == "anchor"
        and row["endpoint"] == "c0"
    )
    state = ocsc.parse_state(anchor["state"])
    tape = list(state["r"])
    tape[0] = str((int(tape[0]) + 1) % 10)
    tape[1] = str((int(tape[1]) + 2) % 10)
    state["r"] = "".join(tape)
    rows[target_index] = rebuild_hidden_row(tokenizer, target, state)
    carry_mate_index = next(
        index
        for index, row in enumerate(rows)
        if row["site_id"] == target["site_id"]
        and row["prefix_variant"] == "intervention"
        and row["endpoint"] == "c1"
    )
    carry_mate = rows[carry_mate_index]
    carry_mate_state = dict(state)
    carry_mate_state["c"] = 1
    rows[carry_mate_index] = rebuild_hidden_row(tokenizer, carry_mate, carry_mate_state)
    with pytest.raises(ocsc.ContractError, match="singleton"):
        ocsc.validate_hidden_opening_rows(rows, tokenizer, commitment)

    canonical_rows = [
        ocsc.canonical_json_bytes(row) for row in list(generated["hidden_rows"])
    ]
    assert ocsc.hidden_merkle_root(canonical_rows) == commitment["merkle_root"]
    tampered = list(canonical_rows)
    tampered[0] = tampered[0].replace(b"hidden-transition", b"tamper-transition", 1)
    assert ocsc.hidden_merkle_root(tampered) != commitment["merkle_root"]
    valid_audit = ocsc.validate_hidden_opening_rows(
        list(generated["hidden_rows"]), tokenizer, commitment
    )
    with pytest.raises(ocsc.ContractError, match="overlaps train"):
        ocsc.assert_hidden_disjoint_from_committed_sets(
            valid_audit,
            [
                {
                    "normalized_prompt_sha256": next(iter(valid_audit["seen_prompts"])),
                    "semantic_signature_sha256": "0" * 64,
                }
            ],
        )
    secret_row = next(
        row
        for row in read_jsonl(Path(generated["registry"]))
        if row["use"] == "secret_confirmation"
    )
    secret_overlap = {
        "normalized_prompt_sha256": secret_row["normalized_prompt_sha256"],
        "semantic_signature_sha256": next(iter(valid_audit["seen_signatures"])),
    }
    with pytest.raises(ocsc.ContractError, match="confirmation"):
        ocsc.assert_hidden_disjoint_from_committed_sets(valid_audit, [secret_overlap])


def test_hidden_serializer_collapse_palindrome_and_operand_rejections(
    generated: dict[str, Path | list[dict]],
) -> None:
    tokenizer, commitment = hidden_validation_context(generated)
    original = [dict(row) for row in list(generated["hidden_rows"])]
    serializer_indices = [
        index for index, row in enumerate(original) if row["kind"] == "serializer"
    ]

    collapsed = [dict(row) for row in original]
    by_width_slice = defaultdict(list)
    for index in serializer_indices:
        row = collapsed[index]
        by_width_slice[(row["width"], row["serializer_slice"])].append(index)
    for indices in by_width_slice.values():
        candidate_tapes = [
            ocsc.parse_state(collapsed[index]["state"])["r"] for index in indices
        ]
        tape = next(
            candidate
            for candidate in candidate_tapes
            if all(
                candidate
                != ocsc.state_at(
                    state["op"],
                    ocsc.value_lsf(state["a"]),
                    ocsc.value_lsf(state["b"]),
                    state["w"],
                    state["w"],
                )["r"]
                for state in (
                    ocsc.parse_state(collapsed[index]["state"]) for index in indices
                )
            )
        )
        for index in indices:
            state = ocsc.parse_state(collapsed[index]["state"])
            state["r"] = tape
            collapsed[index] = rebuild_hidden_row(tokenizer, collapsed[index], state)
    assert (
        len(
            {
                row["semantic_signature_sha256"]
                for row in collapsed
                if row["kind"] == "serializer"
            }
        )
        == 15
    )
    with pytest.raises(ocsc.ContractError, match="duplicate hidden semantic signature"):
        ocsc.validate_hidden_opening_rows(collapsed, tokenizer, commitment)

    palindrome = [dict(row) for row in original]
    pair = next(
        [row for row in palindrome if row["pair_id"] == pair_id]
        for pair_id in {
            row["pair_id"] for row in palindrome if row["kind"] == "serializer"
        }
        if next(row for row in palindrome if row["pair_id"] == pair_id)[
            "serializer_slice"
        ]
        == "add_c0"
    )
    width = pair[0]["width"]
    palindrome_tape = "1" + "2" * (width - 2) + "1"
    for row in pair:
        index = palindrome.index(row)
        state = ocsc.parse_state(row["state"])
        state["r"] = palindrome_tape
        palindrome[index] = rebuild_hidden_row(tokenizer, row, state)
    with pytest.raises(ocsc.ContractError, match="palindrome|reversal"):
        ocsc.validate_hidden_opening_rows(palindrome, tokenizer, commitment)

    different_operands = [dict(row) for row in original]
    pair_id = next(
        row["pair_id"]
        for row in different_operands
        if row["kind"] == "serializer"
        and row["serializer_slice"] == "add_c0"
        and row["orientation"] == "forward"
    )
    endpoint_index = next(
        index
        for index, row in enumerate(different_operands)
        if row["pair_id"] == pair_id and row["orientation"] == "reverse"
    )
    endpoint = different_operands[endpoint_index]
    state = ocsc.parse_state(endpoint["state"])
    digits = list(state["a"])
    digits[0] = str((int(digits[0]) + 1) % 10)
    state["a"] = "".join(digits)
    different_operands[endpoint_index] = rebuild_hidden_row(tokenizer, endpoint, state)
    with pytest.raises(ocsc.ContractError, match="operands or reversal"):
        ocsc.validate_hidden_opening_rows(different_operands, tokenizer, commitment)


def test_hidden_custody_rejects_self_rehash_substitute_registry_and_bad_opening(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    tampered_root = tmp_path / "tampered-publication-root"
    tampered_root.mkdir()
    tampered_path = tampered_root / "prepublication_commitment.json"
    document = json.loads(
        Path(generated["publication_commitment"]).read_text(encoding="ascii")
    )
    document["signature_hex"] = (
        "0" if document["signature_hex"][0] != "0" else "1"
    ) + document["signature_hex"][1:]
    tampered_path.write_bytes(ocsc.canonical_json_bytes(document, newline=True))
    freeze_custody_root(tampered_path)
    with pytest.raises(ocsc.ContractError, match="signature failed"):
        ocsc.verify_hidden_opening(
            Path(generated["opening"]),
            Path(generated["confirmation"]),
            Path(generated["bundle"]),
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["custodian_opening"]),
            tampered_path,
            Path(generated["independent_review_receipt"]),
        )

    substitute_root = tmp_path / "substitute-registry-root"
    substitute_root.mkdir()
    substitute = substitute_root / "prompt_registry.jsonl"
    shutil.copyfile(Path(generated["registry"]), substitute)
    freeze_custody_root(substitute)
    with pytest.raises(ocsc.ContractError, match="request/source mismatch"):
        ocsc.verify_hidden_opening(
            Path(generated["opening"]),
            Path(generated["confirmation"]),
            Path(generated["bundle"]),
            Path(generated["tokenizer"]),
            substitute,
            Path(generated["custodian_opening"]),
            Path(generated["publication_commitment"]),
            Path(generated["independent_review_receipt"]),
        )

    bad_root = tmp_path / "bad-custodian-root"
    bad_root.mkdir()
    bad_path, bad_document = make_custodian_opening(
        bad_root / "custodian_opening.json",
        list(generated["hidden_rows"])[0]["board_id"],
    )
    bad_document["nonce_hex"] = "f" * 64
    bad_path.write_bytes(ocsc.canonical_json_bytes(bad_document, newline=True))
    freeze_custody_root(bad_path)
    commitment, _ = ocsc.load_confirmation_commitment(Path(generated["confirmation"]))
    with pytest.raises(ocsc.ContractError, match="authentication failed"):
        ocsc.load_custodian_opening(bad_path, commitment)

    late_root = tmp_path / "late-publication-root"
    late_root.mkdir()
    late_path = freeze_custody_root(
        make_publication_commitment(
            late_root / "prepublication_commitment.json",
            Path(generated["bundle"]),
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["confirmation"]),
        )
    )
    late_request = ocsc.publication_commitment_request(
        "test",
        Path(generated["bundle"]),
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        0,
    )
    with pytest.raises(ocsc.ContractError, match="not consumed before output"):
        ocsc.load_publication_commitment(
            late_path, late_request, "test", require_unpublished=True
        )

    copied_bundle = tmp_path / "self-rehashed-bundle"
    shutil.copytree(Path(generated["bundle"]), copied_bundle)
    os.chmod(copied_bundle, 0o555)
    with pytest.raises(ocsc.ContractError, match="request/source mismatch"):
        ocsc.verify_hidden_opening(
            Path(generated["opening"]),
            Path(generated["confirmation"]),
            copied_bundle,
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["custodian_opening"]),
            Path(generated["publication_commitment"]),
            Path(generated["independent_review_receipt"]),
        )

    wrong_source_root = tmp_path / "wrong-source-publication-root"
    wrong_source_root.mkdir()
    wrong_request = dict(late_request)
    wrong_request["source_manifest"] = json.loads(
        json.dumps(late_request["source_manifest"])
    )
    wrong_request["source_manifest"]["sources"]["train/digitwise_protocol.py"][
        "sha256"
    ] = "0" * 64
    wrong_request.pop("payload_sha256")
    wrong_request = ocsc.with_payload_hash(wrong_request, "payload_sha256")
    wrong_source_path = freeze_custody_root(
        write_signed_publication_commitment(
            wrong_source_root / "prepublication_commitment.json", wrong_request
        )
    )
    with pytest.raises(ocsc.ContractError, match="request/source mismatch"):
        ocsc.load_publication_commitment(
            wrong_source_path, late_request, "test", require_unpublished=False
        )


def test_build_receipt_recheck_rejects_forgery_and_signed_input_drift(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    bundle = Path(generated["bundle"])
    request = ocsc.publication_commitment_request(
        "test",
        bundle,
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        0,
    )
    receipt = ocsc.load_publication_commitment(
        Path(generated["publication_commitment"]),
        request,
        "test",
        require_unpublished=False,
    )
    forged = dict(receipt)
    forged["physical_sha256"] = "0" * 64
    with pytest.raises(ocsc.ContractError, match="forged or drifted"):
        ocsc.revalidate_publication_receipt(
            forged,
            "test",
            bundle,
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["confirmation"]),
            0,
            require_unpublished=False,
        )

    drift_registry_root = tmp_path / "drift-registry-root"
    drift_registry_root.mkdir()
    drift_registry = drift_registry_root / "prompt_registry.jsonl"
    shutil.copyfile(Path(generated["registry"]), drift_registry)
    freeze_custody_root(drift_registry)
    future_bundle = tmp_path / "future-bundle"
    drift_publication_root = tmp_path / "drift-publication-root"
    drift_publication_root.mkdir()
    drift_commitment = freeze_custody_root(
        make_publication_commitment(
            drift_publication_root / "prepublication_commitment.json",
            future_bundle,
            Path(generated["tokenizer"]),
            drift_registry,
            Path(generated["confirmation"]),
        )
    )
    signed_request = ocsc.publication_commitment_request(
        "test",
        future_bundle,
        Path(generated["tokenizer"]),
        drift_registry,
        Path(generated["confirmation"]),
        0,
    )
    signed_receipt = ocsc.load_publication_commitment(
        drift_commitment,
        signed_request,
        "test",
        require_unpublished=True,
    )
    original_registry = drift_registry.read_bytes()
    os.chmod(drift_registry_root, 0o755)
    os.chmod(drift_registry, 0o644)
    drift_registry.write_bytes(original_registry + b"\n")
    freeze_custody_root(drift_registry)
    with pytest.raises(ocsc.ContractError, match="request/source mismatch"):
        ocsc.revalidate_publication_receipt(
            signed_receipt,
            "test",
            future_bundle,
            Path(generated["tokenizer"]),
            drift_registry,
            Path(generated["confirmation"]),
            0,
            require_unpublished=True,
        )


def test_independent_review_binds_full_closure_and_test_key_cannot_produce(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    artifacts = {
        name: (Path(generated["bundle"]) / name).read_bytes()
        for name in ocsc.ARTIFACT_NAMES
    }
    test_request = ocsc.publication_commitment_request(
        "test",
        Path(generated["bundle"]),
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        0,
    )
    review_request = ocsc.independent_review_request(test_request, artifacts)
    assert set(review_request["reviewed_source_bytes"]) == set(
        ocsc.REVIEWED_SOURCE_PATHS
    )
    assert review_request["reviewed_oracle"] == {
        "path": ocsc.ORACLE_SOURCE_PATH,
        "bytes": ocsc.DIGITWISE_PROTOCOL_BYTES,
        "sha256": ocsc.DIGITWISE_PROTOCOL_SHA256,
    }
    assert review_request["tokenizer"] == test_request["inputs"]["tokenizer"]
    assert (
        review_request["prompt_registry"] == test_request["inputs"]["prompt_registry"]
    )
    assert (
        review_request["hidden_commitment"]
        == test_request["inputs"]["secret_confirmation_commitment"]
    )
    assert review_request["publication_request"] == test_request
    assert set(review_request["expected_output_identity"]["files"]) == set(
        ocsc.ARTIFACT_NAMES
    )

    production_output = tmp_path / "production-output"
    production_request = ocsc.publication_commitment_request(
        "production",
        production_output,
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        0,
    )
    production_review_request = ocsc.independent_review_request(
        production_request, artifacts
    )
    root = tmp_path / "production-review-root"
    root.mkdir()
    test_signed_path = freeze_custody_root(
        write_signed_independent_review_receipt(
            root / "independent_review_receipt.json",
            production_review_request,
        )
    )
    with pytest.raises(ocsc.ContractError, match="trust root is not configured"):
        ocsc.load_independent_review_receipt(
            test_signed_path,
            production_review_request,
            "production",
        )


def test_publish_rejects_ten_arbitrary_bytes_after_signed_review(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    output = tmp_path / "arbitrary-output"
    publication_root = tmp_path / "arbitrary-publication-root"
    publication_root.mkdir()
    publication_commitment = freeze_custody_root(
        make_publication_commitment(
            publication_root / "prepublication_commitment.json",
            output,
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["confirmation"]),
        )
    )
    request = ocsc.publication_commitment_request(
        "test",
        output,
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        0,
    )
    publication_receipt = ocsc.load_publication_commitment(
        publication_commitment, request, "test", require_unpublished=True
    )
    arbitrary = {name: b"x" for name in ocsc.ARTIFACT_NAMES}
    review_root = tmp_path / "arbitrary-review-root"
    review_root.mkdir()
    review_path = freeze_custody_root(
        write_signed_independent_review_receipt(
            review_root / "independent_review_receipt.json",
            ocsc.independent_review_request(request, arbitrary),
        )
    )
    review_receipt = ocsc.load_independent_review_receipt(
        review_path,
        ocsc.independent_review_request(request, arbitrary),
        "test",
    )
    stage_name, journal_name = ocsc.publication_staging_names(
        publication_receipt,
        review_receipt,
    )
    lease_name = ocsc.publication_lease_name(
        publication_receipt,
        review_receipt,
    )
    with pytest.raises(ocsc.ContractError, match="invalid JSON in bundle manifest"):
        ocsc.publish_bundle(
            output,
            arbitrary,
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=publication_receipt,
            independent_review_receipt=review_receipt,
        )
    assert not output.exists()
    retained = ocsc.capture_retained_publication_evidence(
        tmp_path,
        crash_point="stage-fsync-before-rename",
        stage_name=stage_name,
        canonical_name=output.name,
        journal_name=journal_name,
        lease_name=lease_name,
    )
    assert retained["stage_state"] == "retained"
    assert retained["journal_state"] == "retained"
    assert retained["lease_state"] == "retained"


def test_manifest_deterministic_verification_modes_links_and_nonfinite(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    bundle = Path(generated["bundle"])
    result = ocsc.verify_bundle(
        bundle,
        Path(generated["tokenizer"]),
        Path(generated["registry"]),
        Path(generated["confirmation"]),
        Path(generated["publication_commitment"]),
        Path(generated["independent_review_receipt"]),
    )
    assert result["verified"] is True
    manifest = ocsc.load_manifest(bundle)
    assert set(manifest["files"]) == set(ocsc.ARTIFACT_NAMES) - {"manifest.json"}
    assert manifest["inputs"]["prompt_registry_path"] == str(
        Path(generated["registry"]).resolve()
    )
    assert (
        manifest["inputs"]["prompt_registry_bytes"]
        == Path(generated["registry"]).stat().st_size
    )
    assert manifest["inputs"]["prompt_registry_sha256"] == sha256_file(
        Path(generated["registry"])
    )
    assert manifest["inputs"]["prepublication_commitment_sha256"] == sha256_file(
        Path(generated["publication_commitment"])
    )
    assert manifest["source_manifest"]["sources"]["train/digitwise_protocol.py"] == {
        "bytes": ocsc.DIGITWISE_PROTOCOL_BYTES,
        "sha256": ocsc.DIGITWISE_PROTOCOL_SHA256,
    }

    extra = tmp_path / "manifest-extra"
    extra.mkdir()
    shutil.copy2(bundle / "manifest.json", extra / "manifest.json")
    path = extra / "manifest.json"
    os.chmod(path, 0o644)
    document = json.loads(path.read_text(encoding="ascii"))
    document["extra"] = 1
    path.write_bytes(ocsc.pretty_json_bytes(document))
    os.chmod(path, 0o444)
    with pytest.raises(ocsc.ContractError, match="manifest key mismatch"):
        ocsc.load_manifest(extra)

    nonfinite = tmp_path / "manifest-nonfinite"
    nonfinite.mkdir()
    shutil.copy2(bundle / "manifest.json", nonfinite / "manifest.json")
    path = nonfinite / "manifest.json"
    os.chmod(path, 0o644)
    payload = path.read_text(encoding="ascii").replace(
        '"pad_token_id": 0', '"pad_token_id": NaN'
    )
    path.write_text(payload, encoding="ascii")
    os.chmod(path, 0o444)
    with pytest.raises(ocsc.ContractError, match="invalid JSON|nonfinite"):
        ocsc.load_manifest(nonfinite)

    packs_path = bundle / "packs.jsonl"
    os.chmod(packs_path, 0o644)
    try:
        with pytest.raises(
            ocsc.ContractError,
            match="bundle artifact identity/mode mismatch",
        ):
            ocsc.verify_bundle(
                bundle,
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["confirmation"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
    finally:
        os.chmod(packs_path, 0o444)

    backup = tmp_path / "packs-original.jsonl"
    os.chmod(bundle, 0o755)
    packs_path.rename(backup)
    os.link(backup, packs_path)
    os.chmod(bundle, 0o555)
    try:
        with pytest.raises(
            ocsc.ContractError,
            match="bundle artifact identity/mode mismatch",
        ):
            ocsc.verify_bundle(
                bundle,
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["confirmation"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
    finally:
        os.chmod(bundle, 0o755)
        packs_path.unlink()
        backup.rename(packs_path)
        os.chmod(bundle, 0o555)

    schedule_path = bundle / "schedule.jsonl"
    manifest_path = bundle / "manifest.json"
    original_schedule = schedule_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    tampered_schedule = original_schedule.replace(
        b'"pack_id":"', b'"pack_id":"tampered-', 1
    )
    try:
        os.chmod(schedule_path, 0o644)
        schedule_path.write_bytes(tampered_schedule)
        os.chmod(schedule_path, 0o444)
        os.chmod(manifest_path, 0o644)
        document = json.loads(original_manifest)
        document["files"]["schedule.jsonl"] = {
            "bytes": len(tampered_schedule),
            "sha256": sha256_bytes(tampered_schedule),
        }
        document.pop("payload_sha256")
        document["payload_sha256"] = ocsc.hash_json(document)
        manifest_path.write_bytes(ocsc.pretty_json_bytes(document))
        os.chmod(manifest_path, 0o444)
        with pytest.raises(
            ocsc.ContractError,
            match="independent review receipt request/output mismatch",
        ):
            ocsc.verify_bundle(
                bundle,
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["confirmation"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
        tampered_artifacts = {
            name: (bundle / name).read_bytes() for name in ocsc.ARTIFACT_NAMES
        }
        tampered_request = ocsc.publication_commitment_request(
            "test",
            bundle,
            Path(generated["tokenizer"]),
            Path(generated["registry"]),
            Path(generated["confirmation"]),
            0,
        )
        hostile_review_root = tmp_path / "hostile-review-root"
        hostile_review_root.mkdir()
        hostile_review = freeze_custody_root(
            write_signed_independent_review_receipt(
                hostile_review_root / "independent_review_receipt.json",
                ocsc.independent_review_request(
                    tampered_request,
                    tampered_artifacts,
                ),
            )
        )
        with pytest.raises(ocsc.ContractError, match="deterministic reconstruction"):
            ocsc.verify_bundle(
                bundle,
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["confirmation"]),
                Path(generated["publication_commitment"]),
                hostile_review,
            )
    finally:
        os.chmod(schedule_path, 0o644)
        schedule_path.write_bytes(original_schedule)
        os.chmod(schedule_path, 0o444)
        os.chmod(manifest_path, 0o644)
        manifest_path.write_bytes(original_manifest)
        os.chmod(manifest_path, 0o444)


def test_custody_snapshot_rejects_parent_swap_during_child_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custody_root = tmp_path / "race-custody-root"
    custody_root.mkdir()
    trusted = custody_root / "commitment.json"
    trusted.write_bytes(b"trusted\n")
    freeze_custody_root(trusted)
    moved_root = tmp_path / "race-custody-root-moved"
    original_open = ocsc.os.open
    triggered = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal triggered
        if dir_fd is not None and path == trusted.name and not triggered:
            triggered = True
            custody_root.rename(moved_root)
            custody_root.mkdir()
            substitute = custody_root / trusted.name
            substitute.write_bytes(b"substitute\n")
            freeze_custody_root(substitute)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(ocsc.os, "open", racing_open)
    with pytest.raises(ocsc.ContractError, match="changed during the operation"):
        ocsc.read_file_snapshot(
            trusted,
            "raced commitment",
            exact_mode=0o444,
            custody_root=True,
        )
    assert triggered
    assert (moved_root / trusted.name).read_bytes() == b"trusted\n"
    assert trusted.read_bytes() == b"substitute\n"


def test_publication_rejects_output_parent_swap_and_retains_evidence(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "signed-output-parent"
    output_parent.mkdir()
    output = output_parent / "bundle"
    case_root = tmp_path / "race-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    _, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    lease_name = ocsc.publication_lease_name(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    moved_parent = tmp_path / "signed-output-parent-moved"
    original_path_rename = ocsc.os.rename
    original_publish_rename = ocsc._rename_directory_noreplace
    triggered = False

    def racing_rename(src, dst, src_dir_fd, dst_dir_fd):
        nonlocal triggered
        if dst == output.name and not triggered:
            triggered = True
            original_path_rename(output_parent, moved_parent)
            output_parent.mkdir()
        return original_publish_rename(src, dst, src_dir_fd, dst_dir_fd)

    monkeypatch.setattr(ocsc, "_rename_directory_noreplace", racing_rename)
    monkeypatch.setattr(
        ocsc, "_strict_verify_publication_tree", lightweight_publication_verifier
    )
    with pytest.raises(ocsc.ContractError, match="changed during the operation"):
        ocsc.publish_bundle(
            output,
            case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=case["publication_receipt"],
            independent_review_receipt=case["independent_review_receipt"],
        )
    assert triggered
    assert not output.exists()
    retained_output = moved_parent / output.name
    assert retained_output.is_dir()
    assert (moved_parent / journal_name).is_file()
    assert (moved_parent / lease_name).is_file()


def test_publication_atomic_noreplace_rejects_late_output_creation(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "late-output-parent"
    output_parent.mkdir()
    output = output_parent / "bundle"
    case_root = tmp_path / "late-output-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    lease_name = ocsc.publication_lease_name(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    original_publish_rename = ocsc._rename_directory_noreplace
    triggered = False

    def racing_rename(src, dst, src_dir_fd, dst_dir_fd):
        nonlocal triggered
        if dst == output.name and not triggered:
            os.mkdir(dst, mode=0o700, dir_fd=dst_dir_fd)
            triggered = True
        return original_publish_rename(src, dst, src_dir_fd, dst_dir_fd)

    monkeypatch.setattr(ocsc, "_rename_directory_noreplace", racing_rename)
    monkeypatch.setattr(
        ocsc, "_strict_verify_publication_tree", lightweight_publication_verifier
    )
    with pytest.raises(ocsc.ContractError, match="refusing to overwrite"):
        ocsc.publish_bundle(
            output,
            case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=case["publication_receipt"],
            independent_review_receipt=case["independent_review_receipt"],
        )
    assert triggered
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert (output_parent / stage_name).is_dir()
    assert (output_parent / journal_name).is_file()
    assert (output_parent / lease_name).is_file()


def test_publication_rejects_postrename_child_replacement(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_parent = tmp_path / "postrename-output-parent"
    output_parent.mkdir()
    output = output_parent / "bundle"
    moved_bundle = output_parent / "moved-owned-bundle"
    case_root = tmp_path / "postrename-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    parent_identity = output_parent.stat()
    original_fsync = ocsc.os.fsync
    triggered = False

    def racing_fsync(descriptor: int) -> None:
        nonlocal triggered
        metadata = os.fstat(descriptor)
        if (
            not triggered
            and (metadata.st_dev, metadata.st_ino)
            == (parent_identity.st_dev, parent_identity.st_ino)
            and output.is_dir()
        ):
            output.rename(moved_bundle)
            output.mkdir(mode=0o700)
            os.chmod(output, 0o555)
            triggered = True
        original_fsync(descriptor)

    monkeypatch.setattr(ocsc.os, "fsync", racing_fsync)
    monkeypatch.setattr(
        ocsc, "_strict_verify_publication_tree", lightweight_publication_verifier
    )
    with pytest.raises(
        ocsc.ContractError,
        match="bundle directory changed during the operation",
    ):
        ocsc.publish_bundle(
            output,
            case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=case["publication_receipt"],
            independent_review_receipt=case["independent_review_receipt"],
        )
    assert triggered
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert sorted(path.name for path in moved_bundle.iterdir()) == sorted(
        ocsc.ARTIFACT_NAMES
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires process hard-exit")
@pytest.mark.parametrize("crash_point", ocsc.QUALIFICATION_CRASH_POINTS)
def test_exact_qualification_crash_points_retain_evidence_for_restart(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    output = tmp_path / "bundle"
    case_root = tmp_path / "death-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    lease_name = ocsc.publication_lease_name(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    foreign = tmp_path / ".ocsc.partial.foreign-tree"
    foreign.mkdir()
    monkeypatch.setattr(
        ocsc, "_strict_verify_publication_tree", lightweight_publication_verifier
    )

    child = os.fork()
    if child == 0:
        try:
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
                qualification_crash_point=crash_point,
            )
        except BaseException:
            os._exit(79)
        os._exit(80)

    _, status_code = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status_code)
    assert os.WTERMSIG(status_code) == signal.SIGKILL
    journal_expected = crash_point != "stage-created-before-journal"
    assert (tmp_path / journal_name).is_file() is journal_expected
    assert (tmp_path / lease_name).is_file()
    retained = ocsc.capture_retained_publication_evidence(
        tmp_path,
        crash_point=crash_point,
        stage_name=stage_name,
        canonical_name=output.name,
        journal_name=journal_name,
        lease_name=lease_name,
    )
    assert retained["crash_point"] == crash_point
    assert retained["journal_state"] == ("retained" if journal_expected else "absent")
    assert retained["lease_state"] == "retained"
    if crash_point != "canonical-before-parent-fsync":
        stage_before = (tmp_path / stage_name).stat()
        lease_before = (tmp_path / lease_name).stat()
        retained_paths = [
            (tmp_path / stage_name, stage_before),
            (tmp_path / lease_name, lease_before),
        ]
        if journal_expected:
            retained_paths.append(
                (tmp_path / journal_name, (tmp_path / journal_name).stat())
            )
        expected_error = (
            "unauthenticated publication stage collision"
            if not journal_expected
            else "permanently retained"
        )
        with pytest.raises(ocsc.ContractError, match=expected_error):
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
            )
        assert not output.exists()
        for path, before in retained_paths:
            after = path.stat()
            assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    else:
        ocsc.publish_bundle(
            output,
            case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=case["publication_receipt"],
            independent_review_receipt=case["independent_review_receipt"],
        )
        assert output.is_dir()
        assert sorted(path.name for path in output.iterdir()) == sorted(
            ocsc.ARTIFACT_NAMES
        )
        assert not (tmp_path / stage_name).exists()
        assert (tmp_path / journal_name).is_file()
        assert (tmp_path / lease_name).is_file()
    assert foreign.is_dir()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires concurrent processes")
def test_live_publication_lease_prevents_concurrent_stage_recovery(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    output = tmp_path / "bundle"
    case_root = tmp_path / "live-concurrency-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    lease_name = ocsc.publication_lease_name(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.close(release_write)
        signaled = False

        def blocking_verifier(path: Path, directory_fd: int, **kwargs) -> dict:
            nonlocal signaled
            if Path(path).name == stage_name and not signaled:
                os.write(ready_write, b"R")
                if os.read(release_read, 1) != b"G":
                    os._exit(91)
                signaled = True
            return lightweight_publication_verifier(path, directory_fd, **kwargs)

        ocsc._strict_verify_publication_tree = blocking_verifier
        try:
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
            )
        except BaseException:
            os._exit(92)
        os._exit(0)

    os.close(ready_write)
    os.close(release_read)
    try:
        assert os.read(ready_read, 1) == b"R"
        stage_before = (tmp_path / stage_name).stat()
        journal_before = (tmp_path / journal_name).read_bytes()
        lease_before = (tmp_path / lease_name).stat()
        with pytest.raises(
            ocsc.ContractError,
            match="live concurrent publisher holds the publication lease",
        ):
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
            )
        stage_after = (tmp_path / stage_name).stat()
        lease_after = (tmp_path / lease_name).stat()
        assert (stage_before.st_dev, stage_before.st_ino) == (
            stage_after.st_dev,
            stage_after.st_ino,
        )
        assert (lease_before.st_dev, lease_before.st_ino) == (
            lease_after.st_dev,
            lease_after.st_ino,
        )
        assert (tmp_path / journal_name).read_bytes() == journal_before
        os.write(release_write, b"G")
        _, status_code = os.waitpid(child, 0)
        assert os.WIFEXITED(status_code)
        assert os.WEXITSTATUS(status_code) == 0
        assert output.is_dir()
        assert not (tmp_path / stage_name).exists()
        assert (tmp_path / journal_name).is_file()
        assert (tmp_path / lease_name).is_file()
    finally:
        os.close(ready_read)
        os.close(release_write)


def test_source_bound_cli_build_verify_and_two_subprocess_publishers(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
) -> None:
    output = tmp_path / "source-bound-bundle"
    common = [
        "--mode",
        "test",
        "--tokenizer",
        str(generated["tokenizer"]),
        "--prompt-registry",
        str(generated["registry"]),
        "--secret-confirmation-commitment",
        str(generated["confirmation"]),
        "--pad-token-id",
        "0",
    ]
    prepared = run_source_bound(["--prepare-publication-request", str(output), *common])
    request = json.loads(prepared.stdout)
    assert (
        request["source_manifest"]["bootstrap_source_identity"]["source_bound"] is True
    )
    assert request["source_manifest"]["runtime_closure"]["source_bound"] is True

    publication_root = tmp_path / "source-bound-publication-root"
    publication_root.mkdir(mode=0o700)
    publication_commitment = freeze_custody_root(
        write_signed_publication_commitment(
            publication_root / "prepublication_commitment.json",
            request,
        )
    )
    review_preparation = run_source_bound(
        [
            "--prepare-independent-review-request",
            str(output),
            *common,
            "--publication-commitment",
            str(publication_commitment),
        ]
    )
    review_request = json.loads(review_preparation.stdout)
    review_root = tmp_path / "source-bound-review-root"
    review_root.mkdir(mode=0o700)
    independent_review_receipt = freeze_custody_root(
        write_signed_independent_review_receipt(
            review_root / "independent_review_receipt.json",
            review_request,
        )
    )

    primary_control = tmp_path / "primary-control"
    secondary_control = tmp_path / "secondary-control"
    primary_control.mkdir(mode=0o700)
    secondary_control.mkdir(mode=0o700)
    primary_lease_receipt = primary_control / "lease.json"
    primary_result_receipt = primary_control / "result.json"
    release_receipt = primary_control / "release.json"
    secondary_lease_receipt = secondary_control / "lease.json"
    secondary_result_receipt = secondary_control / "result.json"
    receipts = [
        "--publication-commitment",
        str(publication_commitment),
        "--independent-review-receipt",
        str(independent_review_receipt),
    ]
    primary_args = [
        "--qualification-output-dir",
        str(output),
        *common,
        *receipts,
        "--qualification-publisher-id",
        "local-primary-publisher",
        "--qualification-publisher-sequence",
        "1",
        "--qualification-publisher-nonce-hex",
        sha256_bytes(b"local-primary-publisher"),
        "--qualification-lease-receipt-out",
        str(primary_lease_receipt),
        "--qualification-result-receipt-out",
        str(primary_result_receipt),
        "--qualification-release-receipt",
        str(release_receipt),
        "--qualification-release-timeout-seconds",
        "300",
    ]
    process_env = {"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"}
    primary = subprocess.Popen(
        source_bound_command(primary_args),
        cwd=ROOT,
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 300
    while not primary_lease_receipt.exists():
        if primary.poll() is not None:
            stdout, stderr = primary.communicate()
            pytest.fail(
                "primary publisher exited before lease receipt: {} {}".format(
                    stdout, stderr
                )
            )
        if time.monotonic() >= deadline:
            primary.kill()
            primary.communicate()
            pytest.fail("primary publisher lease receipt timed out")
        time.sleep(0.05)
    primary_receipt = ocsc.validate_qualification_publisher_receipt(
        json.loads(primary_lease_receipt.read_text(encoding="ascii"))
    )
    assert primary_receipt["event"] == "lease-acquired"

    secondary_args = [
        "--qualification-output-dir",
        str(output),
        *common,
        *receipts,
        "--qualification-publisher-id",
        "local-secondary-publisher",
        "--qualification-publisher-sequence",
        "1",
        "--qualification-publisher-nonce-hex",
        sha256_bytes(b"local-secondary-publisher"),
        "--qualification-lease-receipt-out",
        str(secondary_lease_receipt),
        "--qualification-result-receipt-out",
        str(secondary_result_receipt),
    ]
    secondary = subprocess.run(
        source_bound_command(secondary_args),
        cwd=ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert secondary.returncode != 0
    assert "live concurrent publisher holds the publication lease" in secondary.stderr
    secondary_receipt = ocsc.validate_qualification_publisher_receipt(
        json.loads(secondary_lease_receipt.read_text(encoding="ascii"))
    )
    assert secondary_receipt["event"] == "live-lease-rejected"
    assert secondary_receipt["publisher_pid"] != primary_receipt["publisher_pid"]
    assert secondary_receipt["lease_record"] == primary_receipt["lease_record"]
    assert not secondary_result_receipt.exists()

    release = ocsc.qualification_release_contract(
        {
            "publisher_id": primary_receipt["publisher_id"],
            "sequence": primary_receipt["sequence"],
            "nonce_hex": primary_receipt["nonce_hex"],
        },
        primary_receipt["lease_record"],
    )
    release_receipt.write_bytes(ocsc.canonical_json_bytes(release, newline=True))
    os.chmod(release_receipt, 0o444)
    primary_stdout, primary_stderr = primary.communicate(timeout=300)
    assert primary.returncode == 0, primary_stderr
    primary_result = json.loads(primary_stdout)
    assert primary_result["generated"] is True
    assert primary_result["qualification_mode"] is True
    completion = ocsc.validate_qualification_publisher_receipt(
        json.loads(primary_result_receipt.read_text(encoding="ascii"))
    )
    assert completion["event"] == "publication-verified"
    assert completion["lease_record"] == primary_receipt["lease_record"]

    verified = run_source_bound(
        [
            "--verify",
            str(output),
            "--tokenizer",
            str(generated["tokenizer"]),
            "--prompt-registry",
            str(generated["registry"]),
            "--secret-confirmation-commitment",
            str(generated["confirmation"]),
            "--publication-commitment",
            str(publication_commitment),
            "--independent-review-receipt",
            str(independent_review_receipt),
        ]
    )
    verification = json.loads(verified.stdout)
    assert verification["verified"] is True
    assert verification["artifact_count"] == len(ocsc.ARTIFACT_NAMES)
    source_manifest = request["source_manifest"]
    qualification_command = source_bound_command(
        ["--verify-linux-lustre-qualification-receipt", "<receipt>"]
    )
    evidence = make_qualification_evidence(
        tmp_path / "qualification-evidence",
        source_manifest["payload_sha256"],
    )
    signing_root = tmp_path / "qualification-signing-root"
    signing_root.mkdir(mode=0o700)
    signing_key = signing_root / "qualification-signing-key.hex"
    signing_key.write_text(
        TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX + "\n",
        encoding="ascii",
    )
    os.chmod(signing_key, 0o400)
    os.chmod(signing_root, 0o555)
    raw_request = ocsc.with_payload_hash(
        {
            "schema": "shohin-ocsc-qualification-raw-evidence-request-v1",
            "reviewer_id": "local-test-reviewer",
            "sequence": 1,
            "nonce_hex": sha256_bytes(b"local-test-qualification"),
            "command": qualification_command,
            "raw_events": evidence["events"],
            "broker_requests": evidence["broker_requests"],
            "broker_receipts": evidence["broker_receipts"],
        },
        "payload_sha256",
    )
    request_root = tmp_path / "qualification-request-root"
    request_root.mkdir(mode=0o700)
    request_path = request_root / "raw-evidence-request.json"
    request_path.write_bytes(ocsc.canonical_json_bytes(raw_request, newline=True))
    os.chmod(request_path, 0o444)
    os.chmod(request_root, 0o555)
    qualification_root = tmp_path / "qualification-receipt-root"
    qualification_root.mkdir(mode=0o700)
    package_result = run_source_bound(
        [
            "--qualification-write-evidence-package",
            str(qualification_root),
            "--qualification-raw-evidence-request",
            str(request_path),
            "--qualification-signing-key",
            str(signing_key),
            "--mode",
            "test",
        ]
    )
    package = json.loads(package_result.stdout)
    assert package["verification"]["event_derivation_verified"] is True
    assert package["verification"]["qualification_authority"] is False
    qualification_receipt = Path(package["receipt_path"])
    qualification_verified = run_source_bound(
        [
            "--verify-linux-lustre-qualification-receipt",
            str(qualification_receipt),
            "--mode",
            "test",
            "--prompt-registry",
            str(generated["registry"]),
            "--secret-confirmation-commitment",
            str(generated["confirmation"]),
        ]
    )
    qualification_result = json.loads(qualification_verified.stdout)
    assert qualification_result["signature_verified"] is True
    assert qualification_result["cross_node_verified"] is True
    assert qualification_result["event_derivation_verified"] is True
    assert qualification_result["qualification_authority"] is False

    forged_unsigned = copy.deepcopy(package["receipt"])
    forged_unsigned.pop("signature_hex")
    forged_unsigned["derived_report"]["summary"]["retained_evidence_count"] += 1
    forged_unsigned["derived_report"].pop("payload_sha256")
    forged_unsigned["derived_report"] = ocsc.with_payload_hash(
        forged_unsigned["derived_report"], "payload_sha256"
    )
    forged_root = tmp_path / "forged-qualification-receipt-root"
    forged_root.mkdir(mode=0o700)
    forged_receipt = freeze_custody_root(
        write_signed_linux_lustre_qualification_receipt(
            forged_root / "qualification.json",
            forged_unsigned,
        )
    )
    forged_result = run_source_bound(
        [
            "--verify-linux-lustre-qualification-receipt",
            str(forged_receipt),
            "--mode",
            "test",
            "--prompt-registry",
            str(generated["registry"]),
            "--secret-confirmation-commitment",
            str(generated["confirmation"]),
        ],
        check=False,
    )
    assert forged_result.returncode != 0
    assert "report or marker was not event-derived" in forged_result.stderr

    with pytest.raises(
        ocsc.ContractError,
        match="production Linux/Lustre qualification trust root is not configured",
    ):
        ocsc.load_linux_lustre_qualification_receipt(
            qualification_receipt,
            json.loads(
                (Path(generated["bundle"]) / "manifest.json").read_text(
                    encoding="ascii"
                )
            )["source_manifest"],
            "production",
        )
    retained_publication_records = [
        path for path in tmp_path.iterdir() if path.name.startswith(".ocsc.partial.")
    ]
    assert any(
        path.name.endswith(".recovery.json") for path in retained_publication_records
    )
    assert any(path.name.endswith(".lease") for path in retained_publication_records)


def test_external_bootstrap_rejects_generator_hash_drift_before_execution() -> None:
    command = source_bound_command(["--help"])
    index = command.index("--generator-sha256") + 1
    command[index] = "0" * 64
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={"LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "generate_orthogonal_carry_serializer_curriculum.py" in result.stderr
    assert "approved hash" in result.stderr


def test_external_bootstrap_executes_only_pinned_source_bytes(tmp_path: Path) -> None:
    checkout = make_fake_source_checkout(tmp_path / "fake-checkout")
    result = subprocess.run(
        fake_source_bound_command(checkout, ["--probe"]),
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "argv": ["--probe"],
        "source_count": len(ocsc_runner.SOURCE_PATHS),
        "runtime_count": external_bootstrap_artifacts()["runtime_fd_count"],
    }


def test_external_runtime_manifest_closes_source_manifest_before_action(
    tmp_path: Path,
) -> None:
    result = run_source_bound(["--print-source-manifest"])
    inspection = json.loads(result.stdout)
    source = inspection["source_manifest"]
    assert source["schema"] == "shohin-ocsc-source-manifest-v4"
    assert source["bootstrap_source_identity"]["source_bound"] is True
    assert source["runtime_closure"]["schema"] == "shohin-ocsc-runtime-closure-v3"
    assert source["runtime_closure"]["pre_attestation_complete"] is True
    assert inspection["qualification_authority"] is False
    assert inspection["publication_authority"] is False

    evidence = make_qualification_evidence(
        tmp_path / "external-action-evidence",
        source["payload_sha256"],
    )
    signing_root = tmp_path / "qualification-signing-custody"
    signing_root.mkdir(mode=0o700)
    signing_key = signing_root / "qualification-signing-key.hex"
    signing_key.write_text(
        TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX + "\n",
        encoding="ascii",
    )
    os.chmod(signing_key, 0o400)
    os.chmod(signing_root, 0o555)
    broker_records = tmp_path / "external-broker-records"
    broker_events = tmp_path / "external-broker-events"
    broker_records.mkdir(mode=0o700)
    broker_events.mkdir(mode=0o700)
    broker_result = run_source_bound(
        [
            "--qualification-broker-transfer-event",
            evidence["transfer"]["destination_path"],
            "--qualification-signing-key",
            str(signing_key),
            "--qualification-broker-record-dir",
            str(broker_records),
            "--qualification-publication-event-dir",
            str(broker_events),
            "--qualification-broker-id",
            "external-production-broker",
            "--qualification-broker-sequence",
            "1",
            "--qualification-previous-request-sha256",
            "0" * 64,
            "--qualification-previous-receipt-sha256",
            "0" * 64,
            "--mode",
            "test",
        ]
    )
    broker = json.loads(broker_result.stdout)
    assert broker["request"]["source_manifest_sha256"] == source["payload_sha256"]
    assert broker["receipt"]["source_manifest_sha256"] == source["payload_sha256"]

    raw_request = ocsc.with_payload_hash(
        {
            "schema": "shohin-ocsc-qualification-raw-evidence-request-v1",
            "reviewer_id": "external-source-test-reviewer",
            "sequence": 1,
            "nonce_hex": sha256_bytes(b"external-source-test-package"),
            "command": ["reviewed-two-host-lustre-qualification-command"],
            "raw_events": evidence["events"],
            "broker_requests": [broker["request"]],
            "broker_receipts": [broker["receipt"]],
        },
        "payload_sha256",
    )
    request_root = tmp_path / "qualification-request-custody"
    request_root.mkdir(mode=0o700)
    request_path = request_root / "raw-evidence-request.json"
    request_path.write_bytes(ocsc.canonical_json_bytes(raw_request, newline=True))
    os.chmod(request_path, 0o444)
    os.chmod(request_root, 0o555)
    package_root = tmp_path / "external-evidence-package"
    package_root.mkdir(mode=0o700)
    package_result = run_source_bound(
        [
            "--qualification-write-evidence-package",
            str(package_root),
            "--qualification-raw-evidence-request",
            str(request_path),
            "--qualification-signing-key",
            str(signing_key),
            "--mode",
            "test",
        ]
    )
    package = json.loads(package_result.stdout)
    assert package["receipt"]["source_manifest_sha256"] == source["payload_sha256"]
    verified = package["verification"]
    assert verified["event_derivation_verified"] is True
    assert verified["qualification_authority"] is False


def test_external_bootstrap_rejects_runner_substitution_before_compile(
    tmp_path: Path,
) -> None:
    checkout = make_fake_source_checkout(tmp_path / "runner-substitution")
    runner = checkout / ocsc_runner.RUNNER_RELATIVE_PATH
    approved_hash = sha256_file(runner)
    marker = checkout / "runner-executed"
    with runner.open("ab") as stream:
        stream.write(
            ("\nopen({!r}, 'wb').write(b'executed')\n".format(str(marker))).encode(
                "ascii"
            )
        )
    result = subprocess.run(
        fake_source_bound_command(
            checkout,
            ["--probe"],
            runner_sha256=approved_hash,
        ),
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert ocsc_runner.RUNNER_RELATIVE_PATH in result.stderr
    assert "approved hash" in result.stderr
    assert not marker.exists()


def test_external_bootstrap_binds_actual_executable_bytes(tmp_path: Path) -> None:
    external = external_bootstrap_artifacts()
    tampered = tmp_path / "tampered-external-bootstrap"
    tampered.write_bytes(Path(external["bootstrap"]).read_bytes() + b"tamper\n")
    tampered.chmod(0o755)
    command = source_bound_command(["--help"], profile="test")
    command[0] = str(tampered)
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "bootstrap executable bytes do not match approved hash" in result.stderr


def test_runner_rejects_direct_execution_and_source_toctou(tmp_path: Path) -> None:
    direct = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(RUNNER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode != 0
    assert "direct runner execution is forbidden" in direct.stderr

    checkout = make_fake_source_checkout(tmp_path / "source-toctou")
    raced = subprocess.run(
        fake_source_bound_command(checkout, ["--replace-tests-path"]),
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert raced.returncode != 0
    assert "attested source" in raced.stderr
    assert "one regular file with one hard link" in raced.stderr

    component_checkout = make_fake_source_checkout(tmp_path / "component-replacement")
    component_race = subprocess.run(
        fake_source_bound_command(component_checkout, ["--replace-pipeline-component"]),
        cwd=component_checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert component_race.returncode != 0
    assert "No such file or directory" in component_race.stderr
    assert "generate_orthogonal_carry_serializer_curriculum.py" in (
        component_race.stderr
    )


def test_external_bootstrap_rejects_preexisting_symlink_aliases(
    tmp_path: Path,
) -> None:
    checkout = make_fake_source_checkout(tmp_path / "real-checkout")
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(checkout, target_is_directory=True)
    aliased_command = fake_source_bound_command(checkout, ["--probe"])
    root_index = aliased_command.index("--checkout-root") + 1
    aliased_command[root_index] = str(alias)
    aliased = subprocess.run(
        aliased_command,
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert aliased.returncode != 0
    assert "checkout root component is not safely openable" in aliased.stderr

    source_checkout = make_fake_source_checkout(tmp_path / "source-alias")
    source = source_checkout / ocsc_runner.SOURCE_PATHS[2]
    target = source.with_name(source.name + ".target")
    source.rename(target)
    source.symlink_to(target.name)
    source_alias = subprocess.run(
        fake_source_bound_command(source_checkout, ["--probe"]),
        cwd=source_checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert source_alias.returncode != 0
    assert ocsc_runner.SOURCE_PATHS[2] in source_alias.stderr
    assert "safely openable" in source_alias.stderr


def test_caller_context_and_preattestation_callable_substitution_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ocsc,
        "_BOOTSTRAP_EXECUTION_CONTEXT",
        {
            "contract": {"schema": "caller-constructed"},
            "checkout_root_fd": 0,
            "checkout_root_path": str(ROOT),
            "source_fds": {},
            "source_snapshots": {},
            "runtime_fds": {},
            "runtime_snapshots": {},
        },
    )
    monkeypatch.setattr(ocsc, "hash_json", lambda _value: "0" * 64)
    with pytest.raises(ocsc.ContractError, match="execution contract mismatch"):
        ocsc.bootstrap_execution_contract(required=True)

    shadow = tmp_path / "shadow-runtime"
    shadow.mkdir()
    marker = tmp_path / "mutable-hashlib-imported"
    (shadow / "hashlib.py").write_text(
        "open({!r}, 'wb').write(b'imported')\n".format(str(marker)),
        encoding="ascii",
    )
    result = subprocess.run(
        source_bound_command(["--help"], profile="test"),
        cwd=ROOT,
        env={"LC_ALL": "C", "PYTHONPATH": str(shadow)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_nonexecuting_consumer_rejects_boolean_counts_and_execution() -> None:
    contract = ocsc.nonexecuting_consumer_contract()
    request = {
        "schema": "shohin-ocsc-nonexecuting-consumer-request-v1",
        "action": "validate-contract-reference-only",
        "consumer_contract_sha256": contract["payload_sha256"],
        "bundle_manifest_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "evaluation_gate_contract_sha256": ocsc.evaluation_gate_contract()[
            "payload_sha256"
        ],
        "run_cell": "A",
        "paired_seed": ocsc.PAIRED_SEEDS[0],
        "updates": ocsc.UPDATES_PER_ARM,
        "batch_slots": ocsc.BATCH_SLOTS,
        "sequence_length": ocsc.SEQUENCE_LENGTH,
        "training_requested": False,
        "evaluation_requested": False,
        "publication_requested": False,
    }
    assert ocsc.validate_nonexecuting_consumer_request(request) == request
    for field in ("paired_seed", "updates", "batch_slots", "sequence_length"):
        forged = copy.deepcopy(request)
        forged[field] = True
        with pytest.raises(ocsc.ContractError, match="consumer request mismatch"):
            ocsc.validate_nonexecuting_consumer_request(forged)
    executing = copy.deepcopy(request)
    executing["training_requested"] = True
    with pytest.raises(ocsc.ContractError, match="consumer request mismatch"):
        ocsc.validate_nonexecuting_consumer_request(executing)

    gate_forgery = copy.deepcopy(ocsc.evaluation_gate_contract())
    gate_forgery["hidden"]["direct_overall"]["minimum"] = True
    gate_forgery.pop("payload_sha256")
    gate_forgery = ocsc.with_payload_hash(gate_forgery, "payload_sha256")
    with pytest.raises(ocsc.ContractError, match="evaluation gate contract"):
        ocsc.validate_evaluation_gate_contract(gate_forgery)


def test_qualification_events_broker_and_counts_are_mechanically_derived(
    tmp_path: Path,
) -> None:
    source_hash = sha256_bytes(b"qualification-source-manifest")
    evidence = make_qualification_evidence(tmp_path / "qualification", source_hash)
    report = evidence["report"]
    assert report["summary"] == {
        "raw_event_count": len(ocsc.LINUX_LUSTRE_QUALIFICATION_CHECKS),
        "broker_request_count": 1,
        "broker_receipt_count": 1,
        "derived_check_count": len(ocsc.LINUX_LUSTRE_QUALIFICATION_CHECKS),
        "retained_evidence_count": len(ocsc.QUALIFICATION_CRASH_POINTS),
        "distinct_host_count": 2,
    }
    assert set(report["check_evidence"]) == set(ocsc.LINUX_LUSTRE_QUALIFICATION_CHECKS)
    assert not any(
        isinstance(value, bool)
        for evidence_ids in report["check_evidence"].values()
        for value in evidence_ids
    )
    for path_key in ("request_path", "destination_path", "receipt_path"):
        retained_path = Path(evidence["transfer"][path_key])
        assert retained_path.is_file()
        assert stat.S_IMODE(retained_path.stat().st_mode) == 0o444

    with pytest.raises(ocsc.ContractError, match="no authority"):
        ocsc.validate_qualification_summary({"forged": True}, True)

    forged_event = copy.deepcopy(evidence["events"][0])
    forged_event["details"]["outcome"] = True
    with pytest.raises(ocsc.ContractError, match="may not contain booleans"):
        ocsc.derive_qualification_report(
            [forged_event, *evidence["events"][1:]],
            evidence["broker_requests"],
            evidence["broker_receipts"],
            source_hash,
            "test",
        )

    with pytest.raises(ocsc.ContractError, match="broker request inventory"):
        ocsc.derive_qualification_report(
            evidence["events"],
            [],
            evidence["broker_receipts"],
            source_hash,
            "test",
        )

    without_publication = [
        event
        for event in evidence["events"]
        if event["event_type"] != "publication_path_complete"
    ]
    with pytest.raises(ocsc.ContractError, match="event/check inventory"):
        ocsc.derive_qualification_report(
            without_publication,
            evidence["broker_requests"],
            evidence["broker_receipts"],
            source_hash,
            "test",
        )

    forged_crash_events = copy.deepcopy(evidence["events"])
    crash_index = next(
        index
        for index, event in enumerate(forged_crash_events)
        if event["event_type"] == "all_crash_evidence_permanently_retained"
    )
    forged_crash = forged_crash_events[crash_index]
    forged_crash.pop("signature_hex")
    forged_crash["details"]["retained_evidence"].pop()
    forged_crash["details"]["evidence_sha256"] = ocsc.hash_json(
        {
            key: value
            for key, value in forged_crash["details"].items()
            if key != "evidence_sha256"
        }
    )
    forged_signed_crash = ocsc.sign_qualification_event(
        forged_crash,
        TEST_LINUX_LUSTRE_QUALIFICATION_PRIVATE_KEY_HEX,
        "test",
    )
    forged_crash_events[crash_index] = forged_signed_crash
    with pytest.raises(ocsc.ContractError, match="retained-evidence inventory"):
        ocsc.derive_qualification_report(
            forged_crash_events,
            evidence["broker_requests"],
            evidence["broker_receipts"],
            source_hash,
            "test",
        )

    forged_report = copy.deepcopy(report)
    forged_report["summary"]["retained_evidence_count"] += 1
    forged_report.pop("payload_sha256")
    forged_report = ocsc.with_payload_hash(forged_report, "payload_sha256")
    with pytest.raises(ocsc.ContractError, match="cannot derive marker"):
        ocsc.qualification_marker(forged_report)


def test_qualification_hook_requires_real_cross_node_lustre_contract() -> None:
    contract = ocsc_runner.qualification_contract()
    assert contract["status"] == "unexecuted-source-contract-only"
    assert contract["source_inventory"] == list(ocsc_runner.SOURCE_PATHS)
    assert contract["required_checks"] == list(ocsc.LINUX_LUSTRE_QUALIFICATION_CHECKS)
    assert contract["runtime_closure_required_before_generator_action"] is True
    assert contract["inputs"]["synthetic_tokenizer_authorized"] is False
    assert contract["filesystem"]["lustre_required"] is True
    assert contract["filesystem"]["same_host_fork_counts_as_cross_node"] is False
    assert contract["roles"]["distinct_host_kernel_identities_required"] is True
    assert contract["required_crash_points"] == list(ocsc.QUALIFICATION_CRASH_POINTS)
    assert contract["executable_actions"] == {
        "source_manifest_inspection": "--print-source-manifest",
        "publisher_and_restart_path": "--qualification-output-dir",
        "publisher_hard_crash_selector": "--qualification-crash-point",
        "production_broker_transfer": "--qualification-broker-transfer-event",
        "event_derived_report_marker_receipt": (
            "--qualification-write-evidence-package"
        ),
    }
    assert contract["event_authority"]["caller_summary_authority"] is False
    assert contract["event_authority"]["caller_check_map_authority"] is False
    assert contract["evidence_retention"]["delete_or_unlink_authorized"] is False
    assert contract["evidence_retention"]["rewrite_authorized"] is False
    assert contract["bundle_publication_authority"] is False
    assert contract["consumer_integration_authority"] is False
    assert contract["fit_or_evaluation_authority"] is False
    assert contract["qualification_authority"] is False
    assert contract["payload_sha256"] == ocsc_runner.hash_json(
        {key: value for key, value in contract.items() if key != "payload_sha256"}
    )


@pytest.mark.parametrize("journal_present", (False, True))
def test_recovery_retains_partial_stage_at_crash_points_without_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_present: bool,
) -> None:
    stage_name = ".ocsc.partial.mechanics-only"
    journal_name = ".ocsc.mechanics-only.recovery.json"
    stage = tmp_path / stage_name
    stage.mkdir(mode=0o700)
    child = stage / "manifest.json"
    child.write_bytes(b"partial evidence")
    stage_before = stage.stat()
    child_before = child.stat()
    journal = tmp_path / journal_name
    if journal_present:
        journal.write_bytes(b"authenticated journal placeholder")
        journal_before = journal.stat()
    monkeypatch.setattr(
        ocsc,
        "publication_staging_names",
        lambda *args: (stage_name, journal_name),
    )
    if journal_present:
        authenticated_record = {"authenticated": True}
        monkeypatch.setattr(
            ocsc,
            "_read_recovery_record_at",
            lambda *args: (authenticated_record, journal.stat()),
        )
        monkeypatch.setattr(
            ocsc,
            "publication_recovery_record",
            lambda *args: authenticated_record,
        )
    parent_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        expected_message = (
            "permanently retained"
            if journal_present
            else "unauthenticated publication stage collision"
        )
        with pytest.raises(ocsc.ContractError, match=expected_message):
            ocsc._recover_interrupted_publication(
                parent_fd,
                str(tmp_path.resolve()),
                tmp_path / "bundle",
                {name: b"expected" for name in ocsc.ARTIFACT_NAMES},
                tokenizer_path=tmp_path / "unused-tokenizer",
                prompt_registry_path=tmp_path / "unused-registry",
                confirmation_path=tmp_path / "unused-confirmation",
                publication_commitment_path=tmp_path / "unused-commitment",
                independent_review_receipt_path=tmp_path / "unused-review",
                input_snapshots={},
                publication_commitment_snapshot=None,
                independent_review_snapshot=None,
                publication_receipt={},
                independent_review_receipt={},
                source_manifest_contract={},
                lease_record={},
                lease_was_created=False,
            )
    finally:
        os.close(parent_fd)
    assert child.read_bytes() == b"partial evidence"
    assert (stage.stat().st_dev, stage.stat().st_ino) == (
        stage_before.st_dev,
        stage_before.st_ino,
    )
    assert (child.stat().st_dev, child.stat().st_ino) == (
        child_before.st_dev,
        child_before.st_ino,
    )
    if journal_present:
        assert (journal.stat().st_dev, journal.stat().st_ino) == (
            journal_before.st_dev,
            journal_before.st_ino,
        )


def test_recovery_never_deletes_foreign_or_inode_substituted_stage(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    output = tmp_path / "bundle"
    case_root = tmp_path / "substitution-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    parent_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    moved_stage = tmp_path / "owned-stage-moved"
    replacement = tmp_path / stage_name
    try:
        _, lease_record = leave_stale_publication_lease(
            parent_fd,
            case["publication_receipt"],
            case["independent_review_receipt"],
        )
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        stage_fd = os.open(
            stage_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_fd,
        )
        try:
            stage_metadata = os.fstat(stage_fd)
            record = ocsc.publication_recovery_record(
                case["publication_receipt"],
                case["independent_review_receipt"],
                case["artifacts"],
                stage_metadata.st_dev,
                stage_metadata.st_ino,
                stage_metadata.st_uid,
                lease_record,
            )
            ocsc._publish_recovery_record_at(parent_fd, journal_name, record)
        finally:
            os.close(stage_fd)
        os.rename(
            stage_name, moved_stage.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
        )
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        (replacement / "foreign.txt").write_bytes(b"foreign\n")
        with pytest.raises(ocsc.ContractError, match="journal authentication failed"):
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
            )
        assert moved_stage.is_dir()
        assert replacement.is_dir()
        assert (replacement / "foreign.txt").read_bytes() == b"foreign\n"
        assert (tmp_path / journal_name).is_file()
        assert not output.exists()
    finally:
        os.close(parent_fd)


def test_recovery_rejects_numerically_equal_forged_journal_without_deletion(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    output = tmp_path / "bundle"
    case_root = tmp_path / "type-strict-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    parent_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        _, lease_record = leave_stale_publication_lease(
            parent_fd,
            case["publication_receipt"],
            case["independent_review_receipt"],
        )
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        stage_metadata = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        record = ocsc.publication_recovery_record(
            case["publication_receipt"],
            case["independent_review_receipt"],
            case["artifacts"],
            stage_metadata.st_dev,
            stage_metadata.st_ino,
            stage_metadata.st_uid,
            lease_record,
        )
        forged = copy.deepcopy(record)
        forged["output_parent"]["device"] = float(forged["output_parent"]["device"])
        forged.pop("payload_sha256")
        forged = ocsc.with_payload_hash(forged, "payload_sha256")
        ocsc._publish_recovery_record_at(parent_fd, journal_name, forged)
        with pytest.raises(ocsc.ContractError, match="journal authentication failed"):
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
            )
        assert (tmp_path / stage_name).is_dir()
        assert (tmp_path / journal_name).is_file()
        assert not output.exists()
    finally:
        os.close(parent_fd)


def test_recovery_rejects_coherent_forgery_and_preserves_foreign_child(
    generated: dict[str, Path | list[dict]], tmp_path: Path
) -> None:
    output = tmp_path / "bundle"
    case_root = tmp_path / "coherent-forgery-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    parent_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        lease_name, lease_record = leave_stale_publication_lease(
            parent_fd,
            case["publication_receipt"],
            case["independent_review_receipt"],
        )
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        foreign_child = tmp_path / stage_name / "manifest.json"
        foreign_child.write_bytes(b"foreign-custody-payload\n")
        os.chmod(foreign_child, 0o600)
        child_before = foreign_child.stat()
        stage_metadata = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        forged = ocsc.publication_recovery_record(
            case["publication_receipt"],
            case["independent_review_receipt"],
            case["artifacts"],
            stage_metadata.st_dev,
            stage_metadata.st_ino,
            stage_metadata.st_uid,
            lease_record,
        )
        ocsc._publish_recovery_record_at(parent_fd, journal_name, forged)
        with pytest.raises(
            ocsc.ContractError,
            match="permanently retained",
        ):
            ocsc.publish_bundle(
                output,
                case["artifacts"],
                mode="test",
                tokenizer_path=Path(generated["tokenizer"]),
                prompt_registry_path=Path(generated["registry"]),
                confirmation_path=Path(generated["confirmation"]),
                pad_token_id=0,
                publication_receipt=case["publication_receipt"],
                independent_review_receipt=case["independent_review_receipt"],
            )
        child_after = foreign_child.stat()
        assert foreign_child.read_bytes() == b"foreign-custody-payload\n"
        assert (child_before.st_dev, child_before.st_ino) == (
            child_after.st_dev,
            child_after.st_ino,
        )
        assert stat.S_IMODE((tmp_path / stage_name).stat().st_mode) == 0o700
        assert (tmp_path / journal_name).is_file()
        assert (tmp_path / lease_name).is_file()
        assert not output.exists()
    finally:
        os.close(parent_fd)


def test_failure_retains_foreign_replacement_child_and_metadata(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "foreign-replacement-output"
    case_root = tmp_path / "foreign-replacement-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    lease_name = ocsc.publication_lease_name(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    observed = {}

    def replace_child_then_fail(path: Path, directory_fd: int, **kwargs) -> dict:
        stage = Path(path)
        victim = stage / "manifest.json"
        os.chmod(stage, 0o700)
        os.chmod(victim, 0o600)
        victim.unlink()
        victim.write_bytes(b"foreign replacement\n")
        os.chmod(victim, 0o600)
        observed["identity"] = (victim.stat().st_dev, victim.stat().st_ino)
        raise OSError(errno.EIO, "injected foreign replacement")

    monkeypatch.setattr(
        ocsc,
        "_strict_verify_publication_tree",
        replace_child_then_fail,
    )
    with pytest.raises(OSError, match="injected foreign replacement"):
        ocsc.publish_bundle(
            output,
            case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=case["publication_receipt"],
            independent_review_receipt=case["independent_review_receipt"],
        )
    retained_stage = tmp_path / stage_name
    retained_child = retained_stage / "manifest.json"
    assert retained_child.read_bytes() == b"foreign replacement\n"
    assert (retained_child.stat().st_dev, retained_child.stat().st_ino) == observed[
        "identity"
    ]
    assert (tmp_path / journal_name).is_file()
    assert (tmp_path / lease_name).is_file()
    assert not output.exists()


def test_nonregular_partial_evidence_has_no_cleanup_callable(tmp_path: Path) -> None:
    stage_name = ".ocsc.partial.nonregular-child"
    stage = tmp_path / stage_name
    stage.mkdir(mode=0o700)
    child = stage / ocsc.ARTIFACT_NAMES[0]
    os.mkfifo(child, mode=0o600)
    child_before = child.lstat()
    assert not hasattr(ocsc, "_remove_bundle_at")
    assert not hasattr(ocsc, "_unlink_owned_entry_at")
    child_after = child.lstat()
    assert stat.S_ISFIFO(child_after.st_mode)
    assert (child_before.st_dev, child_before.st_ino) == (
        child_after.st_dev,
        child_after.st_ino,
    )
    assert stage.is_dir()


def test_failure_retains_exact_expected_subset(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "exact-subset-output"
    case_root = tmp_path / "exact-subset-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    stage_name, journal_name = ocsc.publication_staging_names(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    lease_name = ocsc.publication_lease_name(
        case["publication_receipt"], case["independent_review_receipt"]
    )
    original_open = ocsc.os.open
    injected = False

    def fail_before_second_artifact(
        path,
        flags,
        mode=0o777,
        *,
        dir_fd=None,
    ):
        nonlocal injected
        if (
            not injected
            and path == ocsc.ARTIFACT_NAMES[1]
            and dir_fd is not None
            and (tmp_path / stage_name).is_dir()
        ):
            directory_state = os.fstat(dir_fd)
            stage_state = (tmp_path / stage_name).stat()
            if (directory_state.st_dev, directory_state.st_ino) == (
                stage_state.st_dev,
                stage_state.st_ino,
            ):
                injected = True
                raise OSError(errno.EIO, "injected exact-subset failure")
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(ocsc.os, "open", fail_before_second_artifact)
    with pytest.raises(OSError, match="injected exact-subset failure"):
        ocsc.publish_bundle(
            output,
            case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=case["publication_receipt"],
            independent_review_receipt=case["independent_review_receipt"],
        )
    assert injected
    assert not output.exists()
    retained_stage = tmp_path / stage_name
    assert retained_stage.is_dir()
    assert (retained_stage / ocsc.ARTIFACT_NAMES[0]).is_file()
    assert not (retained_stage / ocsc.ARTIFACT_NAMES[1]).exists()
    assert (tmp_path / journal_name).is_file()
    assert (tmp_path / lease_name).is_file()


def test_publication_injected_write_and_postrename_fsync_fail_closed(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_output = tmp_path / "write-output"
    write_case_root = tmp_path / "write-case"
    write_case_root.mkdir()
    write_case = authorized_publication_case(generated, write_case_root, write_output)
    write_stage_name, write_journal_name = ocsc.publication_staging_names(
        write_case["publication_receipt"],
        write_case["independent_review_receipt"],
    )
    write_lease_name = ocsc.publication_lease_name(
        write_case["publication_receipt"],
        write_case["independent_review_receipt"],
    )
    original_write = ocsc.os.write
    write_failed = False
    partial_payload = b""

    def failing_artifact_write(descriptor: int, payload: bytes) -> int:
        nonlocal partial_payload, write_failed
        descriptor_state = os.fstat(descriptor)
        stage = tmp_path / write_stage_name
        artifact_descriptor = stage.is_dir() and any(
            path.is_file()
            and (path.stat().st_dev, path.stat().st_ino)
            == (descriptor_state.st_dev, descriptor_state.st_ino)
            for path in stage.iterdir()
        )
        if artifact_descriptor and not write_failed:
            partial_payload = payload[:17]
            original_write(descriptor, partial_payload)
            write_failed = True
            raise OSError(errno.EIO, "injected artifact write failure")
        return original_write(descriptor, payload)

    monkeypatch.setattr(ocsc.os, "write", failing_artifact_write)
    with pytest.raises(OSError, match="injected artifact write failure"):
        ocsc.publish_bundle(
            write_output,
            write_case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=write_case["publication_receipt"],
            independent_review_receipt=write_case["independent_review_receipt"],
        )
    assert write_failed
    assert not write_output.exists()
    retained_stage = tmp_path / write_stage_name
    retained_child = retained_stage / ocsc.ARTIFACT_NAMES[0]
    assert retained_stage.is_dir()
    assert retained_child.is_file()
    assert retained_child.read_bytes() == partial_payload
    assert (
        0 < len(partial_payload) < len(write_case["artifacts"][ocsc.ARTIFACT_NAMES[0]])
    )
    assert (tmp_path / write_journal_name).is_file()
    assert (tmp_path / write_lease_name).is_file()
    retained_write = ocsc.capture_retained_publication_evidence(
        tmp_path,
        crash_point="partial-artifact-write",
        stage_name=write_stage_name,
        canonical_name=write_output.name,
        journal_name=write_journal_name,
        lease_name=write_lease_name,
    )
    assert retained_write["stage_state"] == "retained"
    assert retained_write["journal_state"] == "retained"
    assert retained_write["lease_state"] == "retained"

    monkeypatch.setattr(ocsc.os, "write", original_write)
    fsync_output = tmp_path / "fsync-output"
    fsync_case_root = tmp_path / "fsync-case"
    fsync_case_root.mkdir()
    fsync_case = authorized_publication_case(generated, fsync_case_root, fsync_output)
    fsync_stage_name, fsync_journal_name = ocsc.publication_staging_names(
        fsync_case["publication_receipt"],
        fsync_case["independent_review_receipt"],
    )
    fsync_lease_name = ocsc.publication_lease_name(
        fsync_case["publication_receipt"],
        fsync_case["independent_review_receipt"],
    )
    original_fsync = ocsc.os.fsync
    parent_identity = tmp_path.stat()
    fsync_failed = False
    monkeypatch.setattr(
        ocsc, "_strict_verify_publication_tree", lightweight_publication_verifier
    )

    def failing_postrename_fsync(descriptor: int) -> None:
        nonlocal fsync_failed
        metadata = os.fstat(descriptor)
        if (
            not fsync_failed
            and fsync_output.is_dir()
            and (metadata.st_dev, metadata.st_ino)
            == (parent_identity.st_dev, parent_identity.st_ino)
        ):
            fsync_failed = True
            raise OSError(errno.EIO, "injected post-rename fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(ocsc.os, "fsync", failing_postrename_fsync)
    with pytest.raises(OSError, match="injected post-rename fsync failure"):
        ocsc.publish_bundle(
            fsync_output,
            fsync_case["artifacts"],
            mode="test",
            tokenizer_path=Path(generated["tokenizer"]),
            prompt_registry_path=Path(generated["registry"]),
            confirmation_path=Path(generated["confirmation"]),
            pad_token_id=0,
            publication_receipt=fsync_case["publication_receipt"],
            independent_review_receipt=fsync_case["independent_review_receipt"],
        )
    assert fsync_failed
    assert fsync_output.is_dir()
    assert sorted(path.name for path in fsync_output.iterdir()) == sorted(
        ocsc.ARTIFACT_NAMES
    )
    assert not (tmp_path / fsync_stage_name).exists()
    assert (tmp_path / fsync_journal_name).is_file()
    assert (tmp_path / fsync_lease_name).is_file()
    retained_canonical = ocsc.capture_retained_publication_evidence(
        tmp_path,
        crash_point="canonical-before-parent-fsync",
        stage_name=fsync_stage_name,
        canonical_name=fsync_output.name,
        journal_name=fsync_journal_name,
        lease_name=fsync_lease_name,
    )
    assert retained_canonical["stage_state"] == "renamed-canonical"
    assert retained_canonical["canonical_state"] == "retained"


def restore_swapped_bundle(bundle: Path, moved: Path, substitute: Path) -> None:
    discarded = bundle.parent / (bundle.name + "-discarded-substitute")
    if bundle.exists():
        os.chmod(bundle, 0o755)
        bundle.rename(discarded)
    if moved.exists():
        os.chmod(moved, 0o755)
        moved.rename(bundle)
        os.chmod(bundle, 0o555)
    if discarded.exists():
        os.chmod(discarded, 0o755)
        for path in discarded.iterdir():
            os.chmod(path, 0o644)
        shutil.rmtree(discarded)
    if substitute.exists():
        os.chmod(substitute, 0o755)
        for path in substitute.iterdir():
            os.chmod(path, 0o644)
        shutil.rmtree(substitute)


def swap_bundle_path(bundle: Path, moved: Path, substitute: Path) -> None:
    # macOS requires write permission on the directory being renamed. Restore
    # the original mode before the verifier observes either directory again.
    os.chmod(bundle, 0o755)
    bundle.rename(moved)
    os.chmod(moved, 0o555)
    os.chmod(substitute, 0o755)
    substitute.rename(bundle)
    os.chmod(bundle, 0o555)


def test_deterministic_verification_rejects_bundle_directory_swap(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = Path(generated["bundle"])
    moved = tmp_path / "verified-bundle-moved"
    substitute = tmp_path / "verified-bundle-substitute"
    shutil.copytree(bundle, substitute)
    triggered = False

    def swapping_build(*args, **kwargs):
        nonlocal triggered
        artifacts = {name: (bundle / name).read_bytes() for name in ocsc.ARTIFACT_NAMES}
        swap_bundle_path(bundle, moved, substitute)
        triggered = True
        return artifacts

    monkeypatch.setattr(ocsc, "build_artifacts", swapping_build)
    try:
        with pytest.raises(ocsc.ContractError, match="bundle directory changed"):
            ocsc.verify_bundle(
                bundle,
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["confirmation"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
        assert triggered
    finally:
        restore_swapped_bundle(bundle, moved, substitute)


def test_hidden_verification_retains_verified_bundle_snapshot_after_swap(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = Path(generated["bundle"])
    moved = tmp_path / "hidden-bundle-moved"
    substitute = tmp_path / "hidden-bundle-substitute"
    shutil.copytree(bundle, substitute)
    triggered = False

    def swapping_verification(bundle_snapshot, *args, **kwargs):
        nonlocal triggered
        manifest = ocsc.load_manifest(
            Path(bundle_snapshot.resolved_path),
            bundle_snapshot.files["manifest.json"],
        )
        result = {
            "prepublication_commitment_sha256": manifest["inputs"][
                "prepublication_commitment_sha256"
            ]
        }
        swap_bundle_path(bundle, moved, substitute)
        triggered = True
        return result, manifest

    monkeypatch.setattr(ocsc, "_verify_bundle_snapshot", swapping_verification)
    try:
        with pytest.raises(ocsc.ContractError, match="bundle directory changed"):
            ocsc.verify_hidden_opening(
                Path(generated["opening"]),
                Path(generated["confirmation"]),
                bundle,
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["custodian_opening"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
        assert triggered
    finally:
        restore_swapped_bundle(bundle, moved, substitute)


def test_hidden_verification_rejects_inter_snapshot_signed_input_swap(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Path(generated["registry"])
    custody_root = registry.parent
    signed_holding = tmp_path / "signed-prompt-registry.jsonl"
    attacker_source = tmp_path / "attacker-prompt-registry.jsonl"
    attacker_holding = tmp_path / "attacker-prompt-registry-holding.jsonl"
    discarded_attacker = tmp_path / "discarded-attacker-prompt-registry.jsonl"
    shutil.copyfile(registry, attacker_source)
    os.chmod(attacker_source, 0o444)
    original_verification = ocsc._verify_bundle_snapshot
    triggered = False

    def exchange(current: Path, holding: Path, replacement: Path) -> None:
        os.chmod(custody_root, 0o755)
        current.rename(holding)
        replacement.rename(current)
        os.chmod(custody_root, 0o555)

    # Present a byte-identical but unsigned inode for the first hidden snapshot.
    exchange(registry, signed_holding, attacker_source)

    def swapping_verification(bundle_snapshot, *args, **kwargs):
        nonlocal triggered
        # Restore the signed inode only while bundle authentication runs, then
        # restore the unsigned inode before hidden validation and the final read.
        exchange(registry, attacker_holding, signed_holding)
        triggered = True
        try:
            return original_verification(bundle_snapshot, *args, **kwargs)
        finally:
            exchange(registry, signed_holding, attacker_holding)

    monkeypatch.setattr(ocsc, "_verify_bundle_snapshot", swapping_verification)
    try:
        with pytest.raises(
            ocsc.ContractError,
            match="prepublication commitment request/source mismatch",
        ):
            ocsc.verify_hidden_opening(
                Path(generated["opening"]),
                Path(generated["confirmation"]),
                Path(generated["bundle"]),
                Path(generated["tokenizer"]),
                registry,
                Path(generated["custodian_opening"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
        assert triggered
    finally:
        os.chmod(custody_root, 0o755)
        if registry.exists():
            registry.rename(discarded_attacker)
        if signed_holding.exists():
            signed_holding.rename(registry)
        os.chmod(custody_root, 0o555)
        if discarded_attacker.exists():
            os.chmod(discarded_attacker, 0o644)
            discarded_attacker.unlink()


def test_runtime_closure_binds_native_code_and_rejects_shadow_imports(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ocsc.runtime_closure_contract()
    assert contract["schema"] == "shohin-ocsc-runtime-closure-v3"
    assert contract["isolated_cli_required"] is True
    assert set(contract["distributions"]) == set(ocsc.RUNTIME_DISTRIBUTIONS)
    assert contract["interpreter"]["sha256"] == sha256_file(
        Path(sys.executable).resolve()
    )
    for distribution in contract["distributions"].values():
        assert distribution["file_count"] == len(distribution["files"])
        assert distribution["files_sha256"] == ocsc.hash_json(distribution["files"])
        assert distribution["native_files"]
        assert distribution["payload_sha256"] == ocsc.hash_json(
            {
                key: value
                for key, value in distribution.items()
                if key != "payload_sha256"
            }
        )

    shadow_root = tmp_path / "shadow-runtime"
    shadow_root.mkdir()
    (shadow_root / "tokenizers.py").write_text(
        "raise RuntimeError('shadow module executed')\n", encoding="ascii"
    )
    monkeypatch.syspath_prepend(str(shadow_root))
    with pytest.raises(ocsc.ContractError, match="shadow import rejected: tokenizers"):
        ocsc.runtime_closure_contract()

    command = [sys.executable, "-I", "-S", "-B", str(GENERATOR), "--help"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "direct generator execution is forbidden" in result.stderr


def test_linux_execution_identity_rejects_path_bytes_for_a_different_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapped_path = tmp_path / "mapped-native.so"
    replacement_path = tmp_path / "replacement-native.so"
    mapped_path.write_bytes(b"same bytes")
    replacement_path.write_bytes(b"same bytes")
    mapped = mapped_path.stat()
    mapping_line = "1000-2000 r-xp 00000000 {:x}:{:x} {} {}".format(
        os.major(mapped.st_dev),
        os.minor(mapped.st_dev),
        mapped.st_ino,
        mapped_path,
    )
    mapping = ocsc._parse_linux_native_image_mappings([mapping_line])[str(mapped_path)]
    conflicting_line = "1000-2000 r-xp 00000000 {:x}:{:x} {} {}".format(
        os.major(mapped.st_dev),
        os.minor(mapped.st_dev),
        mapped.st_ino + 1,
        mapped_path,
    )
    with pytest.raises(
        ocsc.ContractError,
        match="maps multiple file identities",
    ):
        ocsc._parse_linux_native_image_mappings([mapping_line, conflicting_line])

    monkeypatch.setattr(
        ocsc,
        "_resolved_runtime_snapshot",
        lambda path, label: {
            "resolved_path": str(replacement_path),
            "bytes": len(b"same bytes"),
            "sha256": sha256_bytes(b"same bytes"),
            "mode": 0o644,
            "owner_uid": os.geteuid(),
            "device": replacement_path.stat().st_dev,
            "inode": replacement_path.stat().st_ino,
            "hard_links": 1,
        },
    )
    with pytest.raises(
        ocsc.ContractError,
        match="does not name the executed mapping identity",
    ):
        ocsc._bound_linux_native_image(str(mapped_path), mapping)

    descriptor = os.open(mapped_path, os.O_RDONLY)
    try:
        ocsc_runner.verify_executed_python_identity(
            descriptor,
            platform_name="linux",
            execution_path=mapped_path,
        )
        with pytest.raises(
            RuntimeError,
            match="does not name the interpreter executing the runner",
        ):
            ocsc_runner.verify_executed_python_identity(
                descriptor,
                platform_name="linux",
                execution_path=replacement_path,
            )
    finally:
        os.close(descriptor)


def test_input_schema_path_publication_fsync_and_ascii(
    generated: dict[str, Path | list[dict]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "tokenizer",
        "registry",
        "opening",
        "confirmation",
        "custodian_opening",
        "publication_commitment",
        "independent_review_receipt",
    ):
        path = Path(generated[key])
        ocsc.validate_custody_root_file(path, key)
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o555

    crowded_root = tmp_path / "crowded-root"
    crowded_root.mkdir()
    crowded_file = crowded_root / "prepublication_commitment.json"
    crowded_file.write_bytes(Path(generated["publication_commitment"]).read_bytes())
    (crowded_root / "extra").write_bytes(b"x")
    os.chmod(crowded_file, 0o444)
    os.chmod(crowded_root / "extra", 0o444)
    os.chmod(crowded_root, 0o555)
    with pytest.raises(ocsc.ContractError, match="exactly one file"):
        ocsc.validate_custody_root_file(crowded_file, "crowded")

    crowded_tokenizer_root = tmp_path / "crowded-tokenizer-root"
    crowded_tokenizer_root.mkdir()
    crowded_tokenizer = crowded_tokenizer_root / "tokenizer.json"
    shutil.copyfile(Path(generated["tokenizer"]), crowded_tokenizer)
    (crowded_tokenizer_root / "extra").write_bytes(b"x")
    os.chmod(crowded_tokenizer, 0o444)
    os.chmod(crowded_tokenizer_root / "extra", 0o444)
    os.chmod(crowded_tokenizer_root, 0o555)
    with pytest.raises(ocsc.ContractError, match="exactly one file"):
        ocsc.publication_commitment_request(
            "test",
            tmp_path / "crowded-output",
            crowded_tokenizer,
            Path(generated["registry"]),
            Path(generated["confirmation"]),
            0,
        )

    mutable_registry_root = tmp_path / "mutable-registry-root"
    mutable_registry_root.mkdir()
    mutable_registry = mutable_registry_root / "prompt_registry.jsonl"
    shutil.copyfile(Path(generated["registry"]), mutable_registry)
    os.chmod(mutable_registry, 0o444)
    with pytest.raises(ocsc.ContractError, match="directory must be mode 0555"):
        ocsc.publication_commitment_request(
            "test",
            tmp_path / "mutable-output",
            Path(generated["tokenizer"]),
            mutable_registry,
            Path(generated["confirmation"]),
            0,
        )

    registry_root = Path(generated["registry"]).parent
    os.chmod(registry_root, 0o755)
    try:
        with pytest.raises(ocsc.ContractError, match="directory must be mode 0555"):
            ocsc.verify_bundle(
                Path(generated["bundle"]),
                Path(generated["tokenizer"]),
                Path(generated["registry"]),
                Path(generated["confirmation"]),
                Path(generated["publication_commitment"]),
                Path(generated["independent_review_receipt"]),
            )
    finally:
        os.chmod(registry_root, 0o555)

    registry_rows = read_jsonl(Path(generated["registry"]))
    duplicate = [dict(row) for row in registry_rows]
    duplicate[-1]["normalized_prompt_sha256"] = duplicate[0]["normalized_prompt_sha256"]
    duplicate_path = tmp_path / "duplicate-registry.jsonl"
    write_jsonl(duplicate_path, duplicate)
    with pytest.raises(ocsc.ContractError, match="duplicate normalized"):
        ocsc.load_prompt_registry(duplicate_path)

    semantic_duplicate = [dict(row) for row in registry_rows]
    semantic_duplicate[-1]["semantic_signature_sha256"] = semantic_duplicate[0][
        "semantic_signature_sha256"
    ]
    semantic_duplicate_path = tmp_path / "semantic-duplicate-registry.jsonl"
    write_jsonl(semantic_duplicate_path, semantic_duplicate)
    with pytest.raises(ocsc.ContractError, match="duplicate semantic"):
        ocsc.load_prompt_registry(semantic_duplicate_path)

    hardlink_registry = tmp_path / "registry-hardlink.jsonl"
    os.link(Path(generated["registry"]), hardlink_registry)
    with pytest.raises(ocsc.ContractError, match="exactly one hard link"):
        ocsc.load_prompt_registry(Path(generated["registry"]))
    hardlink_registry.unlink()

    unsafe_manifest = tmp_path / "unsafe-manifest"
    unsafe_manifest.mkdir()
    shutil.copy2(
        Path(generated["bundle"]) / "manifest.json",
        unsafe_manifest / "manifest.json",
    )
    manifest_path = unsafe_manifest / "manifest.json"
    os.chmod(manifest_path, 0o644)
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    files = manifest["files"]
    files["../schedule.jsonl"] = files.pop("schedule.jsonl")
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = ocsc.hash_json(manifest)
    manifest_path.write_bytes(ocsc.pretty_json_bytes(manifest))
    os.chmod(manifest_path, 0o444)
    with pytest.raises(ocsc.ContractError, match="file inventory|unsafe path"):
        ocsc.load_manifest(unsafe_manifest)

    fsync_modes = []
    original_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsync_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(ocsc.os, "fsync", recording_fsync)
    output = tmp_path / "published"
    case_root = tmp_path / "direct-publication-case"
    case_root.mkdir()
    case = authorized_publication_case(generated, case_root, output)
    publication_kwargs = {
        "mode": "test",
        "tokenizer_path": Path(generated["tokenizer"]),
        "prompt_registry_path": Path(generated["registry"]),
        "confirmation_path": Path(generated["confirmation"]),
        "pad_token_id": 0,
        "publication_receipt": case["publication_receipt"],
        "independent_review_receipt": case["independent_review_receipt"],
    }
    monkeypatch.setattr(
        ocsc, "_strict_verify_publication_tree", lightweight_publication_verifier
    )
    ocsc.publish_bundle(
        output,
        case["artifacts"],
        **publication_kwargs,
    )
    directory_fsyncs = sum(stat.S_ISDIR(mode) for mode in fsync_modes)
    assert directory_fsyncs >= 2
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in output.iterdir())
    ocsc.publish_bundle(output, case["artifacts"], **publication_kwargs)

    for path in (
        ROOT / "R12_ORTHOGONAL_CARRY_SERIALIZER_CURRICULUM_PREREG.md",
        GENERATOR,
        Path(__file__),
    ):
        path.read_bytes().decode("ascii")
