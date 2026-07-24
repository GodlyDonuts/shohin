"""Hash-bound frozen Shohin residual features with exact byte alignment.

This module is deliberately agnostic to EFC source and query grammars.  The
caller supplies pre-tokenized IDs and exact token-to-byte bounds.  Every payload
byte must be covered by exactly one valid token; malformed alignments fail
before the frozen GPT is evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.util
import inspect
import json
import marshal
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import torch
import torch.nn as nn

try:
    from .model import GPT, GPTConfig
except ImportError:  # Direct execution with train/ on sys.path.
    from model import GPT, GPTConfig


_SHA256 = re.compile(r"[0-9a-f]{64}")
DEFAULT_MAX_PAYLOAD_BYTES = 16_384
EXPECTED_MODEL_RUNTIME_MANIFEST_SHA256 = (
    "c411a1a594b5933046d6ba24d54d1b1cb86a6662a1d2e461575674abde3a00d2"
)
EXPECTED_TRUNK_RUNTIME_MANIFEST_SHA256 = (
    "c2c2a1554c29c922ad3d1288986196cd7b13816c28fc4c3850db6400ffa195b4"
)
_EXTERNAL_RUNTIME_METHODS = frozenset(
    (
        "__call__",
        "__getattr__",
        "__getattribute__",
        "__getitem__",
        "__iter__",
        "__len__",
        "_call_impl",
        "_slow_forward",
        "_wrapped_call_impl",
        "forward",
    )
)
_TRUNK_EXECUTION_METHODS = (
    "train",
    "encode_batch",
    "flatten_byte_features",
    "_validate_inputs",
    "forward",
)


class FrozenShohinTrunkError(ValueError):
    """The checkpoint, trunk, tensor, or byte-alignment contract failed."""


@dataclass(frozen=True, slots=True)
class ShohinTrunkParameterReceipt:
    checkpoint_sha256: str
    checkpoint_verified: bool
    parent_unique_parameters: int
    adapter_unique_parameters: int
    integrated_unique_parameters: int
    trainable_unique_parameters: int


@dataclass(frozen=True, slots=True)
class ShohinTrunkBatch:
    payloads: tuple[bytes, ...]
    token_ids: torch.Tensor
    token_valid: torch.Tensor
    token_byte_bounds: torch.Tensor


@dataclass(frozen=True, slots=True)
class ByteAlignedResidualFeatures:
    """Raw post-block residuals aligned to both tokens and payload bytes."""

    block_indices: tuple[int, ...]
    token_features: torch.Tensor
    token_valid: torch.Tensor
    byte_features: torch.Tensor
    byte_valid: torch.Tensor
    payload_lengths: torch.Tensor

    def __post_init__(self) -> None:
        if self.token_features.ndim != 4 or self.byte_features.ndim != 4:
            raise FrozenShohinTrunkError("residual features must be rank four")
        batch, layers, tokens, width = self.token_features.shape
        if self.byte_features.shape[:2] != (batch, layers):
            raise FrozenShohinTrunkError("token and byte residual batches differ")
        if self.byte_features.shape[-1] != width:
            raise FrozenShohinTrunkError("token and byte residual widths differ")
        if self.token_valid.shape != (batch, tokens) or self.token_valid.dtype != torch.bool:
            raise FrozenShohinTrunkError("token validity geometry differs")
        if (
            self.byte_valid.shape != (batch, self.byte_features.shape[2])
            or self.byte_valid.dtype != torch.bool
        ):
            raise FrozenShohinTrunkError("byte validity geometry differs")
        if self.payload_lengths.shape != (batch,) or self.payload_lengths.dtype != torch.long:
            raise FrozenShohinTrunkError("payload length geometry differs")
        if layers != len(self.block_indices):
            raise FrozenShohinTrunkError("residual block count differs")


def _hash_checkpoint(handle) -> str:
    digest = sha256()
    while True:
        chunk = handle.read(8 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrozenShohinTrunkError(f"checkpoint {field} is not a mapping")
    return value


def _function_metadata_semantics(
    function,
    *,
    local_module: str,
    trail: frozenset[int],
) -> object:
    closure: list[object] = []
    for cell in function.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:
            closure.append({"empty_cell": True})
        else:
            closure.append(
                _semantic_value(
                    contents,
                    local_module=local_module,
                    trail=trail,
                )
            )
    return {
        "function": function.__qualname__,
        "code": sha256(marshal.dumps(function.__code__)).hexdigest(),
        "defaults": _semantic_value(
            function.__defaults__,
            local_module=local_module,
            trail=trail,
        ),
        "kwdefaults": _semantic_value(
            function.__kwdefaults__,
            local_module=local_module,
            trail=trail,
        ),
        "annotations": _semantic_value(
            function.__annotations__,
            local_module=local_module,
            trail=trail,
        ),
        "closure": closure,
        "attributes": _semantic_value(
            vars(function),
            local_module=local_module,
            trail=trail,
        ),
    }


def _semantic_value(
    value: object,
    *,
    local_module: str,
    trail: frozenset[int] = frozenset(),
) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {
            "tuple": [
                _semantic_value(
                    item,
                    local_module=local_module,
                    trail=trail,
                )
                for item in value
            ]
        }

    identity = id(value)
    value_type = type(value)
    type_name = f"{value_type.__module__}.{value_type.__qualname__}"
    if identity in trail:
        return {"cycle": type_name}
    next_trail = trail | {identity}

    if isinstance(value, list):
        return {
            "list": [
                _semantic_value(
                    item,
                    local_module=local_module,
                    trail=next_trail,
                )
                for item in value
            ]
        }
    if isinstance(value, (set, frozenset)):
        items = [
            _semantic_value(
                item,
                local_module=local_module,
                trail=next_trail,
            )
            for item in value
        ]
        items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {"set": items}
    if isinstance(value, Mapping):
        items = [
            (
                _semantic_value(
                    key,
                    local_module=local_module,
                    trail=next_trail,
                ),
                _semantic_value(
                    item,
                    local_module=local_module,
                    trail=next_trail,
                ),
            )
            for key, item in value.items()
        ]
        items.sort(
            key=lambda pair: json.dumps(
                pair[0],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {"mapping": items}
    if inspect.iscode(value):
        return {"code": sha256(marshal.dumps(value)).hexdigest()}
    if inspect.isfunction(value):
        return {
            "function_semantics": _function_semantics(
                value,
                local_module=local_module,
                trail=trail,
            )
        }
    if inspect.isclass(value):
        module_name = (
            "<bound-model-source>"
            if value.__module__ == local_module
            else value.__module__
        )
        return {"type": f"{module_name}.{value.__qualname__}"}
    if inspect.isbuiltin(value):
        return {
            "builtin": (
                f"{getattr(value, '__module__', '<unknown>')}."
                f"{getattr(value, '__qualname__', getattr(value, '__name__', '<unknown>'))}"
            )
        }
    if inspect.ismodule(value):
        return {"module": value.__name__}
    if callable(value):
        module_name = getattr(value, "__module__", value_type.__module__)
        qualified_name = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", value_type.__qualname__),
        )
        return {
            "callable": f"{module_name}.{qualified_name}",
            "callable_type": type_name,
        }
    return {
        "object_type": type_name,
        "representation": repr(value),
    }


def _module_reference_semantics(
    module: object,
    *,
    referenced_names: tuple[str, ...],
    local_module: str,
    depth: int = 2,
    trail: frozenset[int] = frozenset(),
) -> object:
    module_name = getattr(module, "__name__", None)
    if not isinstance(module_name, str):
        return _semantic_value(module, local_module=local_module, trail=trail)
    identity = id(module)
    if identity in trail:
        return {"module_cycle": module_name}
    next_trail = trail | {identity}
    namespace = vars(module)
    members: dict[str, object] = {}
    for name in referenced_names:
        if name not in namespace:
            continue
        value = namespace[name]
        if inspect.ismodule(value) and depth > 0:
            members[name] = _module_reference_semantics(
                value,
                referenced_names=referenced_names,
                local_module=local_module,
                depth=depth - 1,
                trail=next_trail,
            )
        elif inspect.isclass(value) and value.__module__ != local_module:
            members[name] = _external_class_semantics(
                value,
                local_module=local_module,
                trail=next_trail,
            )
        elif inspect.isfunction(value):
            members[name] = {
                "function_semantics": _function_semantics(
                    value,
                    local_module=local_module,
                    trail=next_trail,
                )
            }
        else:
            members[name] = _semantic_value(
                value,
                local_module=local_module,
                trail=next_trail,
            )
    return {
        "module": module_name,
        "referenced_members": members,
    }


def _shallow_function_semantics(
    function,
    *,
    local_module: str,
    trail: frozenset[int],
) -> str:
    payload = _semantic_value(
        function,
        local_module=local_module,
        trail=trail,
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


def _external_class_semantics(
    value: type,
    *,
    local_module: str,
    trail: frozenset[int],
) -> object:
    """Fingerprint external methods that can alter an existing module's forward."""
    classes: list[object] = []
    for base in value.__mro__:
        base_name = f"{base.__module__}.{base.__qualname__}"
        if base is object:
            classes.append({"class": base_name, "members": {}})
            continue
        base_identity = id(base)
        if base_identity in trail and base is not value:
            classes.append({"class_cycle": base_name})
            continue
        next_trail = trail | {base_identity}
        members: dict[str, object] = {}
        for name, member in sorted(vars(base).items()):
            if name not in _EXTERNAL_RUNTIME_METHODS:
                continue
            function = None
            if isinstance(member, (classmethod, staticmethod)):
                function = member.__func__
            elif inspect.isfunction(member):
                function = member
            if function is not None:
                members[name] = (
                    {
                        "function_semantics": (
                            _function_semantics(
                                function,
                                local_module=local_module,
                                trail=next_trail,
                            )
                            if name == "forward"
                            else _shallow_function_semantics(
                                function,
                                local_module=local_module,
                                trail=next_trail,
                            )
                        )
                    }
                    if inspect.isfunction(function)
                    else _semantic_value(
                        function,
                        local_module=local_module,
                        trail=next_trail,
                    )
                )
                continue
            if inspect.isbuiltin(member) or callable(member):
                members[name] = _semantic_value(
                    member,
                    local_module=local_module,
                    trail=next_trail,
                )
        classes.append({"class": base_name, "members": members})
    return {"external_class_mro": classes}


def _function_semantics(
    function,
    *,
    local_module: str,
    trail: frozenset[int] = frozenset(),
) -> str:
    identity = id(function)
    if identity in trail:
        canonical = json.dumps(
            {"function_cycle": function.__qualname__},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return sha256(canonical).hexdigest()
    next_trail = trail | {identity}
    referenced_names = tuple(sorted(set(function.__code__.co_names)))
    referenced_globals = {
        name: (
            _module_reference_semantics(
                function.__globals__[name],
                referenced_names=referenced_names,
                local_module=local_module,
                trail=next_trail,
            )
            if inspect.ismodule(function.__globals__[name])
            else {
                "external_class_semantics": _external_class_semantics(
                    function.__globals__[name],
                    local_module=local_module,
                    trail=next_trail,
                )
            }
            if (
                inspect.isclass(function.__globals__[name])
                and function.__globals__[name].__module__ != local_module
            )
            else {
                "function_semantics": _function_semantics(
                    function.__globals__[name],
                    local_module=local_module,
                    trail=next_trail,
                )
            }
            if inspect.isfunction(function.__globals__[name])
            else _semantic_value(
                function.__globals__[name],
                local_module=local_module,
                trail=next_trail,
            )
        )
        for name in referenced_names
        if name in function.__globals__
    }
    builtins_value = function.__builtins__
    builtins_namespace = (
        builtins_value
        if isinstance(builtins_value, Mapping)
        else vars(builtins_value)
    )
    referenced_builtins = {
        name: (
            {
                "function_semantics": _function_semantics(
                    builtins_namespace[name],
                    local_module=local_module,
                    trail=next_trail,
                )
            }
            if inspect.isfunction(builtins_namespace[name])
            else _semantic_value(
                builtins_namespace[name],
                local_module=local_module,
                trail=next_trail,
            )
        )
        for name in referenced_names
        if (
            name not in function.__globals__
            and name in builtins_namespace
        )
    }
    payload = {
        "function": _function_metadata_semantics(
            function,
            local_module=local_module,
            trail=next_trail,
        ),
        "referenced_builtins": referenced_builtins,
        "referenced_globals": referenced_globals,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


def _manifest_sha256(manifest: Mapping[str, str]) -> str:
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


def _property_semantics(
    value: property,
    *,
    local_module: str,
) -> str:
    payload: dict[str, object] = {}
    for role, function in (
        ("get", value.fget),
        ("set", value.fset),
        ("delete", value.fdel),
    ):
        payload[role] = (
            None
            if function is None
            else {
                "function_semantics": _function_semantics(
                    function,
                    local_module=local_module,
                )
            }
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


def _trunk_execution_manifest(trunk_type: type[nn.Module]) -> dict[str, str]:
    manifest: dict[str, str] = {}
    module_name = trunk_type.__module__
    for name in _TRUNK_EXECUTION_METHODS:
        method = vars(trunk_type).get(name)
        if isinstance(method, (classmethod, staticmethod)):
            method = method.__func__
        if not inspect.isfunction(method):
            return {}
        manifest[name] = _function_semantics(
            method,
            local_module=module_name,
        )
    feature_width = inspect.getattr_static(
        trunk_type,
        "feature_width",
        None,
    )
    if not isinstance(feature_width, property):
        return {}
    manifest["feature_width"] = _property_semantics(
        feature_width,
        local_module=module_name,
    )
    local_type_methods = {
        ByteAlignedResidualFeatures: (
            "__init__",
            "__post_init__",
            "__getattribute__",
        ),
        ShohinTrunkBatch: ("__getattribute__",),
    }
    for local_type, names in local_type_methods.items():
        for name in names:
            method = inspect.getattr_static(local_type, name, None)
            key = f"{local_type.__qualname__}.{name}"
            if inspect.isfunction(method):
                manifest[key] = _function_semantics(
                    method,
                    local_module=module_name,
                )
            else:
                canonical = json.dumps(
                    _semantic_value(
                        method,
                        local_module=module_name,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
                manifest[key] = sha256(canonical).hexdigest()
    return manifest


def _trunk_execution_configuration_sha256(
    *,
    block_indices: Sequence[int],
    max_payload_bytes: int,
) -> str:
    canonical = json.dumps(
        {
            "block_indices": [int(index) for index in block_indices],
            "max_payload_bytes": int(max_payload_bytes),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(canonical).hexdigest()


def _trunk_execution_semantics_match(trunk: nn.Module) -> bool:
    if type(trunk) is not FrozenShohinTrunk:
        return False
    if _module_has_runtime_override(trunk) or any(
        name in vars(trunk)
        for name in _TRUNK_EXECUTION_METHODS
    ):
        return False
    manifest = _trunk_execution_manifest(type(trunk))
    if (
        not manifest
        or _manifest_sha256(manifest)
        != EXPECTED_TRUNK_RUNTIME_MANIFEST_SHA256
    ):
        return False
    return trunk._execution_configuration_sha256 == (
        _trunk_execution_configuration_sha256(
            block_indices=trunk.block_indices,
            max_payload_bytes=trunk.max_payload_bytes,
        )
    )


def _code_manifest(module: object) -> dict[str, str]:
    manifest: dict[str, str] = {}
    module_name = getattr(module, "__name__", None)
    if not isinstance(module_name, str):
        return manifest
    for name, value in vars(module).items():
        if inspect.isfunction(value) and value.__module__ == module_name:
            manifest[name] = _function_semantics(
                value,
                local_module=module_name,
            )
        if (
            inspect.isclass(value)
            and value.__module__ == module_name
            and issubclass(value, nn.Module)
        ):
            for method_name, method in vars(value).items():
                if isinstance(method, property):
                    manifest[f"{value.__qualname__}.{method_name}"] = (
                        _property_semantics(
                            method,
                            local_module=module_name,
                        )
                    )
                    continue
                if isinstance(method, (classmethod, staticmethod)):
                    method = method.__func__
                if inspect.isfunction(method):
                    manifest[f"{value.__qualname__}.{method_name}"] = (
                        _function_semantics(
                            method,
                            local_module=module_name,
                        )
                    )
    return manifest


def _simple_runtime_attributes(module: nn.Module) -> dict[str, object]:
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and isinstance(value, (bool, float, int, str, type(None)))
    }


def _module_has_runtime_override(module: nn.Module) -> bool:
    if any(
        name in vars(module)
        for name in ("__call__", "_call_impl", "forward")
    ):
        return True
    if getattr(module, "_compiled_call_impl", None) is not None:
        return True
    return any(
        bool(value)
        for name, value in vars(module).items()
        if "hook" in name and isinstance(value, Mapping)
    )


def _runtime_semantics_match_source(
    parent: GPT,
    config_value: Mapping[str, object],
) -> bool:
    """Compare the executing module graph with a fresh load of model.py."""
    source_path_value = inspect.getsourcefile(GPT)
    if source_path_value is None:
        return False
    source_path = Path(source_path_value).resolve(strict=True)
    module_name = f"_efc_pristine_model_{sha256(source_path.read_bytes()).hexdigest()[:16]}"
    specification = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )
    if specification is None or specification.loader is None:
        return False
    pristine = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = pristine
    try:
        specification.loader.exec_module(pristine)
        current_manifest = _code_manifest(sys.modules[GPT.__module__])
        pristine_manifest = _code_manifest(pristine)
        if (
            current_manifest != pristine_manifest
            or _manifest_sha256(current_manifest)
            != EXPECTED_MODEL_RUNTIME_MANIFEST_SHA256
        ):
            return False
        with torch.random.fork_rng(devices=[]):
            reference = pristine.GPT(
                pristine.GPTConfig(**dict(config_value))
            )
        reference.eval()
        current_modules = dict(parent.named_modules())
        reference_modules = dict(reference.named_modules())
        if tuple(current_modules) != tuple(reference_modules):
            return False
        current_custom_module = GPT.__module__
        for name, current in current_modules.items():
            expected = reference_modules[name]
            current_type = type(current)
            expected_type = type(expected)
            if current_type.__module__ == current_custom_module:
                if (
                    expected_type.__module__ != module_name
                    or current_type.__qualname__
                    != expected_type.__qualname__
                ):
                    return False
            elif current_type is not expected_type:
                return False
            if (
                current.training
                or _module_has_runtime_override(current)
                or _simple_runtime_attributes(current)
                != _simple_runtime_attributes(expected)
            ):
                return False
        current_buffers = dict(parent.named_buffers())
        reference_buffers = dict(reference.named_buffers())
        if tuple(current_buffers) != tuple(reference_buffers):
            return False
        return all(
            value.shape == reference_buffers[name].shape
            and value.dtype == reference_buffers[name].dtype
            and torch.equal(
                value.detach().cpu(),
                reference_buffers[name].detach().cpu(),
            )
            for name, value in current_buffers.items()
        )
    finally:
        sys.modules.pop(module_name, None)


def _parent_matches_checkpoint(
    parent: GPT,
    checkpoint_path: Path | None,
    expected_sha256: str,
) -> bool:
    """Reverify the checkpoint bytes and every frozen parent tensor."""
    if checkpoint_path is None or not checkpoint_path.is_file():
        return False
    try:
        with checkpoint_path.open("rb") as handle:
            if _hash_checkpoint(handle) != expected_sha256:
                return False
            handle.seek(0)
            checkpoint = torch.load(
                handle,
                map_location="cpu",
                weights_only=True,
            )
        checkpoint = _checkpoint_mapping(checkpoint, "root")
        config_value = _checkpoint_mapping(checkpoint.get("cfg"), "cfg")
        model_value = _checkpoint_mapping(checkpoint.get("model"), "model")
        if asdict(parent.cfg) != dict(config_value):
            return False
        current = parent.state_dict()
        if tuple(model_value) != tuple(current):
            return False
        tensors_match = all(
            isinstance(model_value[name], torch.Tensor)
            and model_value[name].shape == value.shape
            and model_value[name].dtype == value.dtype
            and torch.equal(
                model_value[name].detach().cpu(),
                value.detach().cpu(),
            )
            for name, value in current.items()
        )
        return tensors_match and _runtime_semantics_match_source(
            parent,
            config_value,
        )
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


class FrozenShohinTrunk(nn.Module):
    """Frozen GPT prefix that exposes byte-aligned raw post-block residuals."""

    def __init__(
        self,
        parent: GPT,
        *,
        checkpoint_sha256: str,
        block_indices: Sequence[int],
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        super().__init__()
        if not isinstance(parent, GPT):
            raise FrozenShohinTrunkError("parent must be train.model.GPT")
        if parent.cfg.n_loop != 1:
            raise FrozenShohinTrunkError("frozen trunk requires n_loop=1")
        indices = tuple(int(index) for index in block_indices)
        if (
            not indices
            or tuple(sorted(set(indices))) != indices
            or any(index not in range(parent.cfg.n_layer) for index in indices)
        ):
            raise FrozenShohinTrunkError(
                "block indices must be unique increasing zero-based indices"
            )
        if _SHA256.fullmatch(checkpoint_sha256) is None:
            raise FrozenShohinTrunkError("checkpoint SHA-256 is not canonical")
        if not isinstance(max_payload_bytes, int) or max_payload_bytes < 1:
            raise FrozenShohinTrunkError("maximum payload bytes must be positive")
        self.parent = parent
        self.checkpoint_sha256 = checkpoint_sha256
        self._checkpoint_path: Path | None = None
        self.block_indices = indices
        self.max_payload_bytes = max_payload_bytes
        self._execution_configuration_sha256 = (
            _trunk_execution_configuration_sha256(
                block_indices=indices,
                max_payload_bytes=max_payload_bytes,
            )
        )
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.parent.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str,
        block_indices: Sequence[int],
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        device: torch.device | str = "cpu",
    ) -> "FrozenShohinTrunk":
        if _SHA256.fullmatch(expected_sha256) is None:
            raise FrozenShohinTrunkError("expected checkpoint SHA-256 is not canonical")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FrozenShohinTrunkError("checkpoint path is not a file")
        with path.open("rb") as handle:
            observed_sha256 = _hash_checkpoint(handle)
            if observed_sha256 != expected_sha256:
                raise FrozenShohinTrunkError(
                    "checkpoint SHA-256 mismatch: "
                    f"{observed_sha256} != {expected_sha256}"
                )
            handle.seek(0)
            checkpoint = torch.load(
                handle,
                map_location="cpu",
                weights_only=True,
            )
        checkpoint = _checkpoint_mapping(checkpoint, "root")
        config_value = _checkpoint_mapping(checkpoint.get("cfg"), "cfg")
        model_value = _checkpoint_mapping(checkpoint.get("model"), "model")
        try:
            config = GPTConfig(**dict(config_value))
            parent = GPT(config)
            parent.load_state_dict(model_value, strict=True)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise FrozenShohinTrunkError(
                "checkpoint does not strictly instantiate train.model.GPT"
            ) from exc
        parent.to(device)
        trunk = cls(
            parent,
            checkpoint_sha256=observed_sha256,
            block_indices=block_indices,
            max_payload_bytes=max_payload_bytes,
        )
        trunk._checkpoint_path = path.resolve(strict=True)
        return trunk

    def train(self, mode: bool = True) -> "FrozenShohinTrunk":
        super().train(mode)
        self.parent.eval()
        return self

    def parameter_receipt(self) -> ShohinTrunkParameterReceipt:
        checkpoint_verified = _parent_matches_checkpoint(
            self.parent,
            self._checkpoint_path,
            self.checkpoint_sha256,
        ) and _trunk_execution_semantics_match(self)
        parent_ids = {id(parameter) for parameter in self.parent.parameters()}
        parent_count = sum(
            parameter.numel()
            for parameter in self.parent.parameters()
        )
        integrated_count = sum(parameter.numel() for parameter in self.parameters())
        adapter_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in parent_ids
        )
        trainable_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return ShohinTrunkParameterReceipt(
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_verified=checkpoint_verified,
            parent_unique_parameters=parent_count,
            adapter_unique_parameters=adapter_count,
            integrated_unique_parameters=integrated_count,
            trainable_unique_parameters=trainable_count,
        )

    @property
    def feature_width(self) -> int:
        return len(self.block_indices) * self.parent.cfg.d_model

    def encode_batch(
        self,
        batch: ShohinTrunkBatch,
    ) -> ByteAlignedResidualFeatures:
        if not isinstance(batch, ShohinTrunkBatch):
            raise FrozenShohinTrunkError(
                "frozen trunk input must be a ShohinTrunkBatch"
            )
        return self(
            batch.payloads,
            batch.token_ids,
            batch.token_valid,
            batch.token_byte_bounds,
        )

    def flatten_byte_features(
        self,
        features: ByteAlignedResidualFeatures,
    ) -> torch.Tensor:
        if (
            not isinstance(features, ByteAlignedResidualFeatures)
            or features.block_indices != self.block_indices
            or features.byte_features.shape[1] != len(self.block_indices)
            or features.byte_features.shape[-1]
            != self.parent.cfg.d_model
        ):
            raise FrozenShohinTrunkError(
                "byte features do not belong to this frozen trunk"
            )
        batch, layers, byte_count, width = (
            features.byte_features.shape
        )
        return (
            features.byte_features.permute(0, 2, 1, 3)
            .reshape(batch, byte_count, layers * width)
            .contiguous()
        )

    def _validate_inputs(
        self,
        payloads: Sequence[bytes],
        token_ids: torch.Tensor,
        token_valid: torch.Tensor,
        token_byte_bounds: torch.Tensor,
    ) -> tuple[tuple[bytes, ...], tuple[int, ...]]:
        frozen_payloads = tuple(payloads)
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise FrozenShohinTrunkError("token IDs must be rank-two long")
        batch, tokens = token_ids.shape
        if (
            not batch
            or not tokens
            or tokens > self.max_payload_bytes
        ):
            raise FrozenShohinTrunkError("token geometry leaves frozen trunk support")
        if len(frozen_payloads) != batch or any(
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > self.max_payload_bytes
            for payload in frozen_payloads
        ):
            raise FrozenShohinTrunkError(
                "payload batch is empty, malformed, or outside byte support"
            )
        if token_valid.shape != (batch, tokens) or token_valid.dtype != torch.bool:
            raise FrozenShohinTrunkError("token validity geometry differs")
        if (
            token_byte_bounds.shape != (batch, tokens, 2)
            or token_byte_bounds.dtype
            not in (torch.int32, torch.int64)
        ):
            raise FrozenShohinTrunkError("token byte bounds geometry differs")
        if token_ids.numel() and (
            int(token_ids.min()) < 0
            or int(token_ids.max()) >= self.parent.cfg.vocab_size
        ):
            raise FrozenShohinTrunkError("token ID leaves parent vocabulary")

        valid_cpu = token_valid.detach().cpu()
        bounds_cpu = token_byte_bounds.detach().cpu()
        valid_counts: list[int] = []
        for row, payload in enumerate(frozen_payloads):
            count = int(valid_cpu[row].sum())
            if count < 1 or not bool(valid_cpu[row, :count].all()) or bool(
                valid_cpu[row, count:].any()
            ):
                raise FrozenShohinTrunkError(
                    "token validity must be one nonempty contiguous prefix"
                )
            if bool(bounds_cpu[row, count:].ne(0).any()):
                raise FrozenShohinTrunkError(
                    "invalid token byte bounds must be zero"
                )
            coverage = torch.zeros(len(payload), dtype=torch.int16)
            for token in range(count):
                start = int(bounds_cpu[row, token, 0])
                end = int(bounds_cpu[row, token, 1])
                if not 0 <= start < end <= len(payload):
                    raise FrozenShohinTrunkError(
                        "valid token byte bound leaves payload"
                    )
                coverage[start:end] += 1
            if bool(coverage.eq(0).any()):
                raise FrozenShohinTrunkError(
                    "token byte bounds leave uncovered payload bytes"
                )
            if bool(coverage.gt(1).any()):
                raise FrozenShohinTrunkError(
                    "token byte bounds overlap"
                )
            valid_counts.append(count)
        return frozen_payloads, tuple(valid_counts)

    @torch.no_grad()
    def forward(
        self,
        payloads: Sequence[bytes],
        token_ids: torch.Tensor,
        token_valid: torch.Tensor,
        token_byte_bounds: torch.Tensor,
    ) -> ByteAlignedResidualFeatures:
        frozen_payloads, valid_counts = self._validate_inputs(
            payloads,
            token_ids,
            token_valid,
            token_byte_bounds,
        )
        device = next(self.parent.parameters()).device
        ids = token_ids.to(device=device)
        valid = token_valid.to(device=device)
        bounds_cpu = token_byte_bounds.detach().cpu()

        batch, tokens = ids.shape
        token_features = torch.zeros(
            (
                batch,
                len(self.block_indices),
                tokens,
                self.parent.cfg.d_model,
            ),
            dtype=self.parent.tok.weight.dtype,
            device=device,
        )
        wanted = set(self.block_indices)
        for row, count in enumerate(valid_counts):
            for left in range(0, count, self.parent.cfg.seq_len):
                right = min(count, left + self.parent.cfg.seq_len)
                window = ids[row : row + 1, left:right]
                hidden = self.parent.tok(window)
                cos = self.parent.cos[: right - left].to(
                    device=hidden.device
                )
                sin = self.parent.sin[: right - left].to(
                    device=hidden.device
                )
                captured: list[torch.Tensor] = []
                for index, block in enumerate(self.parent.blocks):
                    hidden, _ = block(hidden, cos, sin)
                    if index in wanted:
                        captured.append(hidden)
                if len(captured) != len(self.block_indices):
                    raise FrozenShohinTrunkError(
                        "requested parent residual was not captured"
                    )
                token_features[
                    row,
                    :,
                    left:right,
                ] = torch.stack(captured, dim=1)[0]
        token_features = token_features * valid[:, None, :, None]

        maximum_bytes = max(len(payload) for payload in frozen_payloads)
        byte_features = torch.zeros(
            (
                len(frozen_payloads),
                len(self.block_indices),
                maximum_bytes,
                self.parent.cfg.d_model,
            ),
            dtype=token_features.dtype,
            device=token_features.device,
        )
        byte_valid = torch.zeros(
            (len(frozen_payloads), maximum_bytes),
            dtype=torch.bool,
            device=token_features.device,
        )
        for row, payload in enumerate(frozen_payloads):
            byte_valid[row, : len(payload)] = True
            for token in range(valid_counts[row]):
                start = int(bounds_cpu[row, token, 0])
                end = int(bounds_cpu[row, token, 1])
                byte_features[row, :, start:end] = token_features[
                    row, :, token
                ][:, None]

        return ByteAlignedResidualFeatures(
            block_indices=self.block_indices,
            token_features=token_features,
            token_valid=valid,
            byte_features=byte_features,
            byte_valid=byte_valid,
            payload_lengths=torch.tensor(
                tuple(len(payload) for payload in frozen_payloads),
                dtype=torch.long,
                device=token_features.device,
            ),
        )


__all__ = [
    "ByteAlignedResidualFeatures",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "FrozenShohinTrunk",
    "FrozenShohinTrunkError",
    "ShohinTrunkBatch",
    "ShohinTrunkParameterReceipt",
]
