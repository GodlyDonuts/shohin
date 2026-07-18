#!/usr/bin/env python3
"""Build the fail-closed DWS single-completion development bundle.

This CPU-only generator creates matched full-trace, decomposed one-step, and
multiline-sham data from the same width-four episodes. It also creates exact
fixed-lane token packs, a public development board and commitment, and a frozen
cross-width replication board. It does not train or evaluate a model.
"""

from __future__ import annotations

import _ctypes as _ctypes_native
import _hashlib as _hashlib_native
import _json as _json_native
import _random as _random_native
import _sre as _sre_native
import _stat as _stat_native
import _struct as _struct_native
import argparse
import builtins as builtins_module
import collections as collections_module
from collections import Counter, defaultdict, deque
import ctypes
import dataclasses as dataclasses_module
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import marshal
import math
import os
import pathlib as pathlib_module
from pathlib import Path
import platform
import posix as _posix_native
import random
import re
import stat
import struct
import sys
import sysconfig
import types as types_module
from types import MappingProxyType
import typing as typing_module
from typing import Any, Iterable


_BOUND_SYS_MODULES = sys.modules
_BOUND_SYS_PATH = sys.path
_BOUND_SYSCONFIG_GET_PATH = sysconfig.get_path


class ContractError(ValueError):
    """Raised when generation would weaken or violate the frozen protocol."""


_STARTUP_FLAG_NAMES = (
    "debug",
    "inspect",
    "interactive",
    "optimize",
    "dont_write_bytecode",
    "no_user_site",
    "no_site",
    "ignore_environment",
    "verbose",
    "bytes_warning",
    "quiet",
    "hash_randomization",
    "isolated",
    "dev_mode",
    "utf8_mode",
    "warn_default_encoding",
    "safe_path",
    "int_max_str_digits",
)
_BOUND_SYS_FLAGS = sys.flags


def _startup_flags_receipt() -> dict[str, Any]:
    return {
        "repr": repr(sys.flags),
        "tuple": list(sys.flags),
        "values": {name: getattr(sys.flags, name) for name in _STARTUP_FLAG_NAMES},
    }


def _assert_isolated_startup() -> None:
    if sys.flags is not _BOUND_SYS_FLAGS:
        raise ContractError("Python startup flags object changed")
    required = {
        "dont_write_bytecode": 1,
        "no_user_site": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "safe_path": True,
    }
    for name, expected in required.items():
        if getattr(sys.flags, name) != expected:
            raise ContractError(
                "isolated Python startup requires -I -S -B: {}".format(name)
            )
    if not sys.dont_write_bytecode:
        raise ContractError("isolated Python startup must disable bytecode writes")
    forbidden_startup_modules = {"site", "sitecustomize", "usercustomize"}
    preloaded = forbidden_startup_modules.intersection(_BOUND_SYS_MODULES)
    if preloaded:
        raise ContractError(
            "site startup modules are forbidden: " + ", ".join(sorted(preloaded))
        )


_assert_isolated_startup()
_BOUND_STARTUP_FLAGS_RECEIPT = _startup_flags_receipt()
try:
    _ISOLATED_PURELIB = Path(_BOUND_SYSCONFIG_GET_PATH("purelib")).resolve(strict=True)
except OSError as error:
    raise ContractError("isolated purelib directory cannot be resolved") from error
if not _ISOLATED_PURELIB.is_dir():
    raise ContractError("isolated purelib path is not a directory")
if str(_ISOLATED_PURELIB) not in _BOUND_SYS_PATH:
    _BOUND_SYS_PATH.append(str(_ISOLATED_PURELIB))


ROOT = Path(__file__).resolve().parents[1]
_BOUND_GENERATOR_MODULE = _BOUND_SYS_MODULES.get(__name__)
if _BOUND_GENERATOR_MODULE is None:
    raise ContractError("executing generator module is not registered")
_BOUND_GENERATOR_EXECUTION_NAME = __name__
_BOUND_GENERATOR_MODULE_FILE = getattr(_BOUND_GENERATOR_MODULE, "__file__", None)
_BOUND_GENERATOR_MODULE_SPEC = getattr(_BOUND_GENERATOR_MODULE, "__spec__", None)
_BOUND_GENERATOR_DIRECT_SCRIPT = (
    _BOUND_GENERATOR_EXECUTION_NAME == "__main__"
    and _BOUND_GENERATOR_MODULE_SPEC is None
)

_BOUND_STRUCT_MODULE = struct
_BOUND_STRUCT_NATIVE_MODULE = _struct_native
_BOUND_STRUCT_PACK = struct.pack
_BOUND_STRUCT_UNPACK = struct.unpack
_BOUND_HASHLIB_SHA256 = hashlib.sha256
_BOUND_MARSHAL_DUMPS = marshal.dumps
_BOUND_SYS_PLATFORM = sys.platform
_BOUND_SYS_ADDAUDITHOOK = sys.addaudithook
_BOUND_SYS_AUDIT = sys.audit
_BOUND_BUILTIN_IMPORT = builtins_module.__import__
_BOUND_OS_PATH_MODULE = os.path
_BOUND_ARGUMENT_PARSER = argparse.ArgumentParser
_BOUND_IMPORTLIB_UTIL_MODULE = importlib.util
_BOUND_IMPORTLIB_IMPORT_MODULE = importlib.import_module
_BOUND_IMPORTLIB_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_BOUND_IMPORTLIB_MODULE_FROM_SPEC = importlib.util.module_from_spec
_BOUND_IMPORTLIB_METADATA_VERSION = importlib_metadata.version
_BOUND_IMPORTLIB_METADATA_DISTRIBUTION = importlib_metadata.distribution
_BOUND_IMPORTLIB_METADATA_DISTRIBUTION_CLASS = importlib_metadata.Distribution
_BOUND_IMPORTLIB_METADATA_PACKAGE_NOT_FOUND_ERROR = (
    importlib_metadata.PackageNotFoundError
)
_BOUND_JSON_DUMPS = json.dumps
_BOUND_JSON_LOADS = json.loads
_BOUND_JSON_DECODE_ERROR = json.JSONDecodeError
_BOUND_COLLECTIONS_NAMEDTUPLE = collections_module.namedtuple
_BOUND_CTYPES_PYDLL = ctypes.PyDLL
_BOUND_MATH_ISFINITE = math.isfinite
_BOUND_PLATFORM_PYTHON_BUILD = platform.python_build
_BOUND_PLATFORM_PYTHON_COMPILER = platform.python_compiler
_BOUND_PLATFORM_PYTHON_IMPLEMENTATION = platform.python_implementation
_BOUND_PYTHON_BUILD = _BOUND_PLATFORM_PYTHON_BUILD()
_BOUND_PYTHON_COMPILER = _BOUND_PLATFORM_PYTHON_COMPILER()
_BOUND_PYTHON_IMPLEMENTATION = _BOUND_PLATFORM_PYTHON_IMPLEMENTATION()
_BOUND_RE_COMPILE = re.compile

try:
    _BOUND_JSON_DECODER_MODULE = _BOUND_SYS_MODULES["json.decoder"]
    _BOUND_JSON_ENCODER_MODULE = _BOUND_SYS_MODULES["json.encoder"]
    _BOUND_JSON_SCANNER_MODULE = _BOUND_SYS_MODULES["json.scanner"]
    _BOUND_IMPORTLIB_METADATA_ADAPTERS_MODULE = _BOUND_SYS_MODULES[
        "importlib.metadata._adapters"
    ]
    _BOUND_IMPORTLIB_METADATA_COLLECTIONS_MODULE = _BOUND_SYS_MODULES[
        "importlib.metadata._collections"
    ]
    _BOUND_IMPORTLIB_METADATA_FUNCTOOLS_MODULE = _BOUND_SYS_MODULES[
        "importlib.metadata._functools"
    ]
    _BOUND_IMPORTLIB_METADATA_ITERTOOLS_MODULE = _BOUND_SYS_MODULES[
        "importlib.metadata._itertools"
    ]
    _BOUND_IMPORTLIB_METADATA_META_MODULE = _BOUND_SYS_MODULES[
        "importlib.metadata._meta"
    ]
    _BOUND_IMPORTLIB_METADATA_TEXT_MODULE = _BOUND_SYS_MODULES[
        "importlib.metadata._text"
    ]
except KeyError as error:
    raise ContractError("required runtime dependency module was not loaded") from error


def _implementation_value_bytes(value: Any) -> bytes:
    if value is None:
        return b"none"
    if value is Ellipsis:
        return b"ellipsis"
    if value is NotImplemented:
        return b"not-implemented"
    if type(value) is bool:
        return b"bool:1" if value else b"bool:0"
    if type(value) is int:
        return b"int:" + str(value).encode("ascii")
    if type(value) is float:
        return b"float:" + _BOUND_STRUCT_PACK(">d", value)
    if type(value) is complex:
        return b"complex:" + _BOUND_STRUCT_PACK(">dd", value.real, value.imag)
    if type(value) is str:
        payload = value.encode("utf-8")
        return b"str:" + len(payload).to_bytes(8, "big") + payload
    if type(value) is bytes:
        return b"bytes:" + len(value).to_bytes(8, "big") + value
    if type(value) is types_module.CodeType:
        return _stable_code_bytes(value)
    if isinstance(value, tuple):
        parts = [_implementation_value_bytes(item) for item in value]
        return "tuple:{}.{}:".format(
            type(value).__module__, type(value).__qualname__
        ).encode("ascii") + b"".join(
            len(part).to_bytes(8, "big") + part for part in parts
        )
    if type(value) is list:
        parts = [_implementation_value_bytes(item) for item in value]
        return b"list:" + b"".join(
            len(part).to_bytes(8, "big") + part for part in parts
        )
    if type(value) is dict:
        parts = [
            (_implementation_value_bytes(key), _implementation_value_bytes(item))
            for key, item in value.items()
        ]
        parts.sort(key=lambda pair: pair[0])
        return b"dict:" + b"".join(
            len(key).to_bytes(8, "big") + key + len(item).to_bytes(8, "big") + item
            for key, item in parts
        )
    if type(value) is frozenset or type(value) is set:
        parts = sorted(_implementation_value_bytes(item) for item in value)
        return (b"frozenset:" if type(value) is frozenset else b"set:") + b"".join(
            len(part).to_bytes(8, "big") + part for part in parts
        )
    try:
        return b"marshal:" + _BOUND_MARSHAL_DUMPS(value)
    except (TypeError, ValueError):
        pass
    if type(value) is type:
        return ("type:{}:{}".format(value.__module__, value.__qualname__)).encode(
            "ascii"
        )
    if type(value) is object:
        return b"opaque-sentinel:builtins.object"
    if isinstance(value, types_module.ModuleType):
        return ("module:{}:{}".format(type(value).__qualname__, value.__name__)).encode(
            "ascii"
        )
    if type(value) is types_module.FunctionType:
        return ("function:{}:{}".format(value.__module__, value.__qualname__)).encode(
            "ascii"
        )
    if callable(value):
        callable_module = getattr(value, "__module__", None)
        if type(callable_module) is not str:
            callable_module = type(value).__module__
        callable_name = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__qualname__),
        )
        if type(callable_name) is not str:
            callable_name = type(value).__qualname__
        if type(callable_module) is str and type(callable_name) is str:
            return (
                "native-callable:{}:{}:{}:{}".format(
                    type(value).__module__,
                    type(value).__qualname__,
                    callable_module,
                    callable_name,
                )
            ).encode("ascii")
    descriptor_owner = getattr(value, "__objclass__", None)
    descriptor_name = getattr(value, "__name__", None)
    if type(descriptor_owner) is type and type(descriptor_name) is str:
        return (
            "native-descriptor:{}:{}:{}:{}".format(
                type(value).__module__,
                type(value).__qualname__,
                descriptor_owner.__module__ + "." + descriptor_owner.__qualname__,
                descriptor_name,
            )
        ).encode("ascii")
    raise ContractError(
        "callable implementation state has an unsupported exact type: "
        + type(value).__module__
        + "."
        + type(value).__qualname__
    )


def _stable_code_bytes(code: types_module.CodeType) -> bytes:
    return b"code-v2:" + _implementation_value_bytes(
        (
            code.co_argcount,
            code.co_posonlyargcount,
            code.co_kwonlyargcount,
            code.co_nlocals,
            code.co_stacksize,
            code.co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_qualname,
            code.co_firstlineno,
            code.co_linetable,
            code.co_exceptiontable,
            code.co_freevars,
            code.co_cellvars,
        )
    )


def _implementation_value_sha256(value: Any) -> str:
    return _BOUND_HASHLIB_SHA256(_implementation_value_bytes(value)).hexdigest()


def _capture_value_identities(value: Any) -> tuple[Any, ...]:
    if type(value) is tuple:
        return (tuple, tuple(_capture_value_identities(item) for item in value))
    if type(value) is list:
        return (list, tuple(_capture_value_identities(item) for item in value))
    if type(value) is dict:
        return (
            dict,
            tuple(
                (key, _capture_value_identities(item)) for key, item in value.items()
            ),
        )
    return (type(value), value)


def _value_identities_match(value: Any, expected: tuple[Any, ...]) -> bool:
    expected_type, snapshot = expected
    if type(value) is not expected_type:
        return False
    if expected_type is tuple or expected_type is list:
        return len(value) == len(snapshot) and all(
            _value_identities_match(item, expected_item)
            for item, expected_item in zip(value, snapshot, strict=True)
        )
    if expected_type is dict:
        if tuple(value) != tuple(key for key, _ in snapshot):
            return False
        return all(
            key in value and _value_identities_match(value[key], expected_item)
            for key, expected_item in snapshot
        )
    return value is snapshot


def _capture_callable_implementation(value: Any) -> tuple[Any, ...]:
    code = getattr(value, "__code__", None)
    defaults = getattr(value, "__defaults__", None)
    kwdefaults = getattr(value, "__kwdefaults__", None)
    closure = getattr(value, "__closure__", None)
    closure_cells = None
    if closure is not None:
        closure_cells = tuple(cell.cell_contents for cell in closure)
    return (
        type(value),
        code,
        defaults,
        _implementation_value_sha256(defaults),
        _capture_value_identities(defaults),
        kwdefaults,
        _implementation_value_sha256(kwdefaults),
        _capture_value_identities(kwdefaults),
        closure,
        closure_cells,
        _implementation_value_sha256(closure_cells),
        _capture_value_identities(closure_cells),
    )


def _assert_callable_implementation(
    label: str, value: Any, expected: tuple[Any, ...]
) -> None:
    (
        expected_type,
        expected_code,
        expected_defaults,
        expected_defaults_sha256,
        expected_defaults_identities,
        expected_kwdefaults,
        expected_kwdefaults_sha256,
        expected_kwdefaults_identities,
        expected_closure,
        expected_closure_cells,
        expected_closure_sha256,
        expected_closure_identities,
    ) = expected
    if type(value) is not expected_type:
        raise ContractError("bound callable type changed: " + label)
    if getattr(value, "__code__", None) is not expected_code:
        raise ContractError("bound callable implementation changed: " + label)
    if getattr(value, "__defaults__", None) is not expected_defaults:
        raise ContractError("bound callable defaults changed: " + label)
    if _implementation_value_sha256(expected_defaults) != expected_defaults_sha256:
        raise ContractError("bound callable defaults mutated: " + label)
    if not _value_identities_match(expected_defaults, expected_defaults_identities):
        raise ContractError("bound callable default identities changed: " + label)
    if getattr(value, "__kwdefaults__", None) is not expected_kwdefaults:
        raise ContractError("bound callable keyword defaults changed: " + label)
    if _implementation_value_sha256(expected_kwdefaults) != expected_kwdefaults_sha256:
        raise ContractError("bound callable keyword defaults mutated: " + label)
    if not _value_identities_match(expected_kwdefaults, expected_kwdefaults_identities):
        raise ContractError(
            "bound callable keyword default identities changed: " + label
        )
    closure = getattr(value, "__closure__", None)
    if closure is not expected_closure:
        raise ContractError("bound callable closure changed: " + label)
    if closure is not None:
        live_cells = tuple(cell.cell_contents for cell in closure)
        if len(live_cells) != len(expected_closure_cells) or any(
            live is not frozen
            for live, frozen in zip(live_cells, expected_closure_cells, strict=True)
        ):
            raise ContractError("bound callable closure contents changed: " + label)
    if _implementation_value_sha256(expected_closure_cells) != expected_closure_sha256:
        raise ContractError("bound callable closure contents mutated: " + label)
    if not _value_identities_match(expected_closure_cells, expected_closure_identities):
        raise ContractError("bound callable closure identities changed: " + label)


_RUNTIME_EXPORT_SPECS = {
    "argparse.ArgumentParser": (argparse, "ArgumentParser"),
    "collections.namedtuple": (collections_module, "namedtuple"),
    "ctypes.PyDLL": (ctypes, "PyDLL"),
    "importlib.import_module": (importlib, "import_module"),
    "importlib.util": (importlib, "util"),
    "importlib.util.module_from_spec": (
        _BOUND_IMPORTLIB_UTIL_MODULE,
        "module_from_spec",
    ),
    "importlib.util.spec_from_file_location": (
        _BOUND_IMPORTLIB_UTIL_MODULE,
        "spec_from_file_location",
    ),
    "importlib.metadata.Distribution": (importlib_metadata, "Distribution"),
    "importlib.metadata.PackageNotFoundError": (
        importlib_metadata,
        "PackageNotFoundError",
    ),
    "importlib.metadata.distribution": (importlib_metadata, "distribution"),
    "importlib.metadata.version": (importlib_metadata, "version"),
    "json.JSONDecodeError": (json, "JSONDecodeError"),
    "json.JSONDecoder": (json, "JSONDecoder"),
    "json.JSONEncoder": (json, "JSONEncoder"),
    "json._default_decoder": (json, "_default_decoder"),
    "json._default_encoder": (json, "_default_encoder"),
    "json.dumps": (json, "dumps"),
    "json.loads": (json, "loads"),
    "math.isfinite": (math, "isfinite"),
    "platform.python_build": (platform, "python_build"),
    "platform.python_compiler": (platform, "python_compiler"),
    "platform.python_implementation": (platform, "python_implementation"),
    "re.compile": (re, "compile"),
    "sysconfig.get_path": (sysconfig, "get_path"),
    "json.decoder.JSONDecodeError": (
        _BOUND_JSON_DECODER_MODULE,
        "JSONDecodeError",
    ),
    "json.decoder.JSONDecoder": (_BOUND_JSON_DECODER_MODULE, "JSONDecoder"),
    "json.decoder.c_scanstring": (_BOUND_JSON_DECODER_MODULE, "c_scanstring"),
    "json.decoder.py_scanstring": (_BOUND_JSON_DECODER_MODULE, "py_scanstring"),
    "json.decoder.scanstring": (_BOUND_JSON_DECODER_MODULE, "scanstring"),
    "json.encoder.JSONEncoder": (_BOUND_JSON_ENCODER_MODULE, "JSONEncoder"),
    "json.encoder._make_iterencode": (
        _BOUND_JSON_ENCODER_MODULE,
        "_make_iterencode",
    ),
    "json.encoder.c_encode_basestring": (
        _BOUND_JSON_ENCODER_MODULE,
        "c_encode_basestring",
    ),
    "json.encoder.c_encode_basestring_ascii": (
        _BOUND_JSON_ENCODER_MODULE,
        "c_encode_basestring_ascii",
    ),
    "json.encoder.c_make_encoder": (_BOUND_JSON_ENCODER_MODULE, "c_make_encoder"),
    "json.encoder.encode_basestring": (
        _BOUND_JSON_ENCODER_MODULE,
        "encode_basestring",
    ),
    "json.encoder.encode_basestring_ascii": (
        _BOUND_JSON_ENCODER_MODULE,
        "encode_basestring_ascii",
    ),
    "json.scanner.c_make_scanner": (_BOUND_JSON_SCANNER_MODULE, "c_make_scanner"),
    "json.scanner.make_scanner": (_BOUND_JSON_SCANNER_MODULE, "make_scanner"),
    "json.scanner.py_make_scanner": (
        _BOUND_JSON_SCANNER_MODULE,
        "py_make_scanner",
    ),
    "_json.encode_basestring": (_json_native, "encode_basestring"),
    "_json.encode_basestring_ascii": (_json_native, "encode_basestring_ascii"),
    "_json.make_encoder": (_json_native, "make_encoder"),
    "_json.make_scanner": (_json_native, "make_scanner"),
    "_json.scanstring": (_json_native, "scanstring"),
}
_BOUND_RUNTIME_EXPORTS = MappingProxyType(
    {
        label: (
            owner,
            attribute,
            getattr(owner, attribute),
            (
                _capture_callable_implementation(getattr(owner, attribute))
                if callable(getattr(owner, attribute))
                else None
            ),
        )
        for label, (owner, attribute) in _RUNTIME_EXPORT_SPECS.items()
    }
)
_BOUND_RUNTIME_CONSUMED_EXPORT_GLOBALS = MappingProxyType(
    {
        "argparse.ArgumentParser": "_BOUND_ARGUMENT_PARSER",
        "collections.namedtuple": "_BOUND_COLLECTIONS_NAMEDTUPLE",
        "ctypes.PyDLL": "_BOUND_CTYPES_PYDLL",
        "importlib.import_module": "_BOUND_IMPORTLIB_IMPORT_MODULE",
        "importlib.util.module_from_spec": "_BOUND_IMPORTLIB_MODULE_FROM_SPEC",
        "importlib.util.spec_from_file_location": (
            "_BOUND_IMPORTLIB_SPEC_FROM_FILE_LOCATION"
        ),
        "importlib.metadata.Distribution": (
            "_BOUND_IMPORTLIB_METADATA_DISTRIBUTION_CLASS"
        ),
        "importlib.metadata.PackageNotFoundError": (
            "_BOUND_IMPORTLIB_METADATA_PACKAGE_NOT_FOUND_ERROR"
        ),
        "importlib.metadata.distribution": ("_BOUND_IMPORTLIB_METADATA_DISTRIBUTION"),
        "importlib.metadata.version": "_BOUND_IMPORTLIB_METADATA_VERSION",
        "json.JSONDecodeError": "_BOUND_JSON_DECODE_ERROR",
        "json.dumps": "_BOUND_JSON_DUMPS",
        "json.loads": "_BOUND_JSON_LOADS",
        "math.isfinite": "_BOUND_MATH_ISFINITE",
        "platform.python_build": "_BOUND_PLATFORM_PYTHON_BUILD",
        "platform.python_compiler": "_BOUND_PLATFORM_PYTHON_COMPILER",
        "platform.python_implementation": ("_BOUND_PLATFORM_PYTHON_IMPLEMENTATION"),
        "re.compile": "_BOUND_RE_COMPILE",
        "sysconfig.get_path": "_BOUND_SYSCONFIG_GET_PATH",
    }
)

_BOUND_GENERATOR_CALLABLE_ALIASES = MappingProxyType(
    {
        "collections.Counter": ("Counter", Counter),
        "collections.defaultdict": ("defaultdict", defaultdict),
        "collections.deque": ("deque", deque),
        "pathlib.Path": ("Path", Path),
    }
)

_SERIALIZATION_CLASS_METHOD_SPECS = {
    "json.JSONDecoder.__init__": (json.JSONDecoder, "__init__"),
    "json.JSONDecoder.decode": (json.JSONDecoder, "decode"),
    "json.JSONDecoder.raw_decode": (json.JSONDecoder, "raw_decode"),
    "json.JSONEncoder.__init__": (json.JSONEncoder, "__init__"),
    "json.JSONEncoder.default": (json.JSONEncoder, "default"),
    "json.JSONEncoder.encode": (json.JSONEncoder, "encode"),
    "json.JSONEncoder.iterencode": (json.JSONEncoder, "iterencode"),
}
_BOUND_SERIALIZATION_CLASS_METHODS = MappingProxyType(
    {
        label: (
            owner,
            attribute,
            owner.__dict__[attribute],
            _capture_callable_implementation(owner.__dict__[attribute]),
        )
        for label, (owner, attribute) in _SERIALIZATION_CLASS_METHOD_SPECS.items()
    }
)


_FILESYSTEM_EXPORT_SPECS = {
    "os.close": (os, "close"),
    "os.fchmod": (os, "fchmod"),
    "os.fsencode": (os, "fsencode"),
    "os.fspath": (os, "fspath"),
    "os.fstat": (os, "fstat"),
    "os.fsync": (os, "fsync"),
    "os.geteuid": (os, "geteuid"),
    "os.listdir": (os, "listdir"),
    "os.mkdir": (os, "mkdir"),
    "os.open": (os, "open"),
    "os.path.abspath": (_BOUND_OS_PATH_MODULE, "abspath"),
    "os.read": (os, "read"),
    "os.replace": (os, "replace"),
    "os.stat": (os, "stat"),
    "os.strerror": (os, "strerror"),
    "os.write": (os, "write"),
    "fcntl.flock": (fcntl, "flock"),
    "ctypes.get_errno": (ctypes, "get_errno"),
    "stat.S_IMODE": (stat, "S_IMODE"),
    "stat.S_ISDIR": (stat, "S_ISDIR"),
    "stat.S_ISLNK": (stat, "S_ISLNK"),
    "stat.S_ISREG": (stat, "S_ISREG"),
}
_BOUND_FILESYSTEM_EXPORTS = MappingProxyType(
    {
        label: (
            owner,
            attribute,
            getattr(owner, attribute),
            _capture_callable_implementation(getattr(owner, attribute)),
        )
        for label, (owner, attribute) in _FILESYSTEM_EXPORT_SPECS.items()
    }
)

_BOUND_OS_CLOSE = os.close
_BOUND_OS_FCHMOD = os.fchmod
_BOUND_OS_FSENCODE = os.fsencode
_BOUND_OS_FSPATH = os.fspath
_BOUND_OS_FSTAT = os.fstat
_BOUND_OS_FSYNC = os.fsync
_BOUND_OS_GETEUID = os.geteuid
_BOUND_OS_LISTDIR = os.listdir
_BOUND_OS_MKDIR = os.mkdir
_BOUND_OS_OPEN = os.open
_BOUND_OS_PATH_ABSPATH = _BOUND_OS_PATH_MODULE.abspath
_BOUND_OS_READ = os.read
_BOUND_OS_REPLACE = os.replace
_BOUND_OS_STAT = os.stat
_BOUND_OS_STRERROR = os.strerror
_BOUND_OS_WRITE = os.write
_BOUND_FCNTL_FLOCK = fcntl.flock
_BOUND_CTYPES_GET_ERRNO = ctypes.get_errno
_BOUND_STAT_S_IMODE = stat.S_IMODE
_BOUND_STAT_S_ISDIR = stat.S_ISDIR
_BOUND_STAT_S_ISLNK = stat.S_ISLNK
_BOUND_STAT_S_ISREG = stat.S_ISREG
_BOUND_FILESYSTEM_GLOBAL_NAMES = MappingProxyType(
    {
        "os.close": "_BOUND_OS_CLOSE",
        "os.fchmod": "_BOUND_OS_FCHMOD",
        "os.fsencode": "_BOUND_OS_FSENCODE",
        "os.fspath": "_BOUND_OS_FSPATH",
        "os.fstat": "_BOUND_OS_FSTAT",
        "os.fsync": "_BOUND_OS_FSYNC",
        "os.geteuid": "_BOUND_OS_GETEUID",
        "os.listdir": "_BOUND_OS_LISTDIR",
        "os.mkdir": "_BOUND_OS_MKDIR",
        "os.open": "_BOUND_OS_OPEN",
        "os.path.abspath": "_BOUND_OS_PATH_ABSPATH",
        "os.read": "_BOUND_OS_READ",
        "os.replace": "_BOUND_OS_REPLACE",
        "os.stat": "_BOUND_OS_STAT",
        "os.strerror": "_BOUND_OS_STRERROR",
        "os.write": "_BOUND_OS_WRITE",
        "fcntl.flock": "_BOUND_FCNTL_FLOCK",
        "ctypes.get_errno": "_BOUND_CTYPES_GET_ERRNO",
        "stat.S_IMODE": "_BOUND_STAT_S_IMODE",
        "stat.S_ISDIR": "_BOUND_STAT_S_ISDIR",
        "stat.S_ISLNK": "_BOUND_STAT_S_ISLNK",
        "stat.S_ISREG": "_BOUND_STAT_S_ISREG",
    }
)

_FILESYSTEM_CONSTANT_SPECS = {
    "os.O_CREAT": (os, "O_CREAT"),
    "os.O_DIRECTORY": (os, "O_DIRECTORY"),
    "os.O_EXCL": (os, "O_EXCL"),
    "os.O_NOFOLLOW": (os, "O_NOFOLLOW"),
    "os.O_RDONLY": (os, "O_RDONLY"),
    "os.O_WRONLY": (os, "O_WRONLY"),
    "os.sep": (os, "sep"),
    "fcntl.LOCK_EX": (fcntl, "LOCK_EX"),
    "fcntl.LOCK_NB": (fcntl, "LOCK_NB"),
    "errno.EEXIST": (errno, "EEXIST"),
    "errno.ENOTEMPTY": (errno, "ENOTEMPTY"),
}
try:
    _BOUND_FILESYSTEM_CONSTANTS = MappingProxyType(
        {
            label: (owner, attribute, getattr(owner, attribute))
            for label, (owner, attribute) in _FILESYSTEM_CONSTANT_SPECS.items()
        }
    )
except AttributeError as error:
    raise ContractError("required filesystem constant is unavailable") from error

_BOUND_OS_O_CREAT = os.O_CREAT
_BOUND_OS_O_DIRECTORY = os.O_DIRECTORY
_BOUND_OS_O_EXCL = os.O_EXCL
_BOUND_OS_O_NOFOLLOW = os.O_NOFOLLOW
_BOUND_OS_O_RDONLY = os.O_RDONLY
_BOUND_OS_O_WRONLY = os.O_WRONLY
_BOUND_OS_SEP = os.sep
_BOUND_FCNTL_LOCK_EX = fcntl.LOCK_EX
_BOUND_FCNTL_LOCK_NB = fcntl.LOCK_NB
_BOUND_ERRNO_EEXIST = errno.EEXIST
_BOUND_ERRNO_ENOTEMPTY = errno.ENOTEMPTY
_BOUND_FILESYSTEM_CONSTANT_GLOBAL_NAMES = MappingProxyType(
    {
        "os.O_CREAT": "_BOUND_OS_O_CREAT",
        "os.O_DIRECTORY": "_BOUND_OS_O_DIRECTORY",
        "os.O_EXCL": "_BOUND_OS_O_EXCL",
        "os.O_NOFOLLOW": "_BOUND_OS_O_NOFOLLOW",
        "os.O_RDONLY": "_BOUND_OS_O_RDONLY",
        "os.O_WRONLY": "_BOUND_OS_O_WRONLY",
        "os.sep": "_BOUND_OS_SEP",
        "fcntl.LOCK_EX": "_BOUND_FCNTL_LOCK_EX",
        "fcntl.LOCK_NB": "_BOUND_FCNTL_LOCK_NB",
        "errno.EEXIST": "_BOUND_ERRNO_EEXIST",
        "errno.ENOTEMPTY": "_BOUND_ERRNO_ENOTEMPTY",
    }
)


def _load_bound_atomic_rename() -> tuple[Any, Any, str, int, Any]:
    libc = _BOUND_CTYPES_PYDLL(None, use_errno=True)
    if _BOUND_SYS_PLATFORM == "darwin":
        symbol = "renameatx_np"
        flag = 0x00000004
    elif _BOUND_SYS_PLATFORM.startswith("linux"):
        symbol = "renameat2"
        flag = 1
    else:
        raise ContractError("platform lacks a reviewed atomic no-overwrite rename")
    rename = getattr(libc, symbol, None)
    if rename is None:
        raise ContractError(symbol + " is unavailable for atomic no-overwrite")
    unlinkat = getattr(libc, "unlinkat", None)
    if unlinkat is None:
        raise ContractError("unlinkat is unavailable for descriptor-relative cleanup")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    unlinkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    unlinkat.restype = ctypes.c_int
    if rename.errcheck is not None or unlinkat.errcheck is not None:
        raise ContractError("native publication callables must start without errcheck")
    return libc, rename, symbol, flag, unlinkat


(
    _BOUND_LIBC,
    _BOUND_ATOMIC_RENAME,
    _BOUND_ATOMIC_RENAME_SYMBOL,
    _BOUND_ATOMIC_RENAME_FLAG,
    _BOUND_UNLINKAT,
) = _load_bound_atomic_rename()
_BOUND_ATOMIC_RENAME_TYPE = type(_BOUND_ATOMIC_RENAME)
_BOUND_ATOMIC_RENAME_ARGTYPES = tuple(_BOUND_ATOMIC_RENAME.argtypes)
_BOUND_ATOMIC_RENAME_RESTYPE = _BOUND_ATOMIC_RENAME.restype
_BOUND_ATOMIC_RENAME_ERRCHECK = _BOUND_ATOMIC_RENAME.errcheck
_BOUND_UNLINKAT_TYPE = type(_BOUND_UNLINKAT)
_BOUND_UNLINKAT_ARGTYPES = tuple(_BOUND_UNLINKAT.argtypes)
_BOUND_UNLINKAT_RESTYPE = _BOUND_UNLINKAT.restype
_BOUND_UNLINKAT_ERRCHECK = _BOUND_UNLINKAT.errcheck
_BOUND_AT_REMOVEDIR = 0x80
_BOUND_ATOMIC_RENAME_BINDING = (
    _BOUND_LIBC,
    _BOUND_ATOMIC_RENAME,
    _BOUND_ATOMIC_RENAME_SYMBOL,
    _BOUND_ATOMIC_RENAME_FLAG,
    _BOUND_ATOMIC_RENAME_TYPE,
    _BOUND_ATOMIC_RENAME_ARGTYPES,
    _BOUND_ATOMIC_RENAME_RESTYPE,
    _BOUND_ATOMIC_RENAME_ERRCHECK,
)
_BOUND_UNLINKAT_BINDING = (
    _BOUND_UNLINKAT,
    _BOUND_UNLINKAT_TYPE,
    _BOUND_UNLINKAT_ARGTYPES,
    _BOUND_UNLINKAT_RESTYPE,
    _BOUND_UNLINKAT_ERRCHECK,
    _BOUND_AT_REMOVEDIR,
)

_BOUND_RUNTIME_MODULES = MappingProxyType(
    {
        "_ctypes": (_ctypes_native, "_ctypes_native"),
        "_hashlib": (_hashlib_native, "_hashlib_native"),
        "_json": (_json_native, "_json_native"),
        "_random": (_random_native, "_random_native"),
        "_sre": (_sre_native, "_sre_native"),
        "_stat": (_stat_native, "_stat_native"),
        "_struct": (_struct_native, "_struct_native"),
        "argparse": (argparse, "argparse"),
        "builtins": (builtins_module, "builtins_module"),
        "collections": (collections_module, "collections_module"),
        "ctypes": (ctypes, "ctypes"),
        "dataclasses": (dataclasses_module, "dataclasses_module"),
        "errno": (errno, "errno"),
        "fcntl": (fcntl, "fcntl"),
        "hashlib": (hashlib, "hashlib"),
        "importlib": (importlib, "importlib"),
        "importlib.util": (
            _BOUND_IMPORTLIB_UTIL_MODULE,
            "_BOUND_IMPORTLIB_UTIL_MODULE",
        ),
        "importlib.metadata": (importlib_metadata, "importlib_metadata"),
        "importlib.metadata._adapters": (
            _BOUND_IMPORTLIB_METADATA_ADAPTERS_MODULE,
            "_BOUND_IMPORTLIB_METADATA_ADAPTERS_MODULE",
        ),
        "importlib.metadata._collections": (
            _BOUND_IMPORTLIB_METADATA_COLLECTIONS_MODULE,
            "_BOUND_IMPORTLIB_METADATA_COLLECTIONS_MODULE",
        ),
        "importlib.metadata._functools": (
            _BOUND_IMPORTLIB_METADATA_FUNCTOOLS_MODULE,
            "_BOUND_IMPORTLIB_METADATA_FUNCTOOLS_MODULE",
        ),
        "importlib.metadata._itertools": (
            _BOUND_IMPORTLIB_METADATA_ITERTOOLS_MODULE,
            "_BOUND_IMPORTLIB_METADATA_ITERTOOLS_MODULE",
        ),
        "importlib.metadata._meta": (
            _BOUND_IMPORTLIB_METADATA_META_MODULE,
            "_BOUND_IMPORTLIB_METADATA_META_MODULE",
        ),
        "importlib.metadata._text": (
            _BOUND_IMPORTLIB_METADATA_TEXT_MODULE,
            "_BOUND_IMPORTLIB_METADATA_TEXT_MODULE",
        ),
        "json": (json, "json"),
        "json.decoder": (_BOUND_JSON_DECODER_MODULE, "_BOUND_JSON_DECODER_MODULE"),
        "json.encoder": (_BOUND_JSON_ENCODER_MODULE, "_BOUND_JSON_ENCODER_MODULE"),
        "json.scanner": (_BOUND_JSON_SCANNER_MODULE, "_BOUND_JSON_SCANNER_MODULE"),
        "marshal": (marshal, "marshal"),
        "math": (math, "math"),
        "os": (os, "os"),
        "pathlib": (pathlib_module, "pathlib_module"),
        "posixpath": (_BOUND_OS_PATH_MODULE, "_BOUND_OS_PATH_MODULE"),
        "platform": (platform, "platform"),
        "posix": (_posix_native, "_posix_native"),
        "random": (random, "random"),
        "re": (re, "re"),
        "stat": (stat, "stat"),
        "struct": (struct, "struct"),
        "sys": (sys, "sys"),
        "sysconfig": (sysconfig, "sysconfig"),
        "typing": (typing_module, "typing_module"),
        "types": (types_module, "types_module"),
    }
)


def _bootstrap_source_bytes(
    path: Path, module_name: str
) -> tuple[bytes, dict[str, Any]]:
    flags = _BOUND_OS_O_RDONLY | _BOUND_OS_O_NOFOLLOW
    try:
        descriptor = _BOUND_OS_OPEN(path, flags)
    except OSError as error:
        raise ContractError(
            "reviewed module source cannot be opened safely: {}".format(module_name)
        ) from error
    try:
        before = _BOUND_OS_FSTAT(descriptor)
        if not _BOUND_STAT_S_ISREG(before.st_mode):
            raise ContractError(
                "reviewed module source is not regular: {}".format(module_name)
            )
        blocks = []
        while True:
            block = _BOUND_OS_READ(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = _BOUND_OS_FSTAT(descriptor)
    finally:
        _BOUND_OS_CLOSE(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ContractError(
            "reviewed module changed during import read: {}".format(module_name)
        )
    return payload, {
        "bytes": len(payload),
        "sha256": _BOUND_HASHLIB_SHA256(payload).hexdigest(),
    }


_GENERATOR_PATH = Path(__file__).resolve(strict=True)
_, _generator_import_receipt = _bootstrap_source_bytes(
    _GENERATOR_PATH, "pipeline.generate_dws_single_completion_v1"
)


def _import_reviewed_module(
    module_name: str, expected_path: Path
) -> tuple[Any, dict[str, Any]]:
    if module_name in _BOUND_SYS_MODULES:
        raise ContractError(
            "reviewed module was preloaded before byte binding: {}".format(module_name)
        )
    expected_path = Path(expected_path)
    try:
        expected_metadata = expected_path.lstat()
        expected_resolved = expected_path.resolve(strict=True)
    except OSError as error:
        raise ContractError(
            "reviewed module source is unavailable: {}".format(module_name)
        ) from error
    if expected_path.is_symlink() or not _BOUND_STAT_S_ISREG(expected_metadata.st_mode):
        raise ContractError(
            "reviewed module source must be a regular non-symlink: {}".format(
                module_name
            )
        )
    payload, import_receipt = _bootstrap_source_bytes(expected_path, module_name)
    spec = _BOUND_IMPORTLIB_SPEC_FROM_FILE_LOCATION(module_name, expected_resolved)
    if spec is None:
        raise ContractError(
            "reviewed module has no import specification: " + module_name
        )
    module = _BOUND_IMPORTLIB_MODULE_FROM_SPEC(spec)
    _BOUND_SYS_MODULES[module_name] = module
    import_path_snapshot = list(_BOUND_SYS_PATH)
    try:
        code = compile(payload, str(expected_resolved), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        _BOUND_SYS_MODULES.pop(module_name, None)
        raise
    finally:
        _BOUND_SYS_PATH[:] = import_path_snapshot
    _, post_import_receipt = _bootstrap_source_bytes(expected_path, module_name)
    if post_import_receipt != import_receipt:
        _BOUND_SYS_MODULES.pop(module_name, None)
        raise ContractError(
            "reviewed module bytes changed during import: " + module_name
        )
    module_file = getattr(module, "__file__", None)
    module_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not module_file or not module_origin:
        raise ContractError("reviewed module has no source origin: " + module_name)
    try:
        resolved_file = Path(module_file).resolve(strict=True)
        resolved_origin = Path(module_origin).resolve(strict=True)
    except OSError as error:
        raise ContractError(
            "reviewed module source cannot be resolved: {}".format(module_name)
        ) from error
    if resolved_file != expected_resolved or resolved_origin != expected_resolved:
        raise ContractError("reviewed module path mismatch: " + module_name)
    return module, import_receipt


_DIGITWISE_PROTOCOL_PATH = ROOT / "train/digitwise_protocol.py"
_ROW_BUILDER_PATH = ROOT / "pipeline/generate_digitwise_recurrent_v1.py"
_digitwise_protocol_module, _digitwise_protocol_import_receipt = (
    _import_reviewed_module("digitwise_protocol", _DIGITWISE_PROTOCOL_PATH)
)
_row_builder_module, _row_builder_import_receipt = _import_reviewed_module(
    "pipeline.generate_digitwise_recurrent_v1", _ROW_BUILDER_PATH
)
_IMPORTED_REVIEWED_MODULE_RECEIPTS = {
    "digitwise_protocol": _digitwise_protocol_import_receipt,
    "pipeline.generate_digitwise_recurrent_v1": _row_builder_import_receipt,
}
apply_microstep = _digitwise_protocol_module.apply_microstep
canonical_state = _digitwise_protocol_module.canonical_state
initial_state = _digitwise_protocol_module.initial_state
parse_answer = _digitwise_protocol_module.parse_answer
parse_state = _digitwise_protocol_module.parse_state
state_answer = _digitwise_protocol_module.state_answer
rows_from_episode = _row_builder_module.rows_from_episode
_PYTHON_FUNCTION_TYPE = type(_assert_isolated_startup)
_REVIEWED_FUNCTION_NAMES = {
    "digitwise_protocol": (
        "_digits_lsf",
        "_validate_state",
        "_value_lsf",
        "apply_microstep",
        "canonical_state",
        "digit_prompt",
        "final_prompt",
        "initial_state",
        "microstep_prompt",
        "parse_answer",
        "parse_digit",
        "parse_state",
        "state_answer",
        "state_digit",
    ),
    "pipeline.generate_digitwise_recurrent_v1": (
        "_episode_prompts",
        "counterfactual_episode",
        "deduplicate_rows",
        "episode_from_operands",
        "episode_prompts",
        "episode_signature",
        "main",
        "make_episode",
        "make_operands",
        "normalized",
        "rows_from_episode",
        "sha256_file",
        "write_jsonl",
    ),
}
_REVIEWED_MODULE_OBJECTS = {
    "digitwise_protocol": _digitwise_protocol_module,
    "pipeline.generate_digitwise_recurrent_v1": _row_builder_module,
}
_BOUND_REVIEWED_FUNCTIONS = {}
for _reviewed_module_name, _reviewed_names in _REVIEWED_FUNCTION_NAMES.items():
    _reviewed_module = _REVIEWED_MODULE_OBJECTS[_reviewed_module_name]
    for _reviewed_name in _reviewed_names:
        _reviewed_value = getattr(_reviewed_module, _reviewed_name, None)
        if type(_reviewed_value) is not _PYTHON_FUNCTION_TYPE:
            raise ContractError(
                "reviewed Python function is unavailable: "
                + _reviewed_module_name
                + "."
                + _reviewed_name
            )
        _reviewed_label = _reviewed_module_name + "." + _reviewed_name
        _BOUND_REVIEWED_FUNCTIONS[_reviewed_label] = (
            _reviewed_module,
            _reviewed_name,
            _reviewed_value,
            _capture_callable_implementation(_reviewed_value),
        )
_BOUND_REVIEWED_FUNCTIONS = MappingProxyType(_BOUND_REVIEWED_FUNCTIONS)


class _FrozenReviewedGlobalsMeta(type):
    def __setattr__(cls, name: str, value: Any) -> None:
        del name, value
        raise ContractError("frozen reviewed globals type mutation rejected")

    def __delattr__(cls, name: str) -> None:
        del name
        raise ContractError("frozen reviewed globals type mutation rejected")


class _FrozenReviewedGlobals(dict, metaclass=_FrozenReviewedGlobalsMeta):
    __slots__ = ("_label", "_sealed")

    def __init__(self, label: str, values: dict[str, Any]) -> None:
        dict.__init__(self, values)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_sealed", False)

    def _require_bootstrap(self) -> None:
        if self._sealed:
            raise ContractError(
                "frozen reviewed globals mutation rejected: " + self._label
            )

    def bootstrap_update(self, values: dict[str, Any]) -> None:
        self._require_bootstrap()
        dict.update(self, values)

    def seal(self) -> None:
        self._require_bootstrap()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise ContractError("frozen reviewed globals state mutation rejected")

    def __delattr__(self, name: str) -> None:
        del name
        raise ContractError("frozen reviewed globals state mutation rejected")

    def __setitem__(self, key: str, value: Any) -> None:
        del key, value
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def __delitem__(self, key: str) -> None:
        del key
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def clear(self) -> None:
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def pop(self, key: str, default: Any = None) -> Any:
        del key, default
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def popitem(self) -> tuple[Any, Any]:
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def setdefault(self, key: str, default: Any = None) -> Any:
        del key, default
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)

    def __ior__(self, other: Any) -> Any:
        del other
        raise ContractError("frozen reviewed globals mutation rejected: " + self._label)


def _clone_reviewed_module_functions(
    module_name: str,
    *,
    global_overrides: dict[str, Any] | None = None,
    builtins_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = _REVIEWED_MODULE_OBJECTS[module_name]
    frozen_globals = _FrozenReviewedGlobals(
        module_name + " globals", dict(module.__dict__)
    )
    if builtins_override is None:
        frozen_builtins = _FrozenReviewedGlobals(
            module_name + " builtins", dict(builtins_module.__dict__)
        )
        frozen_builtins.seal()
    else:
        frozen_builtins = builtins_override
    frozen_globals.bootstrap_update({"__builtins__": frozen_builtins})
    if global_overrides is not None:
        frozen_globals.bootstrap_update(global_overrides)
    clones = {}
    for name in _REVIEWED_FUNCTION_NAMES[module_name]:
        original = _BOUND_REVIEWED_FUNCTIONS[module_name + "." + name][2]
        if original.__closure__ is not None:
            raise ContractError(
                "reviewed module function unexpectedly closes over live state: "
                + module_name
                + "."
                + name
            )
        clone = _PYTHON_FUNCTION_TYPE(
            original.__code__,
            frozen_globals,
            original.__name__,
            original.__defaults__,
            None,
        )
        clone.__kwdefaults__ = (
            None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
        )
        clone.__qualname__ = original.__qualname__
        clones[name] = clone
    frozen_globals.bootstrap_update(clones)
    frozen_globals.seal()
    return frozen_globals, MappingProxyType(clones)


_FROZEN_PROTOCOL_GLOBALS, _FROZEN_PROTOCOL_FUNCTIONS = _clone_reviewed_module_functions(
    "digitwise_protocol"
)
_FROZEN_PROTOCOL_IMPORT_TYPE = _BOUND_COLLECTIONS_NAMEDTUPLE(
    "_FrozenDigitwiseProtocolImport", ("parse_state",)
)
_FROZEN_PROTOCOL_IMPORT = _FROZEN_PROTOCOL_IMPORT_TYPE(
    _FROZEN_PROTOCOL_FUNCTIONS["parse_state"]
)


def _frozen_reviewed_import(
    name: str,
    globals_value: dict[str, Any] | None = None,
    locals_value: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
    _protocol: Any = _FROZEN_PROTOCOL_IMPORT,
    _bound_import: Any = _BOUND_BUILTIN_IMPORT,
) -> Any:
    if name == "digitwise_protocol" and level == 0:
        return _protocol
    return _bound_import(name, globals_value, locals_value, fromlist, level)


_FROZEN_ROW_BUILTINS = _FrozenReviewedGlobals(
    "pipeline.generate_digitwise_recurrent_v1 builtins",
    dict(builtins_module.__dict__),
)
_FROZEN_ROW_BUILTINS.bootstrap_update({"__import__": _frozen_reviewed_import})
_FROZEN_ROW_BUILTINS.seal()
_FROZEN_ROW_PROTOCOL_ALIASES = {
    name: _FROZEN_PROTOCOL_FUNCTIONS[name]
    for name in (
        "apply_microstep",
        "canonical_state",
        "digit_prompt",
        "final_prompt",
        "initial_state",
        "microstep_prompt",
        "state_answer",
        "state_digit",
    )
}
_FROZEN_ROW_GLOBALS, _FROZEN_ROW_FUNCTIONS = _clone_reviewed_module_functions(
    "pipeline.generate_digitwise_recurrent_v1",
    global_overrides=_FROZEN_ROW_PROTOCOL_ALIASES,
    builtins_override=_FROZEN_ROW_BUILTINS,
)

apply_microstep = _FROZEN_PROTOCOL_FUNCTIONS["apply_microstep"]
canonical_state = _FROZEN_PROTOCOL_FUNCTIONS["canonical_state"]
initial_state = _FROZEN_PROTOCOL_FUNCTIONS["initial_state"]
parse_answer = _FROZEN_PROTOCOL_FUNCTIONS["parse_answer"]
parse_state = _FROZEN_PROTOCOL_FUNCTIONS["parse_state"]
state_answer = _FROZEN_PROTOCOL_FUNCTIONS["state_answer"]
rows_from_episode = _FROZEN_ROW_FUNCTIONS["rows_from_episode"]
_BOUND_CONSUMED_REVIEWED_FUNCTIONS = MappingProxyType(
    {
        **{
            "digitwise_protocol." + name: value
            for name, value in _FROZEN_PROTOCOL_FUNCTIONS.items()
        },
        **{
            "pipeline.generate_digitwise_recurrent_v1." + name: value
            for name, value in _FROZEN_ROW_FUNCTIONS.items()
        },
    }
)
_BOUND_CONSUMED_REVIEWED_IMPLEMENTATIONS = MappingProxyType(
    {
        label: _capture_callable_implementation(value)
        for label, value in _BOUND_CONSUMED_REVIEWED_FUNCTIONS.items()
    }
)
_BOUND_FROZEN_GLOBAL_MAPPINGS = (
    (
        "digitwise_protocol.globals",
        _FROZEN_PROTOCOL_GLOBALS,
        tuple(_FROZEN_PROTOCOL_GLOBALS.items()),
    ),
    (
        "digitwise_protocol.builtins",
        _FROZEN_PROTOCOL_GLOBALS["__builtins__"],
        tuple(_FROZEN_PROTOCOL_GLOBALS["__builtins__"].items()),
    ),
    (
        "pipeline.generate_digitwise_recurrent_v1.globals",
        _FROZEN_ROW_GLOBALS,
        tuple(_FROZEN_ROW_GLOBALS.items()),
    ),
    (
        "pipeline.generate_digitwise_recurrent_v1.builtins",
        _FROZEN_ROW_BUILTINS,
        tuple(_FROZEN_ROW_BUILTINS.items()),
    ),
)

_BOUND_REVIEWED_GENERATOR_ALIASES = MappingProxyType(
    {
        "apply_microstep": apply_microstep,
        "canonical_state": canonical_state,
        "initial_state": initial_state,
        "parse_answer": parse_answer,
        "parse_state": parse_state,
        "rows_from_episode": rows_from_episode,
        "state_answer": state_answer,
    }
)
_BOUND_REVIEWED_CROSS_MODULE_ALIASES = MappingProxyType(
    {
        name: getattr(_digitwise_protocol_module, name)
        for name in (
            "apply_microstep",
            "canonical_state",
            "digit_prompt",
            "final_prompt",
            "initial_state",
            "microstep_prompt",
            "state_answer",
            "state_digit",
        )
    }
)


def _install_callable_mutation_guard(
    label: str, protected_callables: Iterable[Any]
) -> dict[str, Any]:
    protected = {
        value for value in protected_callables if type(value) is _PYTHON_FUNCTION_TYPE
    }
    protected_attributes = frozenset(("__code__", "__defaults__", "__kwdefaults__"))

    def reject_mutation(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "object.__setattr__" or len(arguments) < 2:
            return
        target, attribute = arguments[:2]
        if (
            type(target) is _PYTHON_FUNCTION_TYPE
            and target in protected
            and attribute in protected_attributes
        ):
            raise ContractError(
                "protected callable implementation mutation rejected: " + label
            )

    protected.add(reject_mutation)
    _BOUND_SYS_ADDAUDITHOOK(reject_mutation)
    probe = next(iter(protected))
    try:
        probe.__code__ = probe.__code__
    except ContractError as error:
        if "protected callable implementation mutation rejected" not in str(error):
            raise
    else:
        raise ContractError("callable mutation audit guard was not installed")
    return {
        "label": label,
        "protected_python_callables": len(protected),
        "blocked_attributes": sorted(protected_attributes),
        "audit_event": "object.__setattr__",
        "append_only_runtime_hook": True,
        "self_tested": True,
    }


_RUNTIME_EXPORT_MUTATION_GUARD = _install_callable_mutation_guard(
    "runtime-exports-v1",
    (
        *(binding[2] for binding in _BOUND_RUNTIME_EXPORTS.values()),
        *(binding[2] for binding in _BOUND_SERIALIZATION_CLASS_METHODS.values()),
    ),
)


_REVIEWED_FILESYSTEM_MUTATION_GUARD = _install_callable_mutation_guard(
    "reviewed-filesystem-v1",
    (
        *(binding[2] for binding in _BOUND_REVIEWED_FUNCTIONS.values()),
        *_BOUND_CONSUMED_REVIEWED_FUNCTIONS.values(),
        *(binding[2] for binding in _BOUND_FILESYSTEM_EXPORTS.values()),
        *(
            value
            for value in _FrozenReviewedGlobals.__dict__.values()
            if type(value) is _PYTHON_FUNCTION_TYPE
        ),
        *(
            value
            for value in _FrozenReviewedGlobalsMeta.__dict__.values()
            if type(value) is _PYTHON_FUNCTION_TYPE
        ),
        _frozen_reviewed_import,
        _install_callable_mutation_guard,
    ),
)


def _load_bound_tokenizers_runtime() -> tuple[Any, Any, Any, Any, Any, str]:
    preloaded = [
        module_name
        for module_name in ("tokenizers", "tokenizers.tokenizers")
        if module_name in _BOUND_SYS_MODULES
    ]
    if preloaded:
        raise ContractError(
            "tokenizers runtime was preloaded before semantic binding: "
            + ", ".join(preloaded)
        )
    try:
        package_module = _BOUND_IMPORTLIB_IMPORT_MODULE("tokenizers")
        native_module = _BOUND_IMPORTLIB_IMPORT_MODULE("tokenizers.tokenizers")
        distribution_version = _BOUND_IMPORTLIB_METADATA_VERSION("tokenizers")
    except (ImportError, _BOUND_IMPORTLIB_METADATA_PACKAGE_NOT_FOUND_ERROR) as error:
        raise ContractError("the tokenizers package is required") from error
    tokenizer_class = getattr(native_module, "Tokenizer", None)
    encoding_class = getattr(native_module, "Encoding", None)
    if (
        tokenizer_class is None
        or getattr(package_module, "Tokenizer", None) is not tokenizer_class
        or encoding_class is None
        or getattr(package_module, "Encoding", None) is not encoding_class
        or tokenizer_class.__module__ != "tokenizers"
        or encoding_class.__module__ != "tokenizers"
        or getattr(package_module, "__version__", None) != distribution_version
    ):
        raise ContractError("tokenizers native class/version identity mismatch")
    from_str_descriptor = tokenizer_class.__dict__.get("from_str")
    from_str_callable = getattr(tokenizer_class, "from_str", None)
    if from_str_descriptor is None or from_str_callable is None:
        raise ContractError("tokenizers native class lacks from_str")
    tokenizer_methods = {
        name: tokenizer_class.__dict__.get(name)
        for name in ("get_vocab_size", "token_to_id", "encode", "decode")
    }
    encoding_ids_descriptor = encoding_class.__dict__.get("ids")
    if any(value is None for value in tokenizer_methods.values()) or (
        encoding_ids_descriptor is None
    ):
        raise ContractError("tokenizers native class lacks a consumed descriptor")
    return (
        package_module,
        native_module,
        tokenizer_class,
        (from_str_descriptor, from_str_callable),
        (encoding_class, encoding_ids_descriptor, tokenizer_methods),
        distribution_version,
    )


(
    _BOUND_TOKENIZERS_MODULE,
    _BOUND_TOKENIZERS_NATIVE_MODULE,
    _BOUND_TOKENIZER_CLASS,
    _bound_tokenizer_from_str,
    _bound_tokenizer_descriptors,
    _BOUND_TOKENIZERS_VERSION,
) = _load_bound_tokenizers_runtime()
(
    _BOUND_TOKENIZER_FROM_STR_DESCRIPTOR,
    _BOUND_TOKENIZER_FROM_STR,
) = _bound_tokenizer_from_str
(
    _BOUND_TOKENIZER_ENCODING_CLASS,
    _BOUND_TOKENIZER_ENCODING_IDS_DESCRIPTOR,
    _bound_tokenizer_methods,
) = _bound_tokenizer_descriptors
_BOUND_TOKENIZER_METHODS = MappingProxyType(dict(_bound_tokenizer_methods))
del _bound_tokenizer_methods
del _bound_tokenizer_descriptors
del _bound_tokenizer_from_str

_BOUND_RANDOM_MODULE = random
_BOUND_RANDOM_NATIVE_MODULE = _random_native
_BOUND_RANDOM_CLASS = random.Random
_BOUND_RANDOM_NATIVE_CLASS = _random_native.Random
_BOUND_RANDOM_BASES = _BOUND_RANDOM_CLASS.__bases__
_RANDOM_METHOD_NAMES = (
    "__new__",
    "__init__",
    "seed",
    "getstate",
    "getrandbits",
    "randrange",
    "randint",
    "choice",
    "shuffle",
    "random",
    "_randbelow",
)


def _defining_class(class_value: type[Any], attribute: str) -> type[Any]:
    for owner in class_value.__mro__:
        if attribute in owner.__dict__:
            return owner
    raise ContractError("bound class lacks required attribute: " + attribute)


_BOUND_RANDOM_METHODS = MappingProxyType(
    {
        name: (
            _defining_class(_BOUND_RANDOM_CLASS, name),
            _defining_class(_BOUND_RANDOM_CLASS, name).__dict__[name],
        )
        for name in _RANDOM_METHOD_NAMES
    }
)
_BOUND_RANDOM_METHOD_IMPLEMENTATIONS = MappingProxyType(
    {
        name: _capture_callable_implementation(descriptor)
        for name, (_, descriptor) in _BOUND_RANDOM_METHODS.items()
    }
)
_RANDOM_MUTATION_GUARD = _install_callable_mutation_guard(
    "python-random-v1",
    (descriptor for _, descriptor in _BOUND_RANDOM_METHODS.values()),
)


def _assert_bound_tokenizer_exports() -> None:
    if (
        _BOUND_SYS_MODULES.get("tokenizers") is not _BOUND_TOKENIZERS_MODULE
        or _BOUND_SYS_MODULES.get("tokenizers.tokenizers")
        is not _BOUND_TOKENIZERS_NATIVE_MODULE
        or getattr(_BOUND_TOKENIZERS_MODULE, "Tokenizer", None)
        is not _BOUND_TOKENIZER_CLASS
        or getattr(_BOUND_TOKENIZERS_NATIVE_MODULE, "Tokenizer", None)
        is not _BOUND_TOKENIZER_CLASS
        or getattr(_BOUND_TOKENIZERS_MODULE, "Encoding", None)
        is not _BOUND_TOKENIZER_ENCODING_CLASS
        or getattr(_BOUND_TOKENIZERS_NATIVE_MODULE, "Encoding", None)
        is not _BOUND_TOKENIZER_ENCODING_CLASS
        or _BOUND_TOKENIZER_CLASS.__dict__.get("from_str")
        is not _BOUND_TOKENIZER_FROM_STR_DESCRIPTOR
        or getattr(_BOUND_TOKENIZER_CLASS, "from_str", None)
        is not _BOUND_TOKENIZER_FROM_STR
        or getattr(_BOUND_TOKENIZERS_MODULE, "__version__", None)
        != _BOUND_TOKENIZERS_VERSION
        or _BOUND_TOKENIZER_ENCODING_CLASS.__dict__.get("ids")
        is not _BOUND_TOKENIZER_ENCODING_IDS_DESCRIPTOR
    ):
        raise ContractError("bound tokenizers runtime exports changed")
    for name, descriptor in _BOUND_TOKENIZER_METHODS.items():
        if _BOUND_TOKENIZER_CLASS.__dict__.get(name) is not descriptor:
            raise ContractError("bound tokenizer method changed: " + name)


def _assert_bound_random_exports() -> None:
    if (
        _BOUND_SYS_MODULES.get("random") is not _BOUND_RANDOM_MODULE
        or _BOUND_SYS_MODULES.get("_random") is not _BOUND_RANDOM_NATIVE_MODULE
        or random.Random is not _BOUND_RANDOM_CLASS
        or _random_native.Random is not _BOUND_RANDOM_NATIVE_CLASS
        or _BOUND_RANDOM_CLASS.__bases__ != _BOUND_RANDOM_BASES
        or _BOUND_RANDOM_BASES != (_BOUND_RANDOM_NATIVE_CLASS,)
    ):
        raise ContractError("bound Python random runtime exports changed")
    for name, (owner, descriptor) in _BOUND_RANDOM_METHODS.items():
        if owner.__dict__.get(name) is not descriptor:
            raise ContractError("bound Python random method changed: " + name)
        _assert_callable_implementation(
            "random.Random." + name,
            descriptor,
            _BOUND_RANDOM_METHOD_IMPLEMENTATIONS[name],
        )


def _assert_bound_runtime_exports() -> None:
    for label, (
        owner,
        attribute,
        bound_value,
        implementation,
    ) in _BOUND_RUNTIME_EXPORTS.items():
        live_value = getattr(owner, attribute, None)
        if live_value is not bound_value:
            raise ContractError("bound runtime export changed: " + label)
        if implementation is not None:
            _assert_callable_implementation(label, live_value, implementation)
    for label, global_name in _BOUND_RUNTIME_CONSUMED_EXPORT_GLOBALS.items():
        if globals().get(global_name) is not _BOUND_RUNTIME_EXPORTS[label][2]:
            raise ContractError("captured runtime export changed: " + label)
    for label, (
        owner,
        attribute,
        bound_descriptor,
        implementation,
    ) in _BOUND_SERIALIZATION_CLASS_METHODS.items():
        live_descriptor = owner.__dict__.get(attribute)
        if live_descriptor is not bound_descriptor:
            raise ContractError("bound serialization method changed: " + label)
        _assert_callable_implementation(label, live_descriptor, implementation)


def _assert_bound_generator_aliases() -> None:
    for label, (global_name, bound_value) in _BOUND_GENERATOR_CALLABLE_ALIASES.items():
        if globals().get(global_name) is not bound_value:
            raise ContractError("captured generator callable alias changed: " + label)


def _assert_sealed_runtime_classes() -> None:
    for class_value in (FrozenTokenizer, _PinnedDirectory):
        if (
            type(class_value) is not _SealedRuntimeClassMeta
            or class_value.__dict__.get("_runtime_descriptors_sealed") is not True
        ):
            raise ContractError(
                "generator runtime class descriptor boundary changed: "
                + class_value.__qualname__
            )


def _assert_bound_runtime_modules() -> None:
    if sys.modules is not _BOUND_SYS_MODULES or sys.path is not _BOUND_SYS_PATH:
        raise ContractError("bound sys module/path registries changed")
    for module_name, (module, global_name) in _BOUND_RUNTIME_MODULES.items():
        if (
            _BOUND_SYS_MODULES.get(module_name) is not module
            or globals().get(global_name) is not module
        ):
            raise ContractError("bound runtime module changed: " + module_name)


def _assert_frozen_generator_builtins() -> None:
    frozen = globals().get("_FROZEN_GENERATOR_BUILTINS")
    expected_items = globals().get("_BOUND_GENERATOR_BUILTIN_ITEMS")
    if (
        type(frozen) is not _FrozenReviewedGlobals
        or frozen._sealed is not True
        or expected_items is None
        or globals().get("__builtins__") is not frozen
        or tuple(frozen.items()) != expected_items
        or tuple(builtins_module.__dict__.items()) != expected_items
    ):
        raise ContractError("frozen generator builtins changed")


def _generator_builtins_receipt() -> dict[str, Any]:
    _assert_frozen_generator_builtins()
    return {
        "mapping_type_module": type(_FROZEN_GENERATOR_BUILTINS).__module__,
        "mapping_type_qualname": type(_FROZEN_GENERATOR_BUILTINS).__qualname__,
        "sealed": True,
        "exact_key_and_value_identity_required": True,
        "ordinary_mutation_methods_rejected": True,
        "entry_count": len(_BOUND_GENERATOR_BUILTIN_ITEMS),
        "entries": {
            name: {
                "type_module": type(value).__module__,
                "type_qualname": type(value).__qualname__,
                "callable": callable(value),
            }
            for name, value in _BOUND_GENERATOR_BUILTIN_ITEMS
        },
    }


def _assert_bound_struct_exports() -> None:
    if (
        _BOUND_SYS_MODULES.get("struct") is not _BOUND_STRUCT_MODULE
        or _BOUND_SYS_MODULES.get("_struct") is not _BOUND_STRUCT_NATIVE_MODULE
        or struct.pack is not _BOUND_STRUCT_PACK
        or _struct_native.pack is not _BOUND_STRUCT_PACK
        or struct.unpack is not _BOUND_STRUCT_UNPACK
        or _struct_native.unpack is not _BOUND_STRUCT_UNPACK
    ):
        raise ContractError("bound struct runtime exports changed")


def _assert_bound_hashlib_exports() -> None:
    if (
        _BOUND_SYS_MODULES.get("hashlib") is not hashlib
        or _BOUND_SYS_MODULES.get("_hashlib") is not _hashlib_native
        or hashlib.sha256 is not _BOUND_HASHLIB_SHA256
        or _hashlib_native.openssl_sha256 is not _BOUND_HASHLIB_SHA256
    ):
        raise ContractError("bound hashlib runtime exports changed")


def _assert_bound_reviewed_functions() -> None:
    for label, (
        module,
        attribute,
        bound_callable,
        implementation,
    ) in _BOUND_REVIEWED_FUNCTIONS.items():
        live_callable = getattr(module, attribute, None)
        if live_callable is not bound_callable:
            raise ContractError("reviewed runtime callable changed: " + label)
        _assert_callable_implementation(label, live_callable, implementation)
    for label, bound_callable in _BOUND_CONSUMED_REVIEWED_FUNCTIONS.items():
        expected_globals = (
            _FROZEN_PROTOCOL_GLOBALS
            if label.startswith("digitwise_protocol.")
            else _FROZEN_ROW_GLOBALS
        )
        if bound_callable.__globals__ is not expected_globals:
            raise ContractError("consumed reviewed function globals changed: " + label)
        _assert_callable_implementation(
            "consumed " + label,
            bound_callable,
            _BOUND_CONSUMED_REVIEWED_IMPLEMENTATIONS[label],
        )
    for attribute, bound_callable in _BOUND_REVIEWED_GENERATOR_ALIASES.items():
        if globals().get(attribute) is not bound_callable:
            raise ContractError("reviewed generator alias changed: " + attribute)
    for attribute, bound_callable in _BOUND_REVIEWED_CROSS_MODULE_ALIASES.items():
        if getattr(_row_builder_module, attribute, None) is not bound_callable:
            raise ContractError("reviewed cross-module alias changed: " + attribute)


def _assert_frozen_reviewed_globals() -> None:
    for label, live_mapping, expected_items in _BOUND_FROZEN_GLOBAL_MAPPINGS:
        if (
            type(live_mapping) is not _FrozenReviewedGlobals
            or live_mapping._sealed is not True
            or set(live_mapping) != {name for name, _ in expected_items}
        ):
            raise ContractError("frozen reviewed globals identity changed: " + label)
        for name, expected_value in expected_items:
            if live_mapping.get(name) is not expected_value:
                raise ContractError(
                    "frozen reviewed global binding changed: " + label + "." + name
                )


def _assert_bound_filesystem_exports() -> None:
    if sys.platform != _BOUND_SYS_PLATFORM or os.path is not _BOUND_OS_PATH_MODULE:
        raise ContractError("bound filesystem platform identity changed")
    if (
        sys.addaudithook is not _BOUND_SYS_ADDAUDITHOOK
        or sys.audit is not _BOUND_SYS_AUDIT
        or builtins_module.__import__ is not _BOUND_BUILTIN_IMPORT
    ):
        raise ContractError("bound runtime mutation-guard exports changed")
    if marshal.dumps is not _BOUND_MARSHAL_DUMPS:
        raise ContractError("bound marshal runtime export changed")
    for label, (
        owner,
        attribute,
        bound_callable,
        implementation,
    ) in _BOUND_FILESYSTEM_EXPORTS.items():
        live_callable = getattr(owner, attribute, None)
        if (
            live_callable is not bound_callable
            or globals().get(_BOUND_FILESYSTEM_GLOBAL_NAMES[label])
            is not bound_callable
        ):
            raise ContractError("bound filesystem runtime callable changed: " + label)
        _assert_callable_implementation(label, live_callable, implementation)
    for label, (owner, attribute, bound_value) in _BOUND_FILESYSTEM_CONSTANTS.items():
        live_value = getattr(owner, attribute, None)
        if (
            type(live_value) is not type(bound_value)
            or live_value != bound_value
            or globals().get(_BOUND_FILESYSTEM_CONSTANT_GLOBAL_NAMES[label])
            != bound_value
        ):
            raise ContractError("bound filesystem runtime constant changed: " + label)


def _assert_bound_atomic_rename() -> None:
    (
        bound_libc,
        bound_rename,
        bound_symbol,
        bound_flag,
        bound_type,
        bound_argtypes,
        bound_restype,
        bound_errcheck,
    ) = _BOUND_ATOMIC_RENAME_BINDING
    if (
        globals().get("_BOUND_LIBC") is not bound_libc
        or globals().get("_BOUND_ATOMIC_RENAME") is not bound_rename
        or globals().get("_BOUND_ATOMIC_RENAME_SYMBOL") != bound_symbol
        or globals().get("_BOUND_ATOMIC_RENAME_FLAG") != bound_flag
        or type(bound_rename) is not bound_type
        or tuple(bound_rename.argtypes or ()) != bound_argtypes
        or bound_rename.restype is not bound_restype
        or bound_errcheck is not None
        or bound_rename.errcheck is not bound_errcheck
        or "_BOUND_ATOMIC_RENAME_ERRCHECK" not in globals()
        or globals().get("_BOUND_ATOMIC_RENAME_ERRCHECK") is not bound_errcheck
    ):
        raise ContractError("bound atomic no-replace callable changed")
    (
        bound_unlinkat,
        bound_unlinkat_type,
        bound_unlinkat_argtypes,
        bound_unlinkat_restype,
        bound_unlinkat_errcheck,
        bound_at_removedir,
    ) = _BOUND_UNLINKAT_BINDING
    if (
        globals().get("_BOUND_UNLINKAT") is not bound_unlinkat
        or type(bound_unlinkat) is not bound_unlinkat_type
        or tuple(bound_unlinkat.argtypes or ()) != bound_unlinkat_argtypes
        or bound_unlinkat.restype is not bound_unlinkat_restype
        or bound_unlinkat_errcheck is not None
        or bound_unlinkat.errcheck is not bound_unlinkat_errcheck
        or "_BOUND_UNLINKAT_ERRCHECK" not in globals()
        or globals().get("_BOUND_UNLINKAT_ERRCHECK") is not bound_unlinkat_errcheck
        or globals().get("_BOUND_AT_REMOVEDIR") != bound_at_removedir
    ):
        raise ContractError("bound descriptor-relative cleanup callable changed")


def _new_bound_random(seed: Any) -> Any:
    _assert_bound_random_exports()
    instance = _BOUND_RANDOM_METHODS["__new__"][1](_BOUND_RANDOM_CLASS)
    _BOUND_RANDOM_METHODS["__init__"][1](instance, seed)
    if type(instance) is not _BOUND_RANDOM_CLASS:
        raise ContractError("bound Python random constructor returned wrong type")
    return instance


def _call_bound_random_method(instance: Any, name: str, *arguments: Any) -> Any:
    _assert_bound_random_exports()
    if type(instance) is not _BOUND_RANDOM_CLASS or name not in _BOUND_RANDOM_METHODS:
        raise ContractError("invalid bound Python random method dispatch")
    return _BOUND_RANDOM_METHODS[name][1](instance, *arguments)


SCHEMA = "shohin-dws-single-completion-v1"
PROTOCOL = "R12-DWS-SINGLE-COMPLETION-DEV-v1"
WIDTH = 4
TRAIN_EPISODES = 2_048
DEVELOPMENT_EPISODES = 256
TRAIN_PER_CELL = 128
DEVELOPMENT_PER_CELL = 16
LANES_PER_PACK = 7
LANE_LENGTH = 768
PACK_ELEMENT_BYTES = 10
PACKS_PER_UPDATE = 2
UPDATES_PER_ARM = TRAIN_EPISODES // PACKS_PER_UPDATE
GENERATION_SEED = 2_026_071_801
PAIRED_TRAINING_SEEDS = (2_026_071_811, 2_026_071_812, 2_026_071_813)
EOS_TOKEN = "<|endoftext|>"
OPERATIONS = ("add", "sub")
INTERMEDIATE_PATTERNS = tuple(
    (first, second, third) for first in (0, 1) for second in (0, 1) for third in (0, 1)
)
DATA_ARMS = ("full_trace", "decomposed_one_step", "multiline_sham")
RUNTIME_ARMS = ("full_history_replay_discard", "commit_reencode_isolation")
RUN_CELLS = tuple(
    "{}__{}".format(data_arm, runtime_arm)
    for data_arm in DATA_ARMS
    for runtime_arm in RUNTIME_ARMS
)
LANE_ROLES = (
    "full_trace_block",
    "multiline_sham_block",
    "decomposed_transition_0",
    "decomposed_transition_1",
    "decomposed_transition_2",
    "decomposed_transition_3",
    "decomposed_final",
)
SEALED_ROOT_NAME = "sealed_manifest.json"
BUNDLE_DIRECTORY_NAME = "bundle"
STAGING_OWNER_SCHEMA = "shohin-dws-single-completion-staging-owner-v1"
STAGING_NEXT_MANIFEST_NAME = ".sealed_manifest.json.next"
INPUT_LOCATION_POLICY = "sha256_authenticated_regular_file_relocation_allowed"
_BOUND_INPUT_LOCATION_POLICY = INPUT_LOCATION_POLICY
HEX64_RE = _BOUND_RE_COMPILE(r"^[0-9a-f]{64}$")
WORD_RE = _BOUND_RE_COMPILE(r"[A-Za-z0-9_]+")

KNOWN_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
KNOWN_REPLICATION_SOURCE_SHA256 = (
    "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
)
KNOWN_TRAIN_EPISODES_SHA256 = (
    "1dd913b12d2ffb2201530997102ef50a1e2d581fe7595c4e9ad5ae8c9fe3f009"
)
REPLICATION_CASE_IDS = (
    "fit_w4-00258",
    "value_ood_w4-00217",
    "fit_w4-00261",
    "fit_w4-00196",
    "fit_w6-00122",
    "value_ood_w6-00028",
    "value_ood_w6-00280",
    "value_ood_w6-00067",
    "width_ood_w8-00120",
    "width_ood_w8-00176",
    "width_ood_w8-00180",
    "width_ood_w8-00103",
)
REPLICATION_CASE_IDS_SHA256 = (
    "1dc75ec7995e61a85f7bec9ae1fa62aa1adaf71bd46172e880aea901482396b9"
)
REPLICATION_PRIOR = {
    "status": "non_preregistered_development_motivation_only",
    "full_history_paired_carry_exact": {
        "overall": "2/12",
        "by_width": {"4": "0/4", "6": "2/4", "8": "0/4"},
    },
    "stale_source_deleted_paired_carry_exact": {
        "overall": "10/12",
        "by_width": {"4": "3/4", "6": "4/4", "8": "3/4"},
    },
    "stale_source_deleted_true_exact": "12/12",
    "stale_source_deleted_counterfactual_exact": "10/12",
    "stale_source_deleted_output_switch": "12/12",
}
LOCAL_CAUSAL_PRIOR = {
    "status": "non_preregistered_development_motivation_only",
    "cases": 16,
    "nominal_second_state_exact": "5/16",
    "carry_flip_target_exact": "7/16",
    "carry_flip_output_target_switch": "3/16",
    "true_carry_one_nominal_wrong_but_flip_target_match": "6/8",
    "result_flip_output_changed": "7/16",
    "result_flip_full_target_exact": "1/16",
}
CACHE_PRUNING_PRIOR = {
    "status": "finite_non_preregistered_negative_control",
    "latest_generated_state_kv_only": "malformed_after_first_interval",
    "immutable_prefix_plus_latest_kv": "failed",
    "drop_only_stale_s0_keys": "approximately_full_history_two_state_prefix_on_two_cases",
    "conclusion": (
        "retained representations were already contextualized by S0, so posthoc KV "
        "slicing cannot support a zero-weight cache-retirement claim"
    ),
}
SOURCE_RETIREMENT_GATES = {
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
PRIMARY_GATES = {
    "treatment_advantage_each_control_min": 0.10,
    "paired_mcnemar_two_sided_p_max": 0.01,
    "model_emitted_eos_rate_min": 0.90,
    "first_state_regression_vs_decomposed_max": 0.05,
    "carry_paired_target_switch_rate_min": 0.50,
    "carry_paired_target_switch_each_nominal_carry_min": 0.40,
    "carry_advantage_each_control_min": 0.10,
}
SOURCE_PATHS = (
    "R12_DWS_SINGLE_COMPLETION_PREREG.md",
    "pipeline/generate_dws_single_completion_v1.py",
    "pipeline/test_generate_dws_single_completion_v1.py",
    "pipeline/generate_digitwise_recurrent_v1.py",
    "train/digitwise_protocol.py",
    "train/muon.py",
    "train/sft.py",
    "train/sft_encoding.py",
)
OPTIMIZER_IMPLEMENTATION_SHA256 = {
    "train/muon.py": "863e79aaaaebb681382f0c88078390b5683ab39be79ac7df60f26d1c04b21762",
    "train/sft.py": "9caa62b38a36addda9eb667b72f74dedb7165062f98bef9e1bfe49102af71921",
}
OPTIMIZER_CONTRACT = {
    "implementation_files": OPTIMIZER_IMPLEMENTATION_SHA256,
    "parameter_split": (
        "Muon receives trainable 2D parameters except names containing tok or head; "
        "AdamW receives every other trainable parameter"
    ),
    "muon": {
        "lr": 0.001,
        "momentum": 0.95,
        "nesterov": True,
        "newton_schulz_steps": 5,
        "coefficients": [3.4445, -4.7750, 2.0315],
        "normalization_epsilon": 1e-7,
        "weight_decay": 0.0,
    },
    "adamw": {
        "lr": 0.0002,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "capturable": False,
        "maximize": False,
    },
    "gradient_clip": {
        "kind": "global_l2_all_trainable_parameters_before_both_steps",
        "max_norm": 1.0,
    },
    "schedule": {
        "updates": UPDATES_PER_ARM,
        "warmup_updates": 50,
        "warmup_formula": "scale=step/50 for step<50",
        "decay_formula": ("r=(step-50)/(1024-50); scale=0.1+0.9*0.5*(1+cos(pi*r))"),
        "minimum_scale": 0.1,
        "early_stopping": False,
    },
    "loss": {
        "function": "cross_entropy",
        "ignore_index": -1,
        "reduction": (
            "sum over supervised tokens divided by the exact supervised-token count "
            "in the two-logical-pack optimizer update"
        ),
    },
    "precision": {
        "parameters": "fp32",
        "autocast": "bf16",
        "loss_and_reduction": "fp32",
        "gradient_accumulation": "fp32",
        "tf32": False,
    },
    "batching": {
        "logical_packs_per_update": PACKS_PER_UPDATE,
        "gradient_accumulation_steps": 1,
        "drop_last": False,
    },
}
ARTIFACT_NAMES = (
    "train_episodes.jsonl",
    "full_trace_train.jsonl",
    "decomposed_one_step_train.jsonl",
    "multiline_sham_train.jsonl",
    "development_board.jsonl",
    "development_commitment.json",
    "cross_width_replication_board.jsonl",
    "pack_receipts.jsonl",
    "seed_schedules.json",
    "full_trace_packs.bin",
    "decomposed_one_step_packs.bin",
    "multiline_sham_packs.bin",
    "training_plan.json",
    "audit_report.json",
)


def sha256_bytes(payload: bytes) -> str:
    return _BOUND_HASHLIB_SHA256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = _BOUND_HASHLIB_SHA256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    _assert_bound_runtime_exports()
    payload = _BOUND_JSON_DUMPS(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def pretty_json_bytes(value: Any) -> bytes:
    _assert_bound_runtime_exports()
    return (
        _BOUND_JSON_DUMPS(
            value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("ascii")


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row, newline=True) for row in rows)


def normalized(text: str) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def validate_regular_file(path: Path, label: str) -> None:
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ContractError("missing {}: {}".format(label, path)) from error
    if _BOUND_STAT_S_ISLNK(info.st_mode) or not _BOUND_STAT_S_ISREG(info.st_mode):
        raise ContractError("{} must be a regular non-symlink file".format(label))


def read_jsonl_payload(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ContractError("{} must be ASCII JSONL".format(label)) from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ContractError("blank JSONL line {} in {}".format(line_number, label))
        row = strict_json_loads((line + "\n").encode("ascii"), label)
        rows.append(row)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    payload, _ = _stable_file_bytes(path, "JSONL {}".format(path))
    return read_jsonl_payload(payload, str(path))


class _SealedRuntimeClassMeta(type, metaclass=_FrozenReviewedGlobalsMeta):
    def __setattr__(cls, name: str, value: Any) -> None:
        if type.__getattribute__(cls, "__dict__").get(
            "_runtime_descriptors_sealed", False
        ):
            raise ContractError(
                "sealed generator runtime class mutation rejected: "
                + cls.__qualname__
                + "."
                + name
            )
        type.__setattr__(cls, name, value)

    def __delattr__(cls, name: str) -> None:
        if type.__getattribute__(cls, "__dict__").get(
            "_runtime_descriptors_sealed", False
        ):
            raise ContractError(
                "sealed generator runtime class mutation rejected: "
                + cls.__qualname__
                + "."
                + name
            )
        type.__delattr__(cls, name)


class FrozenTokenizer(metaclass=_SealedRuntimeClassMeta):
    _runtime_descriptors_sealed = False

    def __init__(self, path: Path, expected_sha256: str):
        _assert_bound_tokenizer_exports()
        if not HEX64_RE.fullmatch(expected_sha256):
            raise ContractError("expected tokenizer SHA-256 is malformed")
        self.path = Path(path)
        payload, receipt = _stable_file_bytes(self.path, "tokenizer")
        self.sha256 = receipt["sha256"]
        self.size = receipt["bytes"]
        if self.sha256 != expected_sha256:
            raise ContractError("tokenizer SHA-256 mismatch")
        try:
            self.tokenizer = _BOUND_TOKENIZER_FROM_STR(
                payload.decode("utf-8", errors="strict")
            )
        except Exception as error:
            raise ContractError("failed to load tokenizer") from error
        if type(self.tokenizer) is not _BOUND_TOKENIZER_CLASS:
            raise ContractError("tokenizer constructor returned an unbound class")
        self.vocab_size = int(
            _BOUND_TOKENIZER_METHODS["get_vocab_size"](self.tokenizer)
        )
        self.eos_id = _BOUND_TOKENIZER_METHODS["token_to_id"](self.tokenizer, EOS_TOKEN)
        if self.eos_id is None:
            raise ContractError("tokenizer lacks frozen EOS token")
        self.eos_id = int(self.eos_id)
        newline = self.encode("\n")
        if len(newline) != 1 or self.decode(newline) != "\n":
            raise ContractError("LF commit delimiter must be one exact tokenizer token")
        self.commit_token_id = newline[0]

    def encode(self, text: str) -> list[int]:
        try:
            text.encode("ascii")
        except UnicodeEncodeError as error:
            raise ContractError("tokenized text must be ASCII") from error
        _assert_bound_tokenizer_exports()
        encoding = _BOUND_TOKENIZER_METHODS["encode"](
            self.tokenizer, text, add_special_tokens=False
        )
        if type(encoding) is not _BOUND_TOKENIZER_ENCODING_CLASS:
            raise ContractError("tokenizer encode returned an unbound encoding class")
        encoding_ids = _BOUND_TOKENIZER_ENCODING_IDS_DESCRIPTOR.__get__(
            encoding, _BOUND_TOKENIZER_ENCODING_CLASS
        )
        decoded = _BOUND_TOKENIZER_METHODS["decode"](
            self.tokenizer, encoding_ids, skip_special_tokens=False
        )
        if decoded != text:
            raise ContractError("tokenizer is not lossless for protocol text")
        return [int(token_id) for token_id in encoding_ids]

    def decode(self, token_ids: Iterable[int]) -> str:
        _assert_bound_tokenizer_exports()
        return _BOUND_TOKENIZER_METHODS["decode"](
            self.tokenizer, list(token_ids), skip_special_tokens=False
        )


def _episode_from_values(
    episode_id: str, split: str, operation: str, left: int, right: int
) -> dict[str, Any]:
    state = initial_state(operation, left, right, WIDTH)
    expected_states = []
    for _ in range(WIDTH):
        state = apply_microstep(state)
        expected_states.append(canonical_state(state))
    parsed = [parse_state(line) for line in expected_states]
    if any(state is None for state in parsed):
        raise AssertionError("solver emitted an unparsable state")
    pattern = [int(state["c"]) for state in parsed[:3]]
    episode = {
        "id": episode_id,
        "split": split,
        "prompt_style": "core",
        "operation": operation,
        "width": WIDTH,
        "left": left,
        "right": right,
        "initial_state": canonical_state(initial_state(operation, left, right, WIDTH)),
        "expected_states": expected_states,
        "expected_answer": state_answer(parsed[-1]),
        "intermediate_carry_pattern": pattern,
        "terminal_carry": int(parsed[-1]["c"]),
    }
    validate_episode(episode, require_terminal_c0=False)
    return episode


def validate_episode(
    episode: dict[str, Any], *, require_terminal_c0: bool = True
) -> None:
    if episode.get("width") != WIDTH or episode.get("operation") not in OPERATIONS:
        raise ContractError("episode has invalid width or operation")
    state = parse_state(episode.get("initial_state", ""))
    if state is None or state["p"] != 0 or state["c"] != 0 or state["z"] != 0:
        raise ContractError("episode initial state is invalid")
    replay = []
    for _ in range(WIDTH):
        state = apply_microstep(state)
        replay.append(canonical_state(state))
    if replay != episode.get("expected_states"):
        raise ContractError("episode expected states fail exact solver replay")
    if require_terminal_c0 and state["c"] != 0:
        raise ContractError("terminal-c=1 is forbidden in this screen")
    if state_answer(state) != episode.get("expected_answer"):
        raise ContractError("episode answer fails exact solver replay")
    pattern = [int(parse_state(line)["c"]) for line in replay[:3]]
    if pattern != episode.get("intermediate_carry_pattern"):
        raise ContractError("episode carry-pattern metadata mismatch")


def episode_scalar_values(episode: dict[str, Any]) -> set[int]:
    return {
        int(episode["left"]),
        int(episode["right"]),
        int(episode["expected_answer"]),
    }


def episode_commitment_values(episode: dict[str, Any]) -> set[int]:
    values = episode_scalar_values(episode)
    interventions = episode.get("generated_history_interventions")
    if interventions is None:
        return values
    if not isinstance(interventions, dict):
        raise ContractError("generated-history interventions must be an object")
    for branch_name in ("nominal", "carry_flip", "written_result_r0_flip"):
        branch = interventions.get(branch_name)
        if not isinstance(branch, dict) or not isinstance(
            branch.get("expected_answer"), int
        ):
            raise ContractError("intervention branch answer is malformed")
        values.add(int(branch["expected_answer"]))
    return values


def empty_overlap_inventory() -> dict[str, set[Any]]:
    return {
        "scalars": set(),
        "strings": set(),
        "exact_prompts": set(),
        "semantic_signatures": set(),
    }


def merge_overlap_inventory(
    target: dict[str, set[Any]], source: dict[str, set[Any]]
) -> None:
    expected_fields = set(empty_overlap_inventory())
    if set(target) != expected_fields or set(source) != expected_fields:
        raise ContractError("overlap inventory schema mismatch")
    for field in target:
        target[field].update(source[field])


def overlap_inventory_intersection(
    left: dict[str, set[Any]], right: dict[str, set[Any]]
) -> dict[str, set[Any]]:
    if set(left) != set(right):
        raise ContractError("overlap inventory schema mismatch")
    return {field: left[field] & right[field] for field in left}


def overlap_inventory_is_disjoint(
    left: dict[str, set[Any]], right: dict[str, set[Any]]
) -> bool:
    return not any(overlap_inventory_intersection(left, right).values())


def episode_overlap_inventory(episode: dict[str, Any]) -> dict[str, set[Any]]:
    inventory = empty_overlap_inventory()
    inventory["scalars"].update(episode_commitment_values(episode))
    initial_line = episode["initial_state"]
    base_prompt = full_trace_prompt(initial_line)
    base_response = full_trace_response(episode)
    inventory["strings"].update(
        {initial_line, base_prompt, base_response, *episode["expected_states"]}
    )
    inventory["exact_prompts"].add(base_prompt)
    inventory["semantic_signatures"].add(
        hash_json(
            {
                "kind": "episode",
                "operation": episode["operation"],
                "width": episode["width"],
                "left": int(episode["left"]),
                "right": int(episode["right"]),
                "states": episode["expected_states"],
                "answer": int(episode["expected_answer"]),
            }
        )
    )
    interventions = episode.get("generated_history_interventions")
    if interventions is not None:
        for branch_name in ("nominal", "carry_flip", "written_result_r0_flip"):
            branch = interventions[branch_name]
            inventory["strings"].update(
                {
                    branch["prefix_state"],
                    branch["full_history_prefix"],
                    branch["fresh_latest_state_prompt"],
                    branch["target_response"],
                    *branch["expected_states"],
                }
            )
            inventory["exact_prompts"].update(
                {
                    branch["full_history_prefix"],
                    branch["fresh_latest_state_prompt"],
                }
            )
            inventory["semantic_signatures"].add(
                hash_json(
                    {
                        "kind": "generated_history_intervention",
                        "operation": episode["operation"],
                        "width": episode["width"],
                        "branch": branch_name,
                        "prefix_state": parse_state(branch["prefix_state"]),
                        "expected_states": branch["expected_states"],
                        "expected_answer": branch["expected_answer"],
                    }
                )
            )
    return inventory


def episode_signature(episode: dict[str, Any]) -> tuple[Any, ...]:
    return (
        episode["operation"],
        int(episode["width"]),
        int(episode["left"]),
        int(episode["right"]),
    )


def _intervention_bundle(
    episode: dict[str, Any], *, require_terminal_c0: bool = True
) -> dict[str, Any]:
    first = parse_state(episode["expected_states"][0])
    if first is None or first["p"] != 1:
        raise ContractError("intervention prefix is malformed")
    initial_line = episode["initial_state"]

    def replay(prefix: dict[str, Any]) -> dict[str, Any]:
        states = []
        state = dict(prefix)
        while not state["z"]:
            state = apply_microstep(state)
            states.append(canonical_state(state))
        if require_terminal_c0 and state["c"] != 0:
            raise ContractError("intervention produced unsupported terminal carry")
        prefix_line = canonical_state(prefix)
        target_response = "\n".join(states + ["answer={}".format(state_answer(state))])
        return {
            "prefix_state": prefix_line,
            "full_history_prefix": full_trace_prompt(initial_line) + prefix_line + "\n",
            "fresh_latest_state_prompt": full_trace_prompt(prefix_line),
            "expected_states": states,
            "expected_answer": state_answer(state),
            "terminal_carry": int(state["c"]),
            "target_response": target_response,
            "supervised_eos_token": EOS_TOKEN,
        }

    nominal = replay(first)
    carry_state = dict(first)
    carry_state["c"] = 1 - int(carry_state["c"])
    carry = replay(carry_state)
    result_state = dict(first)
    result = list(result_state["r"])
    result[0] = str((int(result[0]) + 1) % 10)
    result_state["r"] = "".join(result)
    written_result = replay(result_state)
    if nominal["expected_states"] != episode["expected_states"][1:]:
        raise ContractError("nominal intervention continuation drifted")
    if carry["expected_states"] == nominal["expected_states"]:
        raise ContractError("carry intervention did not change its target")
    if written_result["expected_states"] == nominal["expected_states"]:
        raise ContractError("written-result intervention did not change its target")
    return {
        "intervention_position": 1,
        "nominal": nominal,
        "carry_flip": carry,
        "written_result_r0_flip": written_result,
        "scoring": {
            "target_exactness_required": True,
            "paired_output_target_switch_required": True,
            "carry_target_switch_is_promotion_veto": True,
            "host_prefix_injection_is_secondary_only": True,
        },
    }


def generate_balanced_episodes(
    *,
    rng: Any,
    split: str,
    per_cell: int,
    reserved_signatures: set[tuple[Any, ...]],
    forbidden_inventory: dict[str, set[Any]],
    reserve_inventory_within_split: bool,
    require_interventions: bool,
) -> list[dict[str, Any]]:
    if per_cell <= 0:
        raise ContractError("per-cell episode count must be positive")
    episodes = []
    reserved_inventory = empty_overlap_inventory()
    for operation in OPERATIONS:
        for pattern in INTERMEDIATE_PATTERNS:
            accepted = 0
            attempts = 0
            while accepted < per_cell:
                attempts += 1
                if attempts > per_cell * 200_000:
                    raise ContractError(
                        "could not fill balanced cell {} {}".format(operation, pattern)
                    )
                left = _call_bound_random_method(rng, "randrange", 10**WIDTH)
                right = _call_bound_random_method(rng, "randrange", 10**WIDTH)
                if operation == "sub" and left < right:
                    left, right = right, left
                episode = _episode_from_values(
                    "{}-{}-{}-{:04d}".format(
                        split, operation, "".join(map(str, pattern)), accepted
                    ),
                    split,
                    operation,
                    left,
                    right,
                )
                if tuple(episode["intermediate_carry_pattern"]) != pattern:
                    continue
                if episode["terminal_carry"] != 0:
                    continue
                validate_episode(episode)
                signature = episode_signature(episode)
                base_values = episode_scalar_values(episode)
                if signature in reserved_signatures:
                    continue
                if len(base_values) != 3:
                    continue
                if require_interventions:
                    try:
                        episode["generated_history_interventions"] = (
                            _intervention_bundle(episode)
                        )
                    except (ContractError, ValueError):
                        continue
                candidate_inventory = episode_overlap_inventory(episode)
                if not overlap_inventory_is_disjoint(
                    candidate_inventory, forbidden_inventory
                ) or not overlap_inventory_is_disjoint(
                    candidate_inventory, reserved_inventory
                ):
                    continue
                reserved_signatures.add(signature)
                if reserve_inventory_within_split:
                    merge_overlap_inventory(reserved_inventory, candidate_inventory)
                episodes.append(episode)
                accepted += 1
    expected = len(OPERATIONS) * len(INTERMEDIATE_PATTERNS) * per_cell
    if len(episodes) != expected:
        raise AssertionError("balanced episode count mismatch")
    return episodes


def full_trace_prompt(initial_line: str) -> str:
    return (
        "Autonomously continue this decimal machine. Digits in a, b, and r are "
        "least-significant first. Emit every successive canonical dws state, one per "
        "line, until z=1. Then emit answer=<integer> and end.\n"
        "Initial state: {}\nTrace:\n"
    ).format(initial_line)


def split_full_trace_prompt(initial_line: str) -> tuple[str, str, str]:
    prompt = full_trace_prompt(initial_line)
    marker = "Initial state: "
    start = prompt.index(marker) + len(marker)
    end = start + len(initial_line)
    if prompt[start:end] != initial_line:
        raise AssertionError("full-trace state-span construction failed")
    return prompt[:start], initial_line, prompt[end:]


def full_trace_response(episode: dict[str, Any]) -> str:
    return "\n".join(
        list(episode["expected_states"])
        + ["answer={}".format(episode["expected_answer"])]
    )


def validate_full_trace_response(response: str, episode: dict[str, Any]) -> None:
    if response.endswith("\n") or response.count("\n") != WIDTH:
        raise ContractError("full trace must contain exactly five nonempty lines")
    lines = response.split("\n")
    if len(lines) != WIDTH + 1 or any(not line for line in lines):
        raise ContractError("full trace line count mismatch")
    parsed = [parse_state(line) for line in lines[:WIDTH]]
    if any(state is None for state in parsed):
        raise ContractError("full trace has a malformed DWS line")
    if lines[:WIDTH] != episode["expected_states"]:
        raise ContractError("full trace does not equal exact solver replay")
    if [state["p"] for state in parsed] != [1, 2, 3, 4]:
        raise ContractError("full trace positions are not successive")
    if [state["z"] for state in parsed] != [0, 0, 0, 1]:
        raise ContractError("full trace terminal flags are invalid")
    if parse_answer(lines[-1]) != episode["expected_answer"]:
        raise ContractError("full trace answer is invalid")


def response_segments(episode: dict[str, Any]) -> list[str]:
    return [line + "\n" for line in episode["expected_states"]] + [
        "answer={}".format(episode["expected_answer"])
    ]


def _stable_order(values: Iterable[str], domain: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            sha256_bytes((domain + "\0" + value).encode("ascii")),
            value,
        ),
    )


def _perfect_matching(
    source_ids: list[str],
    candidates: dict[str, list[str]],
    domain: str,
) -> dict[str, str]:
    if len(source_ids) != len(set(source_ids)):
        raise ContractError("sham matching source IDs are not unique")
    source_set = set(source_ids)
    if set(candidates) != source_set:
        raise ContractError("sham matching candidate keys are incomplete")
    ordered_sources = _stable_order(source_ids, domain + "\0sources")
    ordered_candidates = {}
    for source_id in ordered_sources:
        donor_ids = candidates[source_id]
        if any(donor_id not in source_set for donor_id in donor_ids):
            raise ContractError("sham matching candidate is outside donor universe")
        ordered_candidates[source_id] = _stable_order(
            set(donor_ids), domain + "\0" + source_id
        )

    assignment: dict[str, str] = {}
    donor_owner: dict[str, str] = {}
    while len(assignment) < len(ordered_sources):
        distance: dict[str, int] = {}
        queue: deque[str] = deque()
        for source_id in ordered_sources:
            if source_id not in assignment:
                distance[source_id] = 0
                queue.append(source_id)

        shortest_distance: int | None = None
        while queue:
            source_id = queue.popleft()
            next_distance = distance[source_id] + 1
            if shortest_distance is not None and next_distance > shortest_distance:
                continue
            for donor_id in ordered_candidates[source_id]:
                owner = donor_owner.get(donor_id)
                if owner is None:
                    if shortest_distance is None:
                        shortest_distance = next_distance
                elif (
                    shortest_distance is None or next_distance < shortest_distance
                ) and owner not in distance:
                    distance[owner] = next_distance
                    queue.append(owner)
        if shortest_distance is None:
            break

        next_candidate = {source_id: 0 for source_id in distance}
        failed_sources: set[str] = set()
        augmented_in_phase = False
        for start_source in ordered_sources:
            if start_source in assignment or start_source not in distance:
                continue
            stack = [start_source]
            path_donors: list[str] = []
            augmented = False
            while stack:
                source_id = stack[-1]
                descended = False
                candidates_for_source = ordered_candidates[source_id]
                while next_candidate[source_id] < len(candidates_for_source):
                    donor_id = candidates_for_source[next_candidate[source_id]]
                    next_candidate[source_id] += 1
                    owner = donor_owner.get(donor_id)
                    if owner is None:
                        if distance[source_id] + 1 != shortest_distance:
                            continue
                        donor_path = path_donors + [donor_id]
                        for path_source, path_donor in zip(
                            stack, donor_path, strict=True
                        ):
                            assignment[path_source] = path_donor
                            donor_owner[path_donor] = path_source
                        augmented = True
                        augmented_in_phase = True
                        break
                    if (
                        owner not in failed_sources
                        and distance.get(owner) == distance[source_id] + 1
                        and distance[owner] < shortest_distance
                    ):
                        stack.append(owner)
                        path_donors.append(donor_id)
                        descended = True
                        break
                if augmented:
                    break
                if descended:
                    continue
                failed_sources.add(source_id)
                stack.pop()
                if path_donors:
                    path_donors.pop()
        if not augmented_in_phase:
            break

    if set(assignment) != source_set or set(assignment.values()) != source_set:
        raise ContractError("sham donor relation is not a permutation")
    return assignment


def build_sham_rows(
    episodes: list[dict[str, Any]], tokenizer: FrozenTokenizer
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_id = {episode["id"]: episode for episode in episodes}
    segment_ids = {
        episode["id"]: [
            tokenizer.encode(segment) for segment in response_segments(episode)
        ]
        for episode in episodes
    }
    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for episode in episodes:
        strata[
            (episode["operation"], tuple(episode["intermediate_carry_pattern"]))
        ].append(episode["id"])
    donors_by_source = {episode["id"]: [] for episode in episodes}
    for stratum, source_ids in sorted(strata.items()):
        if len(source_ids) < 6:
            raise ContractError("sham matching needs at least six episodes per stratum")
        for part_index in range(WIDTH):
            candidates = {}
            for source_id in source_ids:
                used = set(donors_by_source[source_id]) | {source_id}
                candidates[source_id] = [
                    donor_id
                    for donor_id in source_ids
                    if donor_id not in used
                    and len(segment_ids[donor_id][part_index])
                    == len(segment_ids[source_id][part_index])
                ]
            matching = _perfect_matching(
                source_ids,
                candidates,
                "{}-{}-line{}".format(PROTOCOL, stratum, part_index),
            )
            for source_id, donor_id in matching.items():
                donors_by_source[source_id].append(donor_id)

    all_ids = [episode["id"] for episode in episodes]
    answer_candidates = {}
    for source_id in all_ids:
        source = by_id[source_id]
        last_line_donor = by_id[donors_by_source[source_id][-1]]
        forbidden_answers = {
            int(source["expected_answer"]),
            int(last_line_donor["expected_answer"]),
        }
        used = set(donors_by_source[source_id]) | {source_id}
        answer_candidates[source_id] = [
            donor_id
            for donor_id in all_ids
            if donor_id not in used
            and int(by_id[donor_id]["expected_answer"]) not in forbidden_answers
            and len(segment_ids[donor_id][-1]) == len(segment_ids[source_id][-1])
        ]
    answer_matching = _perfect_matching(
        all_ids, answer_candidates, PROTOCOL + "-answer-donor"
    )
    for source_id, donor_id in answer_matching.items():
        donors_by_source[source_id].append(donor_id)

    sham_rows = []
    for episode in episodes:
        source_id = episode["id"]
        donors = donors_by_source[source_id]
        lines = [
            by_id[donors[index]]["expected_states"][index] for index in range(WIDTH)
        ]
        sham_answer = int(by_id[donors[-1]]["expected_answer"])
        response = "\n".join(lines + ["answer={}".format(sham_answer)])
        if sham_answer == int(episode["expected_answer"]):
            raise ContractError("sham answer leaked the treatment answer")
        parsed = [parse_state(line) for line in lines]
        if any(state is None for state in parsed):
            raise ContractError("sham line is not independently valid")
        if state_answer(parsed[-1]) == sham_answer:
            raise ContractError("sham answer accidentally matches its terminal line")
        previous = parse_state(episode["initial_state"])
        for line, current in zip(lines, parsed, strict=True):
            if canonical_state(apply_microstep(previous)) == line:
                raise ContractError("sham contains a valid cross-line transition")
            previous = current
        if [state["p"] for state in parsed] != [1, 2, 3, 4]:
            raise ContractError("sham line positions are invalid")
        if [state["c"] for state in parsed[:3]] != episode[
            "intermediate_carry_pattern"
        ]:
            raise ContractError("sham carry marginals are not matched")
        if any(state["op"] != episode["operation"] for state in parsed):
            raise ContractError("sham operation marginal is not matched")
        source_segment_lengths = [len(ids) for ids in segment_ids[source_id]]
        sham_segment_lengths = [
            len(segment_ids[donors[index]][index]) for index in range(WIDTH)
        ] + [len(segment_ids[donors[-1]][-1])]
        if sham_segment_lengths != source_segment_lengths:
            raise ContractError("sham token budget is not exactly matched")
        sham_rows.append(
            {
                "id": "sham-" + source_id,
                "episode_id": source_id,
                "question": full_trace_prompt(episode["initial_state"]),
                "completion_prompt": full_trace_prompt(episode["initial_state"]),
                "response": response,
                "source": "dws_single_completion_multiline_sham_v1",
                "training_group": "dws_single_completion_multiline_sham",
                "kind": "multiline_sham",
                "operation": episode["operation"],
                "width": WIDTH,
                "intermediate_carry_pattern": episode["intermediate_carry_pattern"],
                "line_donor_episode_ids": donors[:WIDTH],
                "answer_donor_episode_id": donors[-1],
                "sham_answer": sham_answer,
                "treatment_answer": episode["expected_answer"],
                "all_adjacent_transitions_broken": True,
                "answer_relation_broken": True,
                "eos_supervised_by_packer": True,
            }
        )
    return sham_rows, donors_by_source


def _control_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows_from_episode(episode)
        if row["kind"] in ("transition", "final")
    ]
    if [row["kind"] for row in selected] != [
        "transition",
        "transition",
        "transition",
        "transition",
        "final",
    ]:
        raise ContractError("decomposed control row layout drifted")
    for lane_index, row in enumerate(selected):
        row["id"] = "control-{}-lane{}".format(episode["id"], lane_index)
        row["pack_episode_id"] = episode["id"]
        row["lane_index"] = lane_index
        row["training_group"] = "dws_single_completion_decomposed_control"
        row["supervised_terminator"] = "LF" if lane_index < WIDTH else "EOS"
        row["nonfinal_eos_supervised"] = False
    return selected


def _trace_row(episode: dict[str, Any]) -> dict[str, Any]:
    response = full_trace_response(episode)
    validate_full_trace_response(response, episode)
    prompt = full_trace_prompt(episode["initial_state"])
    return {
        "id": "trace-" + episode["id"],
        "episode_id": episode["id"],
        "question": prompt,
        "completion_prompt": prompt,
        "response": response,
        "source": "dws_single_completion_full_trace_v1",
        "training_group": "dws_single_completion_full_trace",
        "kind": "full_trace",
        "operation": episode["operation"],
        "width": WIDTH,
        "intermediate_carry_pattern": episode["intermediate_carry_pattern"],
        "successive_state_lines": WIDTH,
        "answer_lines": 1,
        "eos_supervised_by_packer": True,
    }


def _encode_prompt_with_state_epochs(
    tokenizer: FrozenTokenizer, prompt: str, state_line: str
) -> tuple[list[int], list[int]]:
    start = prompt.index(state_line)
    end = start + len(state_line)
    pieces = (prompt[:start], state_line, prompt[end:])
    ids = []
    epochs = []
    for index, piece in enumerate(pieces):
        piece_ids = tokenizer.encode(piece)
        ids.extend(piece_ids)
        epochs.extend(([1] if index == 1 else [0]) * len(piece_ids))
    if tokenizer.decode(ids) != prompt:
        raise ContractError("independent prompt pieces are not losslessly composable")
    return ids, epochs


def _pack_lane(
    *,
    tokenizer: FrozenTokenizer,
    prompt: str,
    state_line: str,
    target_pieces: list[tuple[list[int], int]],
    supervise_eos: bool,
    lane_length: int,
) -> dict[str, list[int]]:
    prompt_ids, prompt_epochs = _encode_prompt_with_state_epochs(
        tokenizer, prompt, state_line
    )
    token_ids = list(prompt_ids)
    epoch_ids = list(prompt_epochs)
    loss_mask = [0] * len(token_ids)
    for piece_ids, epoch in target_pieces:
        token_ids.extend(piece_ids)
        epoch_ids.extend([epoch] * len(piece_ids))
        loss_mask.extend([1] * len(piece_ids))
    if supervise_eos:
        token_ids.append(tokenizer.eos_id)
        final_epoch = target_pieces[-1][1] if target_pieces else 1
        epoch_ids.append(final_epoch)
        loss_mask.append(1)
    if len(token_ids) > lane_length:
        raise ContractError(
            "encoded lane exceeds {} tokens: {}".format(lane_length, len(token_ids))
        )
    attention_mask = [1] * len(token_ids)
    padding = lane_length - len(token_ids)
    token_ids.extend([tokenizer.eos_id] * padding)
    attention_mask.extend([0] * padding)
    loss_mask.extend([0] * padding)
    epoch_ids.extend([65_535] * padding)
    if len(token_ids) != lane_length:
        raise AssertionError("lane padding failed")
    if loss_mask[0] or any(
        loss and not attention
        for loss, attention in zip(loss_mask, attention_mask, strict=True)
    ):
        raise ContractError("invalid causal loss mask")
    return {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "epoch_ids": epoch_ids,
        "position_ids": list(range(lane_length)),
    }


def _empty_lane(tokenizer: FrozenTokenizer, lane_length: int) -> dict[str, list[int]]:
    return {
        "token_ids": [tokenizer.eos_id] * lane_length,
        "attention_mask": [0] * lane_length,
        "loss_mask": [0] * lane_length,
        "epoch_ids": [65_535] * lane_length,
        "position_ids": list(range(lane_length)),
    }


def _loss_masked_lane(lane: dict[str, list[int]]) -> dict[str, list[int]]:
    return {
        "token_ids": list(lane["token_ids"]),
        "attention_mask": list(lane["attention_mask"]),
        "loss_mask": [0] * len(lane["loss_mask"]),
        "epoch_ids": list(lane["epoch_ids"]),
        "position_ids": list(lane["position_ids"]),
    }


def _pack_payload(lanes: list[dict[str, list[int]]], lane_length: int) -> bytes:
    if len(lanes) != LANES_PER_PACK:
        raise ContractError("logical pack must contain {} lanes".format(LANES_PER_PACK))
    token_ids = [token for lane in lanes for token in lane["token_ids"]]
    attention = [token for lane in lanes for token in lane["attention_mask"]]
    loss = [token for lane in lanes for token in lane["loss_mask"]]
    epochs = [token for lane in lanes for token in lane["epoch_ids"]]
    positions = [token for lane in lanes for token in lane["position_ids"]]
    expected = LANES_PER_PACK * lane_length
    if not all(
        len(values) == expected
        for values in (token_ids, attention, loss, epochs, positions)
    ):
        raise AssertionError("pack vector length mismatch")
    dense_positions = list(range(lane_length))
    if any(lane["position_ids"] != dense_positions for lane in lanes):
        raise ContractError("lane position IDs are not the normative dense range")
    return b"".join(
        (
            _BOUND_STRUCT_PACK("<{}I".format(expected), *token_ids),
            bytes(attention),
            bytes(loss),
            _BOUND_STRUCT_PACK("<{}H".format(expected), *epochs),
            _BOUND_STRUCT_PACK("<{}H".format(expected), *positions),
        )
    )


def unpack_pack_payload(payload: bytes, lane_length: int) -> list[dict[str, list[int]]]:
    count = LANES_PER_PACK * lane_length
    expected_bytes = count * PACK_ELEMENT_BYTES
    if len(payload) != expected_bytes:
        raise ContractError("pack payload byte length mismatch")
    token_end = count * 4
    attention_end = token_end + count
    loss_end = attention_end + count
    epoch_end = loss_end + count * 2
    token_ids = list(_BOUND_STRUCT_UNPACK("<{}I".format(count), payload[:token_end]))
    attention = list(payload[token_end:attention_end])
    loss = list(payload[attention_end:loss_end])
    epochs = list(
        _BOUND_STRUCT_UNPACK("<{}H".format(count), payload[loss_end:epoch_end])
    )
    positions = list(_BOUND_STRUCT_UNPACK("<{}H".format(count), payload[epoch_end:]))
    lanes = []
    for lane_index in range(LANES_PER_PACK):
        start = lane_index * lane_length
        end = start + lane_length
        lanes.append(
            {
                "token_ids": token_ids[start:end],
                "attention_mask": attention[start:end],
                "loss_mask": loss[start:end],
                "epoch_ids": epochs[start:end],
                "position_ids": positions[start:end],
            }
        )
    dense_positions = list(range(lane_length))
    if any(lane["position_ids"] != dense_positions for lane in lanes):
        raise ContractError("serialized lane position IDs are not dense")
    return lanes


_BOUND_PACK_PAYLOAD = _pack_payload
_BOUND_PACK_PAYLOAD_CODE = _pack_payload.__code__
_PACK_MUTATION_GUARD = _install_callable_mutation_guard(
    "pack-payload-v1", (_BOUND_PACK_PAYLOAD,)
)


def _assert_bound_pack_payload() -> None:
    if (
        globals().get("_pack_payload") is not _BOUND_PACK_PAYLOAD
        or _BOUND_PACK_PAYLOAD.__code__ is not _BOUND_PACK_PAYLOAD_CODE
    ):
        raise ContractError("bound pack payload callable changed")


def _supervised_ids(lanes: list[dict[str, list[int]]]) -> list[int]:
    return [
        token_id
        for lane in lanes
        for token_id, supervised in zip(
            lane["token_ids"], lane["loss_mask"], strict=True
        )
        if supervised
    ]


def _active_tokens(lanes: list[dict[str, list[int]]]) -> int:
    return sum(sum(lane["attention_mask"]) for lane in lanes)


def _pack_arms_for_episode(
    *,
    episode: dict[str, Any],
    sham_row: dict[str, Any],
    tokenizer: FrozenTokenizer,
    lane_length: int,
) -> dict[str, list[dict[str, list[int]]]]:
    source_segments = response_segments(episode)
    source_segment_ids = [tokenizer.encode(segment) for segment in source_segments]
    prompt = full_trace_prompt(episode["initial_state"])
    trace_pieces = [
        (segment_ids, index + 2)
        for index, segment_ids in enumerate(source_segment_ids[:WIDTH])
    ] + [(source_segment_ids[-1], WIDTH + 2)]
    trace_lane = _pack_lane(
        tokenizer=tokenizer,
        prompt=prompt,
        state_line=episode["initial_state"],
        target_pieces=trace_pieces,
        supervise_eos=True,
        lane_length=lane_length,
    )

    control_rows = _control_rows(episode)
    control_lanes = []
    for index, row in enumerate(control_rows):
        piece_ids = source_segment_ids[index]
        state_line = row["state"]
        control_lanes.append(
            _pack_lane(
                tokenizer=tokenizer,
                prompt=row["completion_prompt"],
                state_line=state_line,
                target_pieces=[(piece_ids, 2)],
                supervise_eos=index == WIDTH,
                lane_length=lane_length,
            )
        )

    sham_lines = sham_row["response"].split("\n")
    if len(sham_lines) != WIDTH + 1:
        raise ContractError("sham full-trace surface is malformed")
    sham_segments = [line + "\n" for line in sham_lines[:WIDTH]] + [sham_lines[-1]]
    sham_segment_ids = [tokenizer.encode(segment) for segment in sham_segments]
    if [len(ids) for ids in sham_segment_ids] != [
        len(ids) for ids in source_segment_ids
    ]:
        raise ContractError("sham per-segment token lengths drifted")
    sham_pieces = [
        (segment_ids, index + 2)
        for index, segment_ids in enumerate(sham_segment_ids[:WIDTH])
    ] + [(sham_segment_ids[-1], WIDTH + 2)]
    sham_lane = _pack_lane(
        tokenizer=tokenizer,
        prompt=prompt,
        state_line=episode["initial_state"],
        target_pieces=sham_pieces,
        supervise_eos=True,
        lane_length=lane_length,
    )
    block_lanes = [trace_lane, sham_lane, *control_lanes]
    if len(block_lanes) != LANES_PER_PACK or len(LANE_ROLES) != LANES_PER_PACK:
        raise AssertionError("block-diagonal lane layout drifted")
    full_trace_lanes = [
        lane if index == 0 else _loss_masked_lane(lane)
        for index, lane in enumerate(block_lanes)
    ]
    sham_lanes = [
        lane if index == 1 else _loss_masked_lane(lane)
        for index, lane in enumerate(block_lanes)
    ]
    control_lane_indices = set(range(2, LANES_PER_PACK))
    decomposed_lanes = [
        lane if index in control_lane_indices else _loss_masked_lane(lane)
        for index, lane in enumerate(block_lanes)
    ]
    arms = {
        "full_trace": full_trace_lanes,
        "decomposed_one_step": decomposed_lanes,
        "multiline_sham": sham_lanes,
    }
    trace_targets = _supervised_ids(full_trace_lanes)
    control_targets = _supervised_ids(control_lanes)
    sham_targets = _supervised_ids(sham_lanes)
    if trace_targets != control_targets:
        raise ContractError("treatment/control supervised token sequence mismatch")
    if len(trace_targets) != len(sham_targets):
        raise ContractError("sham supervised token count mismatch")
    reference_tokens = [lane["token_ids"] for lane in full_trace_lanes]
    reference_attention = [lane["attention_mask"] for lane in full_trace_lanes]
    reference_epochs = [lane["epoch_ids"] for lane in full_trace_lanes]
    for arm, lanes in arms.items():
        if [lane["token_ids"] for lane in lanes] != reference_tokens:
            raise ContractError("{} active context token IDs differ".format(arm))
        if [lane["attention_mask"] for lane in lanes] != reference_attention:
            raise ContractError("{} attention surface differs".format(arm))
        if [lane["epoch_ids"] for lane in lanes] != reference_epochs:
            raise ContractError("{} state-epoch surface differs".format(arm))
    return arms


def score_paired_target_switch(
    nominal_output: str,
    nominal_target: str,
    counterfactual_output: str,
    counterfactual_target: str,
) -> dict[str, bool]:
    nominal_exact = nominal_output == nominal_target
    counterfactual_exact = counterfactual_output == counterfactual_target
    output_changed = nominal_output != counterfactual_output
    target_changed = nominal_target != counterfactual_target
    return {
        "nominal_target_exact": nominal_exact,
        "counterfactual_target_exact": counterfactual_exact,
        "output_changed": output_changed,
        "target_changed": target_changed,
        "paired_target_switch": bool(
            nominal_exact and counterfactual_exact and output_changed and target_changed
        ),
    }


def _derived_seed(seed: int, domain: str) -> int:
    digest = _BOUND_HASHLIB_SHA256(
        "{}\0{}\0{}".format(PROTOCOL, seed, domain).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def build_seed_schedules(source_episode_ids: list[str]) -> dict[str, Any]:
    if len(source_episode_ids) != len(set(source_episode_ids)):
        raise ContractError("seed schedule source IDs are not unique")
    schedules = []
    for seed in PAIRED_TRAINING_SEEDS:
        rng = _new_bound_random(seed)
        initial_state_sha256 = hash_json(_call_bound_random_method(rng, "getstate"))
        startup_probe_u64 = [
            _call_bound_random_method(rng, "getrandbits", 64) for _ in range(4)
        ]
        post_probe_state_sha256 = hash_json(_call_bound_random_method(rng, "getstate"))
        order = list(source_episode_ids)
        _call_bound_random_method(rng, "shuffle", order)
        order_sha256 = sha256_bytes(
            "".join(value + "\n" for value in order).encode("ascii")
        )
        schedules.append(
            {
                "seed": seed,
                "rng_initialization": {
                    "python_seed": seed,
                    "numpy_seed": _derived_seed(seed, "numpy"),
                    "torch_cpu_seed": _derived_seed(seed, "torch_cpu"),
                    "torch_cuda_seed": _derived_seed(seed, "torch_cuda"),
                    "initial_python_state_sha256": initial_state_sha256,
                    "first_rng_use": "four frozen 64-bit startup probes, then pack shuffle",
                    "startup_probe_u64": startup_probe_u64,
                    "post_probe_python_state_sha256": post_probe_state_sha256,
                    "dropout": "disabled",
                },
                "pack_order_episode_ids": order,
                "pack_order_sha256": order_sha256,
                "run_cell_pack_order_sha256": {
                    cell: order_sha256 for cell in RUN_CELLS
                },
            }
        )
    schedule_hashes = [row["pack_order_sha256"] for row in schedules]
    startup_probes = [
        tuple(row["rng_initialization"]["startup_probe_u64"]) for row in schedules
    ]
    if len(set(schedule_hashes)) != len(PAIRED_TRAINING_SEEDS):
        raise ContractError("paired seeds produced repeated pack schedules")
    if len(set(startup_probes)) != len(PAIRED_TRAINING_SEEDS):
        raise ContractError("paired seeds produced repeated initial RNG probes")
    return {
        "schema": "shohin-dws-single-completion-seed-schedules-v1",
        "protocol": PROTOCOL,
        "paired_across_run_cells": True,
        "schedules": schedules,
    }


def load_replication_board(
    source_path: Path, expected_sha256: str
) -> list[dict[str, Any]]:
    payload, receipt = _stable_file_bytes(source_path, "cross-width replication source")
    if receipt["sha256"] != expected_sha256:
        raise ContractError("cross-width replication source SHA-256 mismatch")
    source_rows = read_jsonl_payload(payload, "cross-width replication source")
    by_id = {row.get("id"): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ContractError("cross-width replication source has duplicate IDs")
    if len(by_id) == 0 or any(case_id not in by_id for case_id in REPLICATION_CASE_IDS):
        raise ContractError("cross-width replication source lacks frozen IDs")
    joined = "".join(case_id + "\n" for case_id in REPLICATION_CASE_IDS).encode("ascii")
    if sha256_bytes(joined) != REPLICATION_CASE_IDS_SHA256:
        raise AssertionError("frozen replication case-list hash drifted")
    board = []
    for case_id in REPLICATION_CASE_IDS:
        row = by_id[case_id]
        state = parse_state(row.get("initial_state", ""))
        if state is None:
            raise ContractError("replication row has invalid initial state")
        expected_states = row.get("expected_states")
        if not isinstance(expected_states, list) or len(expected_states) != state["w"]:
            raise ContractError("replication row has invalid expected trace")
        replay = []
        current = state
        for _ in range(state["w"]):
            current = apply_microstep(current)
            replay.append(canonical_state(current))
        if replay != expected_states or state_answer(current) != row.get(
            "expected_answer"
        ):
            raise ContractError("replication row fails exact solver replay")
        left = sum(int(digit) * 10**index for index, digit in enumerate(state["a"]))
        right = sum(int(digit) * 10**index for index, digit in enumerate(state["b"]))
        episode = {
            "id": case_id,
            "split": row.get("split"),
            "operation": state["op"],
            "width": state["w"],
            "left": left,
            "right": right,
            "initial_state": row["initial_state"],
            "expected_states": expected_states,
            "expected_answer": row["expected_answer"],
        }
        interventions = _intervention_bundle(episode, require_terminal_c0=False)
        board.append(
            {
                "id": case_id,
                "regime": row.get("split"),
                "operation": state["op"],
                "width": state["w"],
                "left": left,
                "right": right,
                "initial_state": row["initial_state"],
                "expected_states": expected_states,
                "expected_answer": row["expected_answer"],
                "generated_history_interventions": interventions,
                "promotion_authority": False,
            }
        )
    cells = Counter((row["width"], row["operation"]) for row in board)
    expected_cells = {
        (width, operation): 2 for width in (4, 6, 8) for operation in OPERATIONS
    }
    if dict(cells) != expected_cells:
        raise ContractError("cross-width replication board is not 2-per-cell balanced")
    unique_fields = {
        "id": [row["id"] for row in board],
        "initial_state": [row["initial_state"] for row in board],
        "exact_prompt": [full_trace_prompt(row["initial_state"]) for row in board],
        "semantic_signature": [episode_signature(row) for row in board],
    }
    for label, values in unique_fields.items():
        if len(values) != len(set(values)):
            raise ContractError("cross-width board collapsed on {}".format(label))
    branch_pairs = []
    for row in board:
        for branch_name in ("nominal", "carry_flip", "written_result_r0_flip"):
            branch = row["generated_history_interventions"][branch_name]
            branch_pairs.append(
                (
                    branch["full_history_prefix"],
                    branch["fresh_latest_state_prompt"],
                    branch["target_response"],
                )
            )
    if len(branch_pairs) != len(set(branch_pairs)):
        raise ContractError("cross-width intervention prefixes or targets collapsed")
    return board


def source_bindings() -> dict[str, dict[str, Any]]:
    bindings = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        _, receipt = _stable_file_bytes(path, "source {}".format(relative))
        digest = receipt["sha256"]
        if (
            relative in OPTIMIZER_IMPLEMENTATION_SHA256
            and digest != OPTIMIZER_IMPLEMENTATION_SHA256[relative]
        ):
            raise ContractError("frozen optimizer implementation drifted: " + relative)
        bindings[relative] = {
            "bytes": receipt["bytes"],
            "sha256": digest,
        }
    return bindings


def source_bindings_sha256() -> str:
    return hash_json(source_bindings())


def strict_json_loads(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("duplicate JSON key in {}: {}".format(label, key))
            result[key] = value
        return result

    def reject_non_finite(value: Any, location: str = "$") -> None:
        if isinstance(value, float):
            if not _BOUND_MATH_ISFINITE(value):
                raise ContractError(
                    "non-finite decoded JSON number in {} at {}".format(label, location)
                )
            return
        if isinstance(value, dict):
            for key, item in value.items():
                reject_non_finite(item, "{}.{}".format(location, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                reject_non_finite(item, "{}[{}]".format(location, index))

    try:
        text = payload.decode("ascii")
        _assert_bound_runtime_exports()
        value = _BOUND_JSON_LOADS(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ContractError("non-finite JSON constant in {}: {}".format(label, token))
            ),
        )
    except ContractError:
        raise
    except (
        UnicodeDecodeError,
        _BOUND_JSON_DECODE_ERROR,
        TypeError,
        ValueError,
    ) as error:
        raise ContractError("invalid ASCII JSON in {}".format(label)) from error
    reject_non_finite(value)
    if not isinstance(value, dict):
        raise ContractError("{} must contain one JSON object".format(label))
    return value


def _recursively_type_strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        if len(actual) != len(expected) or set(actual) != set(expected):
            return False
        return all(
            _recursively_type_strict_equal(actual[key], expected[key])
            for key in expected
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _recursively_type_strict_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _stable_file_bytes(
    path: Path,
    label: str,
    *,
    exact_mode: int | None = None,
    require_nlink_one: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    path = Path(path)
    flags = _BOUND_OS_O_RDONLY | _BOUND_OS_O_NOFOLLOW
    try:
        descriptor = _BOUND_OS_OPEN(path, flags)
    except OSError as error:
        raise ContractError("{} cannot be opened safely".format(label)) from error
    try:
        before = _BOUND_OS_FSTAT(descriptor)
        if not _BOUND_STAT_S_ISREG(before.st_mode):
            raise ContractError("{} must be a regular file".format(label))
        if require_nlink_one and before.st_nlink != 1:
            raise ContractError("{} must have exactly one hard link".format(label))
        if exact_mode is not None and _BOUND_STAT_S_IMODE(before.st_mode) != exact_mode:
            raise ContractError("{} mode must be {:04o}".format(label, exact_mode))
        blocks = []
        while True:
            block = _BOUND_OS_READ(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = _BOUND_OS_FSTAT(descriptor)
    finally:
        _BOUND_OS_CLOSE(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ContractError("{} changed during its sealed read".format(label))
    return payload, {
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _module_file_receipt(
    module: Any,
    module_name: str,
    *,
    expected_path: Path | None = None,
    allow_direct_script_without_origin: bool = False,
) -> dict[str, Any]:
    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    module_origin = getattr(module_spec, "origin", None)
    if not module_file:
        raise ContractError("runtime module has no file origin: " + module_name)
    if not module_origin and not (
        allow_direct_script_without_origin and module_spec is None
    ):
        raise ContractError("runtime module has no file origin: " + module_name)
    presented_path = Path(module_file).absolute()
    try:
        resolved_path = Path(module_file).resolve(strict=True)
        resolved_origin = (
            None if module_origin is None else Path(module_origin).resolve(strict=True)
        )
    except OSError as error:
        raise ContractError(
            "runtime module path cannot be resolved: {}".format(module_name)
        ) from error
    if resolved_origin is not None and resolved_path != resolved_origin:
        raise ContractError("runtime module file/origin mismatch: " + module_name)
    if expected_path is not None:
        try:
            expected_resolved = Path(expected_path).resolve(strict=True)
        except OSError as error:
            raise ContractError(
                "expected runtime module path cannot be resolved: {}".format(
                    module_name
                )
            ) from error
        if resolved_path != expected_resolved:
            raise ContractError("runtime module path mismatch: " + module_name)
    _, receipt = _stable_file_bytes(resolved_path, "runtime module " + module_name)
    return {
        "module": module_name,
        "path": str(presented_path),
        "resolved_path": str(resolved_path),
        "origin_binding": (
            "direct_script_file_without_import_spec"
            if resolved_origin is None
            else "module_file_equals_import_spec_origin"
        ),
        **receipt,
    }


def _runtime_dependency_receipt(module: Any, module_name: str) -> dict[str, Any]:
    module_spec = getattr(module, "__spec__", None)
    origin = getattr(module_spec, "origin", None)
    loader = getattr(module_spec, "loader", None)
    if origin in ("built-in", "frozen"):
        return {
            "module": module_name,
            "origin": origin,
            "implementation_binding": "Python executable bytes",
            "loader_module": type(loader).__module__,
            "loader_qualname": type(loader).__qualname__,
        }
    return _module_file_receipt(module, module_name)


def _bound_callable_receipt(name: str, value: Any) -> dict[str, Any]:
    receipt = {
        "name": name,
        "callable_module": getattr(value, "__module__", None),
        "callable_name": getattr(value, "__name__", None),
        "callable_qualname": getattr(value, "__qualname__", None),
        "callable_type_module": type(value).__module__,
        "callable_type_qualname": type(value).__qualname__,
        "captured_object_identity_required": True,
        "bound_owner_module": getattr(
            getattr(value, "__self__", None), "__name__", None
        ),
    }
    code = getattr(value, "__code__", None)
    if code is not None:
        receipt.update(
            {
                "implementation_kind": "python_code",
                "code_sha256": sha256_bytes(_stable_code_bytes(code)),
                "defaults_sha256": _implementation_value_sha256(
                    getattr(value, "__defaults__", None)
                ),
                "keyword_defaults_sha256": _implementation_value_sha256(
                    getattr(value, "__kwdefaults__", None)
                ),
                "closure_sha256": _implementation_value_sha256(
                    tuple(
                        cell.cell_contents
                        for cell in (getattr(value, "__closure__", None) or ())
                    )
                ),
            }
        )
    else:
        receipt["implementation_kind"] = "bound_native_runtime"
    return receipt


def _bound_descriptor_receipt(
    owner: type[Any], name: str, descriptor: Any
) -> dict[str, Any]:
    value = {
        "name": name,
        "owner_module": owner.__module__,
        "owner_qualname": owner.__qualname__,
        "descriptor_type": type(descriptor).__name__,
    }
    code = getattr(descriptor, "__code__", None)
    if code is not None:
        value["code_sha256"] = sha256_bytes(_stable_code_bytes(code))
        value["defaults_sha256"] = _implementation_value_sha256(
            getattr(descriptor, "__defaults__", None)
        )
        value["keyword_defaults_sha256"] = _implementation_value_sha256(
            getattr(descriptor, "__kwdefaults__", None)
        )
    return value


def runtime_bindings(
    _assert_generator_runtime: Any = None,
    _generator_runtime_receipt: Any = None,
) -> dict[str, Any]:
    if _assert_generator_runtime is None or _generator_runtime_receipt is None:
        raise ContractError("generator runtime boundary was not finalized")
    _assert_generator_runtime()
    _assert_isolated_startup()
    if _startup_flags_receipt() != _BOUND_STARTUP_FLAGS_RECEIPT:
        raise ContractError("Python startup flags changed after binding")
    _assert_bound_runtime_modules()
    _assert_bound_runtime_exports()
    _assert_bound_generator_aliases()
    _assert_sealed_runtime_classes()
    _assert_frozen_generator_builtins()
    _assert_bound_hashlib_exports()
    _assert_bound_struct_exports()
    _assert_bound_pack_payload()
    _assert_bound_tokenizer_exports()
    _assert_bound_random_exports()
    _assert_bound_reviewed_functions()
    _assert_frozen_reviewed_globals()
    _assert_bound_filesystem_exports()
    _assert_bound_atomic_rename()
    _assert_sealed_generator_module()
    if _BOUND_SYS_MODULES.get(__name__) is not _BOUND_GENERATOR_MODULE:
        raise ContractError("executing generator module identity changed")
    if (
        __name__ != _BOUND_GENERATOR_EXECUTION_NAME
        or getattr(_BOUND_GENERATOR_MODULE, "__file__", None)
        != _BOUND_GENERATOR_MODULE_FILE
        or getattr(_BOUND_GENERATOR_MODULE, "__spec__", None)
        is not _BOUND_GENERATOR_MODULE_SPEC
        or _BOUND_GENERATOR_DIRECT_SCRIPT
        != (
            _BOUND_GENERATOR_EXECUTION_NAME == "__main__"
            and _BOUND_GENERATOR_MODULE_SPEC is None
        )
    ):
        raise ContractError("executing generator origin binding changed")
    generator_receipt = _module_file_receipt(
        _BOUND_GENERATOR_MODULE,
        _BOUND_GENERATOR_EXECUTION_NAME,
        expected_path=_GENERATOR_PATH,
        allow_direct_script_without_origin=_BOUND_GENERATOR_DIRECT_SCRIPT,
    )
    if {
        "bytes": generator_receipt["bytes"],
        "sha256": generator_receipt["sha256"],
    } != _generator_import_receipt:
        raise ContractError("executing generator differs from imported source bytes")
    reviewed_modules = {
        "digitwise_protocol": (
            _digitwise_protocol_module,
            _DIGITWISE_PROTOCOL_PATH,
            "train/digitwise_protocol.py",
        ),
        "pipeline.generate_digitwise_recurrent_v1": (
            _row_builder_module,
            _ROW_BUILDER_PATH,
            "pipeline/generate_digitwise_recurrent_v1.py",
        ),
    }
    reviewed_receipts = {}
    for module_name, (module, path, source_binding_path) in reviewed_modules.items():
        if _BOUND_SYS_MODULES.get(module_name) is not module:
            raise ContractError(
                "reviewed runtime module identity changed: " + module_name
            )
        module_receipt = _module_file_receipt(module, module_name, expected_path=path)
        if {
            "bytes": module_receipt["bytes"],
            "sha256": module_receipt["sha256"],
        } != _IMPORTED_REVIEWED_MODULE_RECEIPTS[module_name]:
            raise ContractError(
                "reviewed runtime module differs from imported bytes: " + module_name
            )
        reviewed_receipts[module_name] = {
            **module_receipt,
            "source_binding_path": source_binding_path,
        }
    try:
        distribution_version = _BOUND_IMPORTLIB_METADATA_VERSION("tokenizers")
    except _BOUND_IMPORTLIB_METADATA_PACKAGE_NOT_FOUND_ERROR as error:
        raise ContractError("the bound tokenizers runtime is unavailable") from error
    if distribution_version != _BOUND_TOKENIZERS_VERSION:
        raise ContractError("tokenizers distribution version changed")

    executable_presented = Path(sys.executable).absolute()
    try:
        executable_resolved = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise ContractError("Python executable cannot be resolved") from error
    _, executable_receipt = _stable_file_bytes(executable_resolved, "Python executable")
    return {
        "schema": "shohin-dws-single-completion-runtime-bindings-v6",
        "python": {
            "implementation": _BOUND_PYTHON_IMPLEMENTATION,
            "version": sys.version,
            "version_info": list(sys.version_info),
            "build": list(_BOUND_PYTHON_BUILD),
            "compiler": _BOUND_PYTHON_COMPILER,
            "api_version": sys.api_version,
            "startup": {
                "required_invocation_flags": ["-I", "-S", "-B"],
                "flags": _BOUND_STARTUP_FLAGS_RECEIPT,
                "isolated_purelib": str(_ISOLATED_PURELIB),
                "site_startup_modules_forbidden": [
                    "site",
                    "sitecustomize",
                    "usercustomize",
                ],
                "automatic_repository_path_disabled": True,
                "environment_configuration_ignored": True,
            },
            "executable": {
                "path": str(executable_presented),
                "resolved_path": str(executable_resolved),
                **executable_receipt,
            },
        },
        "generator_builtins": _generator_builtins_receipt(),
        "executing_generator": {
            **generator_receipt,
            "source_binding_path": "pipeline/generate_dws_single_completion_v1.py",
            "module_object_identity_required": True,
        },
        "generator_live_implementations": _generator_runtime_receipt(),
        "runtime_mutation_boundary": {
            "module_type_module": type(_BOUND_GENERATOR_MODULE).__module__,
            "module_type_qualname": type(_BOUND_GENERATOR_MODULE).__qualname__,
            "protected_global_names": sorted(_PROTECTED_GENERATOR_GLOBAL_NAMES),
            "protected_global_assignment_rejected": True,
            "protected_global_deletion_rejected": True,
            "all_existing_generator_globals_protected": True,
            "production_class_descriptor_assignment_rejected": True,
            "production_class_descriptor_deletion_rejected": True,
            "mutation_guard": _SEALED_MODULE_MUTATION_GUARD,
        },
        "direct_runtime_modules": {
            module_name: _runtime_dependency_receipt(module, module_name)
            for module_name, (module, _) in sorted(_BOUND_RUNTIME_MODULES.items())
        },
        "consumed_runtime_exports": {
            label: (
                _bound_callable_receipt(label, bound_value)
                if callable(bound_value)
                else {
                    "name": label,
                    "value_type_module": type(bound_value).__module__,
                    "value_type_qualname": type(bound_value).__qualname__,
                    "captured_object_identity_required": True,
                }
            )
            for label, (_, _, bound_value, _) in sorted(_BOUND_RUNTIME_EXPORTS.items())
        },
        "consumed_generator_callable_aliases": {
            label: {
                "global_name": global_name,
                **_bound_callable_receipt(label, bound_value),
                "sealed_generator_global_identity_required": True,
            }
            for label, (global_name, bound_value) in sorted(
                _BOUND_GENERATOR_CALLABLE_ALIASES.items()
            )
        },
        "serialization_class_methods": {
            label: _bound_callable_receipt(label, descriptor)
            for label, (_, _, descriptor, _) in sorted(
                _BOUND_SERIALIZATION_CLASS_METHODS.items()
            )
        },
        "packing_semantics": {
            "struct_module": _module_file_receipt(_BOUND_STRUCT_MODULE, "struct"),
            "struct_native_module": _module_file_receipt(
                _BOUND_STRUCT_NATIVE_MODULE, "_struct"
            ),
            "struct_pack": _bound_callable_receipt("struct.pack", _BOUND_STRUCT_PACK),
            "struct_unpack": _bound_callable_receipt(
                "struct.unpack", _BOUND_STRUCT_UNPACK
            ),
            "pack_payload": _bound_callable_receipt(
                "pipeline.generate_dws_single_completion_v1._pack_payload",
                _BOUND_PACK_PAYLOAD,
            ),
            "mutation_guard": _PACK_MUTATION_GUARD,
            "live_exports_must_equal_captured_objects": True,
        },
        "hashing_semantics": {
            "hashlib_module": _module_file_receipt(hashlib, "hashlib"),
            "hashlib_native_module": _module_file_receipt(_hashlib_native, "_hashlib"),
            "sha256": _bound_callable_receipt("hashlib.sha256", _BOUND_HASHLIB_SHA256),
            "live_exports_must_equal_captured_objects": True,
        },
        "reviewed_modules": reviewed_receipts,
        "reviewed_callables": {
            label: _bound_callable_receipt(label, bound_callable)
            for label, (_, _, bound_callable, _) in sorted(
                _BOUND_REVIEWED_FUNCTIONS.items()
            )
        },
        "consumed_reviewed_callables": {
            label: {
                **_bound_callable_receipt(label, bound_callable),
                "private_frozen_globals": True,
                "source_export_mutation_cannot_change_consumed_clone": True,
            }
            for label, bound_callable in sorted(
                _BOUND_CONSUMED_REVIEWED_FUNCTIONS.items()
            )
        },
        "frozen_reviewed_globals": {
            label: {
                "mapping_type_module": type(live_mapping).__module__,
                "mapping_type_qualname": type(live_mapping).__qualname__,
                "entries": len(expected_items),
                "sealed": live_mapping._sealed,
                "exact_key_and_value_identity_required": True,
                "ordinary_mutation_methods_rejected": True,
            }
            for label, live_mapping, expected_items in sorted(
                _BOUND_FROZEN_GLOBAL_MAPPINGS
            )
        },
        "callable_mutation_guards": {
            "complete_generator_runtime": _GENERATOR_RUNTIME_MUTATION_GUARD,
            "reviewed_filesystem": _REVIEWED_FILESYSTEM_MUTATION_GUARD,
            "runtime_exports": _RUNTIME_EXPORT_MUTATION_GUARD,
            "python_random": _RANDOM_MUTATION_GUARD,
            "sys_addaudithook": _bound_callable_receipt(
                "sys.addaudithook", _BOUND_SYS_ADDAUDITHOOK
            ),
            "sys_audit": _bound_callable_receipt("sys.audit", _BOUND_SYS_AUDIT),
            "builtin_import": _bound_callable_receipt(
                "builtins.__import__", _BOUND_BUILTIN_IMPORT
            ),
        },
        "filesystem_semantics": {
            "callables": {
                label: _bound_callable_receipt(label, bound_callable)
                for label, (_, _, bound_callable, _) in sorted(
                    _BOUND_FILESYSTEM_EXPORTS.items()
                )
            },
            "constants": {
                label: bound_value
                for label, (_, _, bound_value) in sorted(
                    _BOUND_FILESYSTEM_CONSTANTS.items()
                )
            },
            "atomic_no_replace": {
                "platform": _BOUND_SYS_PLATFORM,
                "symbol": _BOUND_ATOMIC_RENAME_SYMBOL,
                "flag": _BOUND_ATOMIC_RENAME_FLAG,
                "callable_type_module": _BOUND_ATOMIC_RENAME_TYPE.__module__,
                "callable_type_qualname": _BOUND_ATOMIC_RENAME_TYPE.__qualname__,
                "argument_types": [
                    value.__module__ + "." + value.__qualname__
                    for value in _BOUND_ATOMIC_RENAME_ARGTYPES
                ],
                "result_type": _BOUND_ATOMIC_RENAME_RESTYPE.__module__
                + "."
                + _BOUND_ATOMIC_RENAME_RESTYPE.__qualname__,
                "errcheck": _BOUND_ATOMIC_RENAME_ERRCHECK,
                "errcheck_required_null_at_snapshot": True,
                "errcheck_required_null_immediately_before_call": True,
                "captured_process_image_symbol_required": True,
                "native_implementation_binding": [
                    "Python executable bytes",
                    "_ctypes extension bytes",
                    "captured process-image symbol",
                ],
            },
            "descriptor_relative_cleanup": {
                "symbol": "unlinkat",
                "regular_file_flag": 0,
                "directory_flag": _BOUND_AT_REMOVEDIR,
                "callable_type_module": _BOUND_UNLINKAT_TYPE.__module__,
                "callable_type_qualname": _BOUND_UNLINKAT_TYPE.__qualname__,
                "argument_types": [
                    value.__module__ + "." + value.__qualname__
                    for value in _BOUND_UNLINKAT_ARGTYPES
                ],
                "result_type": _BOUND_UNLINKAT_RESTYPE.__module__
                + "."
                + _BOUND_UNLINKAT_RESTYPE.__qualname__,
                "errcheck": _BOUND_UNLINKAT_ERRCHECK,
                "errcheck_required_null_at_snapshot": True,
                "errcheck_required_null_immediately_before_call": True,
                "captured_process_image_symbol_required": True,
                "python_path_unlink_audit_window_absent": True,
            },
            "captured_callables_are_used_for_all_trust_path_operations": True,
            "live_exports_and_implementations_must_match": True,
            "dispatch_mutation_guard": _FILESYSTEM_DISPATCH_MUTATION_GUARD,
            "fsync_is_captured_in_protected_function_defaults": True,
        },
        "python_random": {
            "class_module": _BOUND_RANDOM_CLASS.__module__,
            "class_qualname": _BOUND_RANDOM_CLASS.__qualname__,
            "native_base_module": _BOUND_RANDOM_NATIVE_CLASS.__module__,
            "native_base_qualname": _BOUND_RANDOM_NATIVE_CLASS.__qualname__,
            "module": _module_file_receipt(_BOUND_RANDOM_MODULE, "random"),
            "native_module": _module_file_receipt(
                _BOUND_RANDOM_NATIVE_MODULE, "_random"
            ),
            "required_methods": {
                name: _bound_descriptor_receipt(owner, name, descriptor)
                for name, (owner, descriptor) in _BOUND_RANDOM_METHODS.items()
            },
            "live_exports_must_equal_bound_objects": True,
        },
        "tokenizers": {
            "distribution_version": distribution_version,
            "module_version": _BOUND_TOKENIZERS_VERSION,
            "tokenizer_class_module": _BOUND_TOKENIZER_CLASS.__module__,
            "tokenizer_class_qualname": _BOUND_TOKENIZER_CLASS.__qualname__,
            "from_str_descriptor_type": type(
                _BOUND_TOKENIZER_FROM_STR_DESCRIPTOR
            ).__name__,
            "consumed_method_descriptors": {
                name: _bound_descriptor_receipt(
                    _BOUND_TOKENIZER_CLASS, name, descriptor
                )
                for name, descriptor in sorted(_BOUND_TOKENIZER_METHODS.items())
            },
            "encoding_class_module": _BOUND_TOKENIZER_ENCODING_CLASS.__module__,
            "encoding_class_qualname": _BOUND_TOKENIZER_ENCODING_CLASS.__qualname__,
            "encoding_ids_descriptor": _bound_descriptor_receipt(
                _BOUND_TOKENIZER_ENCODING_CLASS,
                "ids",
                _BOUND_TOKENIZER_ENCODING_IDS_DESCRIPTOR,
            ),
            "live_exports_must_equal_bound_native_class": True,
            "all_consumed_native_descriptors_are_captured": True,
            "package_module": _module_file_receipt(
                _BOUND_TOKENIZERS_MODULE, "tokenizers"
            ),
            "native_module": _module_file_receipt(
                _BOUND_TOKENIZERS_NATIVE_MODULE, "tokenizers.tokenizers"
            ),
        },
    }


def runtime_bindings_sha256() -> str:
    return hash_json(runtime_bindings())


def _validated_runtime_bindings(
    *,
    expected_runtime_bindings_sha256: str,
    source_receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bindings = runtime_bindings()
    if hash_json(bindings) != expected_runtime_bindings_sha256:
        raise ContractError("externally frozen runtime-binding commitment mismatch")
    source_bound_runtime_receipts = [
        bindings["executing_generator"],
        *bindings["reviewed_modules"].values(),
    ]
    for receipt in source_bound_runtime_receipts:
        source_path = receipt["source_binding_path"]
        source_receipt = source_receipts.get(source_path)
        if source_receipt != {
            "bytes": receipt["bytes"],
            "sha256": receipt["sha256"],
        }:
            raise ContractError(
                "executed module bytes differ from reviewed source binding: "
                + source_path
            )
    return bindings


def _assert_runtime_snapshot(
    expected_bindings: dict[str, Any],
    phase: str,
    _runtime_bindings: Any = runtime_bindings,
) -> None:
    live_bindings = _runtime_bindings()
    if not _recursively_type_strict_equal(live_bindings, expected_bindings):
        raise ContractError("runtime bindings changed " + phase)


_RUNTIME_PHASE_AUDIT_EVENT = "shohin.dws_single_completion.runtime_phase"


def _emit_runtime_phase(
    phase: str,
    out_dir: Path,
    _audit: Any = _BOUND_SYS_AUDIT,
    _event: str = _RUNTIME_PHASE_AUDIT_EVENT,
) -> None:
    _audit(_event, phase, str(out_dir))


@dataclass(frozen=True)
class _PinnedDirectory(metaclass=_SealedRuntimeClassMeta):
    _runtime_descriptors_sealed = False

    path: Path
    descriptors: tuple[int, ...]
    ancestors: tuple[tuple[str, int, int], ...]

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    @property
    def device(self) -> int:
        return self.ancestors[-1][1]

    @property
    def inode(self) -> int:
        return self.ancestors[-1][2]

    def identity_receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "ancestors": [
                {"path": path, "device": device, "inode": inode}
                for path, device, inode in self.ancestors
            ],
        }

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            _BOUND_OS_CLOSE(descriptor)


def _directory_open_flags() -> int:
    return _BOUND_OS_O_RDONLY | _BOUND_OS_O_DIRECTORY | _BOUND_OS_O_NOFOLLOW


def _walk_directory_chain(path: Path, *, create: bool) -> _PinnedDirectory:
    absolute = Path(_BOUND_OS_PATH_ABSPATH(_BOUND_OS_FSPATH(path)))
    if not absolute.is_absolute():
        raise ContractError("publication parent path must be absolute")
    flags = _directory_open_flags()
    try:
        root_descriptor = _BOUND_OS_OPEN(_BOUND_OS_SEP, flags)
    except OSError as error:
        raise ContractError("filesystem root cannot be pinned") from error
    descriptors = [root_descriptor]
    root_metadata = _BOUND_OS_FSTAT(root_descriptor)
    ancestors: list[tuple[str, int, int]] = [
        (_BOUND_OS_SEP, root_metadata.st_dev, root_metadata.st_ino)
    ]
    current = Path(_BOUND_OS_SEP)
    try:
        for component in absolute.parts[1:]:
            if component in ("", ".", "..") or _BOUND_OS_SEP in component:
                raise ContractError("publication parent contains an invalid component")
            try:
                child = _BOUND_OS_OPEN(component, flags, dir_fd=descriptors[-1])
            except FileNotFoundError:
                if not create:
                    raise ContractError(
                        "publication parent ancestor is missing: {}".format(
                            current / component
                        )
                    ) from None
                try:
                    _BOUND_OS_MKDIR(component, mode=0o700, dir_fd=descriptors[-1])
                    child = _BOUND_OS_OPEN(component, flags, dir_fd=descriptors[-1])
                except OSError as error:
                    raise ContractError(
                        "publication parent ancestor cannot be created safely: {}".format(
                            current / component
                        )
                    ) from error
            except OSError as error:
                raise ContractError(
                    "publication ancestor must be a non-symlink directory: {}".format(
                        current / component
                    )
                ) from error
            descriptors.append(child)
            current /= component
            metadata = _BOUND_OS_FSTAT(child)
            if not _BOUND_STAT_S_ISDIR(metadata.st_mode):
                raise ContractError("publication ancestor is not a directory")
            ancestors.append((str(current), metadata.st_dev, metadata.st_ino))
        return _PinnedDirectory(absolute, tuple(descriptors), tuple(ancestors))
    except BaseException:
        for descriptor in reversed(descriptors):
            _BOUND_OS_CLOSE(descriptor)
        raise


def _pin_publication_parent(path: Path, *, create: bool) -> _PinnedDirectory:
    pinned = _walk_directory_chain(path, create=create)
    metadata = _BOUND_OS_FSTAT(pinned.descriptor)
    if metadata.st_uid != _BOUND_OS_GETEUID():
        pinned.close()
        raise ContractError("publication parent must be owned by this user")
    if _BOUND_STAT_S_IMODE(metadata.st_mode) & 0o022:
        pinned.close()
        raise ContractError("publication parent must not be group/world writable")
    return pinned


def _assert_pinned_directory(pinned: _PinnedDirectory) -> None:
    if len(pinned.descriptors) != len(pinned.ancestors):
        raise ContractError("pinned publication ancestor descriptor count changed")
    for descriptor, (_, expected_device, expected_inode) in zip(
        pinned.descriptors, pinned.ancestors, strict=True
    ):
        metadata = _BOUND_OS_FSTAT(descriptor)
        if (metadata.st_dev, metadata.st_ino) != (
            expected_device,
            expected_inode,
        ):
            raise ContractError("pinned publication ancestor identity changed")
    current = _walk_directory_chain(pinned.path, create=False)
    try:
        if current.ancestors != pinned.ancestors:
            raise ContractError("publication ancestor identity was retargeted")
    finally:
        current.close()


def _validate_entry_name(name: str) -> None:
    if name in ("", ".", "..") or _BOUND_OS_SEP in name:
        raise ContractError("unsafe directory entry name")


def _entry_metadata(directory_fd: int, name: str) -> os.stat_result | None:
    _validate_entry_name(name)
    try:
        return _BOUND_OS_STAT(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ContractError(
            "directory entry cannot be inspected safely: " + name
        ) from error


def _open_directory_at(directory_fd: int, name: str, label: str) -> int:
    _validate_entry_name(name)
    try:
        descriptor = _BOUND_OS_OPEN(name, _directory_open_flags(), dir_fd=directory_fd)
    except OSError as error:
        raise ContractError("{} cannot be opened safely".format(label)) from error
    if not _BOUND_STAT_S_ISDIR(_BOUND_OS_FSTAT(descriptor).st_mode):
        _BOUND_OS_CLOSE(descriptor)
        raise ContractError("{} must be a directory".format(label))
    return descriptor


def _assert_regular_file_entry_identity(
    directory_fd: int, name: str, descriptor: int, label: str
) -> os.stat_result:
    descriptor_metadata = _BOUND_OS_FSTAT(descriptor)
    entry_metadata = _entry_metadata(directory_fd, name)
    if (
        entry_metadata is None
        or not _BOUND_STAT_S_ISREG(entry_metadata.st_mode)
        or (entry_metadata.st_dev, entry_metadata.st_ino)
        != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    ):
        raise ContractError(label + " path is not the held regular file descriptor")
    return descriptor_metadata


def _stable_file_descriptor_bytes(
    descriptor: int,
    label: str,
    *,
    directory_fd: int,
    name: str,
    exact_mode: int | None = None,
    require_nlink_one: bool = False,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    before = _assert_regular_file_entry_identity(directory_fd, name, descriptor, label)
    if not _BOUND_STAT_S_ISREG(before.st_mode):
        raise ContractError("{} must be a regular file".format(label))
    if expected_identity is not None and (before.st_dev, before.st_ino) != (
        expected_identity
    ):
        raise ContractError("{} descriptor identity changed".format(label))
    if require_nlink_one and before.st_nlink != 1:
        raise ContractError("{} must have exactly one hard link".format(label))
    if exact_mode is not None and _BOUND_STAT_S_IMODE(before.st_mode) != exact_mode:
        raise ContractError("{} mode must be {:04o}".format(label, exact_mode))
    blocks = []
    while True:
        block = _BOUND_OS_READ(descriptor, 1024 * 1024)
        if not block:
            break
        blocks.append(block)
    payload = b"".join(blocks)
    after = _assert_regular_file_entry_identity(directory_fd, name, descriptor, label)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ContractError("{} changed during its sealed read".format(label))
    return payload, {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def _stable_file_bytes_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    exact_mode: int | None = None,
    require_nlink_one: bool = False,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    _validate_entry_name(name)
    flags = _BOUND_OS_O_RDONLY | _BOUND_OS_O_NOFOLLOW
    try:
        descriptor = _BOUND_OS_OPEN(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ContractError("{} cannot be opened safely".format(label)) from error
    try:
        return _stable_file_descriptor_bytes(
            descriptor,
            label,
            directory_fd=directory_fd,
            name=name,
            exact_mode=exact_mode,
            require_nlink_one=require_nlink_one,
            expected_identity=expected_identity,
        )
    finally:
        _BOUND_OS_CLOSE(descriptor)


def _write_sealed_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    _open: Any = _BOUND_OS_OPEN,
    _write: Any = _BOUND_OS_WRITE,
    _fsync: Any = _BOUND_OS_FSYNC,
    _fchmod: Any = _BOUND_OS_FCHMOD,
    _close: Any = _BOUND_OS_CLOSE,
    _flags: int = (
        _BOUND_OS_O_WRONLY | _BOUND_OS_O_CREAT | _BOUND_OS_O_EXCL | _BOUND_OS_O_NOFOLLOW
    ),
) -> None:
    _validate_entry_name(name)
    try:
        descriptor = _open(name, _flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise ContractError("sealed file cannot be created safely: " + name) from error
    try:
        offset = 0
        while offset < len(payload):
            written = _write(descriptor, payload[offset:])
            if written <= 0:
                raise ContractError("sealed file write made no progress: " + name)
            offset += written
        _fsync(descriptor)
        _fchmod(descriptor, 0o444)
        _fsync(descriptor)
        sealed_metadata = _BOUND_OS_FSTAT(descriptor)
        sealed_identity = (sealed_metadata.st_dev, sealed_metadata.st_ino)
        if (
            not _BOUND_STAT_S_ISREG(sealed_metadata.st_mode)
            or sealed_metadata.st_nlink != 1
            or _BOUND_STAT_S_IMODE(sealed_metadata.st_mode) != 0o444
            or sealed_metadata.st_size != len(payload)
        ):
            raise ContractError("sealed file descriptor validation failed: " + name)
        _assert_regular_file_entry_identity(
            directory_fd, name, descriptor, "newly sealed file"
        )
        readback, _ = _stable_file_bytes_at(
            directory_fd,
            name,
            "newly sealed file",
            exact_mode=0o444,
            require_nlink_one=True,
            expected_identity=sealed_identity,
        )
        if readback != payload:
            raise ContractError("newly sealed file bytes differ from payload: " + name)
        final_metadata = _assert_regular_file_entry_identity(
            directory_fd, name, descriptor, "newly sealed file"
        )
        if (final_metadata.st_dev, final_metadata.st_ino) != sealed_identity:
            raise ContractError(
                "newly sealed file descriptor identity changed: " + name
            )
    finally:
        _close(descriptor)


def _fsync_directory_fd(directory_fd: int, _fsync: Any = _BOUND_OS_FSYNC) -> None:
    _fsync(directory_fd)


_FILESYSTEM_DISPATCH_MUTATION_GUARD = _install_callable_mutation_guard(
    "filesystem-dispatch-v1", (_write_sealed_file_at, _fsync_directory_fd)
)


def _partial_stage_path(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    return out_dir.parent / ".{}.partial".format(out_dir.name)


def _staging_identity(
    *,
    out_dir: Path,
    pinned_parent: _PinnedDirectory,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
) -> dict[str, Any]:
    out_dir = Path(out_dir).absolute()
    if out_dir.parent != pinned_parent.path:
        raise ContractError("publication target does not use the pinned parent")
    return {
        "schema": "shohin-dws-single-completion-staging-identity-v2",
        "protocol": PROTOCOL,
        "target_name": out_dir.name,
        "staging_name": _partial_stage_path(out_dir).name,
        "pinned_parent": pinned_parent.identity_receipt(),
        "mode": mode,
        "generation_seed": seed,
        "train_per_cell": train_per_cell,
        "development_per_cell": development_per_cell,
        "lane_length": lane_length,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "expected_replication_source_sha256": expected_replication_source_sha256,
        "expected_source_bindings_sha256": expected_source_bindings_sha256,
        "expected_runtime_bindings_sha256": expected_runtime_bindings_sha256,
        "input_location_policy": _BOUND_INPUT_LOCATION_POLICY,
    }


def _lock_stage_directory(parent_fd: int, stage_name: str) -> int:
    descriptor = _open_directory_at(parent_fd, stage_name, "staging directory")
    try:
        _BOUND_FCNTL_FLOCK(descriptor, _BOUND_FCNTL_LOCK_EX | _BOUND_FCNTL_LOCK_NB)
    except BlockingIOError as error:
        _BOUND_OS_CLOSE(descriptor)
        raise ContractError("staging directory belongs to a live invocation") from error
    except OSError:
        _BOUND_OS_CLOSE(descriptor)
        raise
    return descriptor


def _lock_existing_publication(parent_fd: int, publication_name: str) -> int:
    descriptor = _open_directory_at(
        parent_fd, publication_name, "existing publication directory"
    )
    try:
        _BOUND_FCNTL_FLOCK(descriptor, _BOUND_FCNTL_LOCK_EX | _BOUND_FCNTL_LOCK_NB)
    except BlockingIOError as error:
        _BOUND_OS_CLOSE(descriptor)
        raise ContractError(
            "publication target belongs to a live invocation"
        ) from error
    except OSError:
        _BOUND_OS_CLOSE(descriptor)
        raise
    return descriptor


def _directory_descriptor_identity(descriptor: int, label: str) -> tuple[int, int]:
    metadata = _BOUND_OS_FSTAT(descriptor)
    if not _BOUND_STAT_S_ISDIR(metadata.st_mode):
        raise ContractError(label + " descriptor is not a directory")
    return metadata.st_dev, metadata.st_ino


def _assert_directory_entry_identity(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int],
    label: str,
) -> None:
    metadata = _entry_metadata(parent_fd, name)
    if metadata is None or not _BOUND_STAT_S_ISDIR(metadata.st_mode):
        raise ContractError(label + " path is not the locked directory")
    if (metadata.st_dev, metadata.st_ino) != expected_identity:
        raise ContractError(label + " path identity changed")


def _directory_entry_has_identity(
    parent_fd: int, name: str, expected_identity: tuple[int, int]
) -> bool:
    metadata = _entry_metadata(parent_fd, name)
    return bool(
        metadata is not None
        and _BOUND_STAT_S_ISDIR(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == expected_identity
    )


def _open_validated_staging_member_at(
    directory_fd: int, name: str, label: str
) -> tuple[int, os.stat_result]:
    _validate_entry_name(name)
    try:
        descriptor = _BOUND_OS_OPEN(
            name,
            _BOUND_OS_O_RDONLY | _BOUND_OS_O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ContractError("{} cannot be opened safely".format(label)) from error
    try:
        metadata = _assert_regular_file_entry_identity(
            directory_fd, name, descriptor, label
        )
        if metadata.st_uid != _BOUND_OS_GETEUID():
            raise ContractError("{} is not owned by this user".format(label))
        if metadata.st_nlink != 1:
            raise ContractError("{} must have exactly one hard link".format(label))
        if _BOUND_STAT_S_IMODE(metadata.st_mode) not in (0o600, 0o444):
            raise ContractError("{} has an invalid staging mode".format(label))
        return descriptor, metadata
    except BaseException:
        _BOUND_OS_CLOSE(descriptor)
        raise


def _open_validated_protocol_staging_tree(
    stage_fd: int, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    opened_descriptors: list[int] = []
    stage_metadata = _BOUND_OS_FSTAT(stage_fd)
    if not _BOUND_STAT_S_ISDIR(stage_metadata.st_mode):
        raise ContractError("staging tree must be a regular directory")
    if stage_metadata.st_uid != _BOUND_OS_GETEUID():
        raise ContractError("staging tree is not owned by this user")
    if _BOUND_STAT_S_IMODE(stage_metadata.st_mode) not in (0o700, 0o555):
        raise ContractError("staging tree has an invalid mode")
    inventory = set(_BOUND_OS_LISTDIR(stage_fd))
    allowed_root_entries = {
        SEALED_ROOT_NAME,
        STAGING_NEXT_MANIFEST_NAME,
        BUNDLE_DIRECTORY_NAME,
    }
    if SEALED_ROOT_NAME not in inventory or not inventory <= allowed_root_entries:
        raise ContractError("staging tree inventory is not protocol-owned")
    try:
        root_descriptor, _ = _open_validated_staging_member_at(
            stage_fd, SEALED_ROOT_NAME, "staging identity manifest"
        )
        opened_descriptors.append(root_descriptor)
        manifest_payload, _ = _stable_file_descriptor_bytes(
            root_descriptor,
            "staging identity manifest",
            directory_fd=stage_fd,
            name=SEALED_ROOT_NAME,
            require_nlink_one=True,
        )
        manifest = strict_json_loads(manifest_payload, "staging identity manifest")
        completed_root = False
        if manifest.get("schema") == STAGING_OWNER_SCHEMA:
            expected_marker = {
                "schema": STAGING_OWNER_SCHEMA,
                "identity": expected_identity,
            }
            if not _recursively_type_strict_equal(manifest, expected_marker):
                raise ContractError("staging owner marker identity mismatch")
        elif manifest.get("schema") == "shohin-dws-single-completion-sealed-root-v1":
            completed_root = True
            actual_identity = manifest.get("publication_layout", {}).get(
                "staging_identity"
            )
            if not _recursively_type_strict_equal(actual_identity, expected_identity):
                raise ContractError("completed staging root identity mismatch")
            if inventory != {SEALED_ROOT_NAME, BUNDLE_DIRECTORY_NAME}:
                raise ContractError("completed staging inventory is not closed")
        else:
            raise ContractError("staging tree lacks a valid protocol owner identity")

        next_descriptor = None
        if STAGING_NEXT_MANIFEST_NAME in inventory:
            next_descriptor, _ = _open_validated_staging_member_at(
                stage_fd, STAGING_NEXT_MANIFEST_NAME, "next staging manifest"
            )
            opened_descriptors.append(next_descriptor)

        bundle_descriptor = None
        artifact_descriptors: list[tuple[str, int]] = []
        if BUNDLE_DIRECTORY_NAME in inventory:
            bundle_descriptor = _open_directory_at(
                stage_fd, BUNDLE_DIRECTORY_NAME, "staging bundle"
            )
            opened_descriptors.append(bundle_descriptor)
            bundle_identity = _directory_descriptor_identity(
                bundle_descriptor, "staging bundle"
            )
            _assert_directory_entry_identity(
                stage_fd,
                BUNDLE_DIRECTORY_NAME,
                bundle_identity,
                "staging bundle",
            )
            bundle_metadata = _BOUND_OS_FSTAT(bundle_descriptor)
            if bundle_metadata.st_uid != _BOUND_OS_GETEUID():
                raise ContractError("staging bundle is not owned by this user")
            if _BOUND_STAT_S_IMODE(bundle_metadata.st_mode) not in (0o700, 0o555):
                raise ContractError("staging bundle has an invalid mode")
            bundle_inventory = set(_BOUND_OS_LISTDIR(bundle_descriptor))
            if not bundle_inventory <= set(ARTIFACT_NAMES):
                raise ContractError("staging bundle inventory is not protocol-owned")
            if completed_root and bundle_inventory != set(ARTIFACT_NAMES):
                raise ContractError("completed staging bundle inventory is not closed")
            for artifact_name in sorted(bundle_inventory):
                artifact_descriptor, metadata = _open_validated_staging_member_at(
                    bundle_descriptor,
                    artifact_name,
                    "staging artifact {}".format(artifact_name),
                )
                opened_descriptors.append(artifact_descriptor)
                artifact_descriptors.append((artifact_name, artifact_descriptor))
                if completed_root and _BOUND_STAT_S_IMODE(metadata.st_mode) != 0o444:
                    raise ContractError("completed staging artifact is not sealed")
        return {
            "root_descriptor": root_descriptor,
            "next_descriptor": next_descriptor,
            "bundle_descriptor": bundle_descriptor,
            "artifact_descriptors": tuple(artifact_descriptors),
            "opened_descriptors": tuple(opened_descriptors),
        }
    except BaseException:
        for opened_descriptor in reversed(opened_descriptors):
            _BOUND_OS_CLOSE(opened_descriptor)
        raise


def _close_validated_protocol_staging_tree(tree: dict[str, Any]) -> None:
    for descriptor in reversed(tree["opened_descriptors"]):
        _BOUND_OS_CLOSE(descriptor)


def _validate_protocol_staging_tree(
    stage_fd: int, expected_identity: dict[str, Any]
) -> None:
    tree = _open_validated_protocol_staging_tree(stage_fd, expected_identity)
    _close_validated_protocol_staging_tree(tree)


def _chmod_directory_fd(descriptor: int, label: str, mode: int) -> None:
    before = _BOUND_OS_FSTAT(descriptor)
    if not _BOUND_STAT_S_ISDIR(before.st_mode):
        raise ContractError(label + " identity changed before chmod")
    _BOUND_OS_FCHMOD(descriptor, mode)
    after = _BOUND_OS_FSTAT(descriptor)
    if (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or not _BOUND_STAT_S_ISDIR(after.st_mode)
        or _BOUND_STAT_S_IMODE(after.st_mode) != mode
    ):
        raise ContractError(label + " chmod did not take effect")


def _native_unlinkat(
    directory_fd: int,
    name: str,
    *,
    remove_directory: bool,
    _unlinkat: Any = _BOUND_UNLINKAT,
    _fsencode: Any = _BOUND_OS_FSENCODE,
    _get_errno: Any = _BOUND_CTYPES_GET_ERRNO,
    _strerror: Any = _BOUND_OS_STRERROR,
    _at_removedir: int = _BOUND_AT_REMOVEDIR,
) -> None:
    _validate_entry_name(name)
    encoded_name = _fsencode(name)
    flags = _at_removedir if remove_directory else 0
    if _unlinkat.errcheck is not None:
        raise ContractError("descriptor-relative cleanup errcheck must remain null")
    result = _unlinkat(
        directory_fd,
        encoded_name,
        flags,
    )
    if result != 0:
        error_number = _get_errno()
        raise ContractError(
            "descriptor-relative cleanup failed for {}: {}".format(
                name, _strerror(error_number)
            )
        )


def _remove_held_regular_file_at(
    directory_fd: int, name: str, descriptor: int, label: str
) -> None:
    before = _assert_regular_file_entry_identity(directory_fd, name, descriptor, label)
    if not _BOUND_STAT_S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ContractError(label + " descriptor identity changed before cleanup")
    _BOUND_OS_FCHMOD(descriptor, 0o600)
    after = _assert_regular_file_entry_identity(directory_fd, name, descriptor, label)
    if (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or not _BOUND_STAT_S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or _BOUND_STAT_S_IMODE(after.st_mode) != 0o600
    ):
        raise ContractError(label + " descriptor changed during cleanup chmod")
    _native_unlinkat(directory_fd, name, remove_directory=False)
    unlinked = _BOUND_OS_FSTAT(descriptor)
    if (
        (unlinked.st_dev, unlinked.st_ino) != (before.st_dev, before.st_ino)
        or not _BOUND_STAT_S_ISREG(unlinked.st_mode)
        or unlinked.st_nlink != 0
        or _entry_metadata(directory_fd, name) is not None
    ):
        raise ContractError(label + " unlink did not remove the held descriptor")


def _remove_held_directory_at(
    parent_fd: int, name: str, descriptor: int, label: str
) -> None:
    identity = _directory_descriptor_identity(descriptor, label)
    _assert_directory_entry_identity(parent_fd, name, identity, label)
    _native_unlinkat(parent_fd, name, remove_directory=True)
    after = _BOUND_OS_FSTAT(descriptor)
    if (
        (after.st_dev, after.st_ino) != identity
        or not _BOUND_STAT_S_ISDIR(after.st_mode)
        or _entry_metadata(parent_fd, name) is not None
    ):
        raise ContractError(label + " removal did not retain descriptor identity")


def _remove_protocol_staging_tree(
    pinned_parent: _PinnedDirectory,
    stage_name: str,
    expected_identity: dict[str, Any],
    *,
    locked_descriptor: int | None = None,
) -> None:
    owns_descriptor = locked_descriptor is None
    descriptor = (
        _lock_stage_directory(pinned_parent.descriptor, stage_name)
        if locked_descriptor is None
        else locked_descriptor
    )
    locked_identity = _directory_descriptor_identity(
        descriptor, "protocol staging cleanup"
    )
    try:
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            stage_name,
            locked_identity,
            "protocol staging cleanup",
        )
        tree = _open_validated_protocol_staging_tree(descriptor, expected_identity)
        try:
            _chmod_directory_fd(descriptor, "protocol staging cleanup", 0o700)
            next_descriptor = tree["next_descriptor"]
            if next_descriptor is not None:
                _remove_held_regular_file_at(
                    descriptor,
                    STAGING_NEXT_MANIFEST_NAME,
                    next_descriptor,
                    "next staging manifest cleanup",
                )
            bundle_descriptor = tree["bundle_descriptor"]
            if bundle_descriptor is not None:
                _chmod_directory_fd(bundle_descriptor, "staging bundle cleanup", 0o700)
                for artifact_name, artifact_descriptor in tree["artifact_descriptors"]:
                    _remove_held_regular_file_at(
                        bundle_descriptor,
                        artifact_name,
                        artifact_descriptor,
                        "staging artifact cleanup " + artifact_name,
                    )
                _fsync_directory_fd(bundle_descriptor)
                _remove_held_directory_at(
                    descriptor,
                    BUNDLE_DIRECTORY_NAME,
                    bundle_descriptor,
                    "staging bundle cleanup",
                )
            _remove_held_regular_file_at(
                descriptor,
                SEALED_ROOT_NAME,
                tree["root_descriptor"],
                "staging identity manifest cleanup",
            )
            _fsync_directory_fd(descriptor)
            _assert_directory_entry_identity(
                pinned_parent.descriptor,
                stage_name,
                locked_identity,
                "protocol staging cleanup",
            )
            _remove_held_directory_at(
                pinned_parent.descriptor,
                stage_name,
                descriptor,
                "protocol staging cleanup",
            )
            _fsync_directory_fd(pinned_parent.descriptor)
        finally:
            _close_validated_protocol_staging_tree(tree)
    finally:
        if owns_descriptor:
            _BOUND_OS_CLOSE(descriptor)


def _recover_protocol_staging_tree(
    pinned_parent: _PinnedDirectory,
    stage_name: str,
    expected_identity: dict[str, Any],
) -> None:
    if _entry_metadata(pinned_parent.descriptor, stage_name) is None:
        return
    _remove_protocol_staging_tree(pinned_parent, stage_name, expected_identity)


def _payload_receipts(payloads: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    if set(payloads) != set(ARTIFACT_NAMES):
        raise ContractError("artifact payload inventory mismatch")
    return {
        name: {"bytes": len(payloads[name]), "sha256": sha256_bytes(payloads[name])}
        for name in ARTIFACT_NAMES
    }


def _atomic_rename_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    locked_source_identity: tuple[int, int],
    _fstat: Any = _BOUND_OS_FSTAT,
    _fsencode: Any = _BOUND_OS_FSENCODE,
    _isdir: Any = _BOUND_STAT_S_ISDIR,
    _rename: Any = _BOUND_ATOMIC_RENAME,
    _rename_flag: int = _BOUND_ATOMIC_RENAME_FLAG,
    _get_errno: Any = _BOUND_CTYPES_GET_ERRNO,
    _collision_errors: tuple[int, int] = (
        _BOUND_ERRNO_EEXIST,
        _BOUND_ERRNO_ENOTEMPTY,
    ),
    _strerror: Any = _BOUND_OS_STRERROR,
    _metadata: Any = _entry_metadata,
    _validate: Any = _validate_entry_name,
    _error: Any = ContractError,
) -> None:
    _validate(source_name)
    _validate(destination_name)
    source_metadata = _metadata(parent_fd, source_name)
    if source_metadata is None or not _isdir(source_metadata.st_mode):
        raise _error("atomic publication source is not a directory")
    if (source_metadata.st_dev, source_metadata.st_ino) != locked_source_identity:
        raise _error("atomic publication source path identity changed")
    if source_metadata.st_dev != _fstat(parent_fd).st_dev:
        raise _error("atomic publication must remain on one filesystem")
    source_bytes = _fsencode(source_name)
    destination_bytes = _fsencode(destination_name)
    if _rename.errcheck is not None:
        raise _error("atomic no-replace errcheck must remain null")
    result = _rename(
        parent_fd,
        source_bytes,
        parent_fd,
        destination_bytes,
        _rename_flag,
    )
    if result != 0:
        error_number = _get_errno()
        if error_number in _collision_errors:
            raise _error("refusing to overwrite existing publication")
        raise _error(
            "atomic no-overwrite publication failed: {}".format(_strerror(error_number))
        )


_BOUND_ATOMIC_RENAME_NOREPLACE_AT = _atomic_rename_noreplace_at


def _inventory_receipt(inventory: dict[str, set[Any]]) -> dict[str, Any]:
    return {
        field: {
            "count": len(values),
            "sha256": hash_json(sorted(values)),
        }
        for field, values in inventory.items()
    }


def _aggregate_episode_inventory(
    episodes: Iterable[dict[str, Any]],
) -> dict[str, set[Any]]:
    inventory = empty_overlap_inventory()
    for episode in episodes:
        merge_overlap_inventory(inventory, episode_overlap_inventory(episode))
    return inventory


def _validate_generation_contract(
    *,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
) -> None:
    if mode not in ("production", "test"):
        raise ContractError("mode must be production or test")
    for label, value in (
        ("generation seed", seed),
        ("train per-cell count", train_per_cell),
        ("development per-cell count", development_per_cell),
        ("lane length", lane_length),
    ):
        if type(value) is not int:
            raise ContractError(label + " must be an exact integer")
    for label, value in (
        ("tokenizer", expected_tokenizer_sha256),
        ("parent checkpoint", parent_checkpoint_sha256),
        ("replication source", expected_replication_source_sha256),
        ("source bindings", expected_source_bindings_sha256),
        ("runtime bindings", expected_runtime_bindings_sha256),
    ):
        if not HEX64_RE.fullmatch(value):
            raise ContractError("expected {} SHA-256 is malformed".format(label))
    if source_bindings_sha256() != expected_source_bindings_sha256:
        raise ContractError("externally frozen source-binding commitment mismatch")
    if mode == "production":
        if (
            seed != GENERATION_SEED
            or train_per_cell != TRAIN_PER_CELL
            or development_per_cell != DEVELOPMENT_PER_CELL
            or lane_length != LANE_LENGTH
            or expected_tokenizer_sha256 != KNOWN_TOKENIZER_SHA256
            or expected_replication_source_sha256 != KNOWN_REPLICATION_SOURCE_SHA256
        ):
            raise ContractError("production constants are immutable")
    elif train_per_cell < 6 or development_per_cell <= 0:
        raise ContractError("test mode needs at least six train episodes per cell")
    if lane_length <= 0 or lane_length > 65_536:
        raise ContractError("lane length must fit dense uint16 position IDs")
    packs = len(OPERATIONS) * len(INTERMEDIATE_PATTERNS) * train_per_cell
    if packs % PACKS_PER_UPDATE:
        raise ContractError("logical pack count must divide exact update size")


def _decision_contract() -> dict[str, Any]:
    primary_contrasts = [
        "full_trace_vs_decomposed__full_history",
        "full_trace_vs_sham__full_history",
        "full_trace_vs_decomposed__commit_reencode",
        "full_trace_vs_sham__commit_reencode",
        "commit_reencode_package_vs_full_history__full_trace",
    ]
    return {
        "seed_rule": {
            "directional_success_required": "3/3 seeds for every primary contrast",
            "per_seed_effect_min": 0.10,
            "per_seed_gates_noncompensatory": True,
            "pooled_favorable_selection_forbidden": True,
            "failed_seed_may_not_be_rescued_by_pooling": True,
        },
        "primary_contrasts": primary_contrasts,
        "reporting": {
            "per_seed": (
                "paired counts, exact McNemar p, effect, 10000 source-episode "
                "cluster-bootstrap replicates, and 95% interval"
            ),
            "seed_level": "all three effects, minimum, median, and range",
            "pooled": "descriptive only; cluster first by seed then source episode",
            "partial_scores_before_all_cells_immutable": False,
        },
        "multiplicity": {
            "family": "five primary contrasts x three frozen seeds = 15 tests",
            "method": "Holm-Bonferroni",
            "familywise_alpha": 0.05,
            "ordering": "ascending exact two-sided McNemar p with contrast then seed tie-break",
            "success": "all 15 adjusted decisions pass in the preregistered direction",
            "veto_gates": (
                "EOS, first-state regression, carry target-switch, and width-separated "
                "SCERT replication gates are noncompensatory and not pooled"
            ),
        },
    }


def _training_plan(
    *,
    pack_count: int,
    lane_length: int,
    supervised_tokens: int,
    active_tokens: int,
    seed_schedules_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "shohin-dws-single-completion-training-plan-v2",
        "protocol": PROTOCOL,
        "status": "cpu_data_contract_only_no_training_authority",
        "run_cells": list(RUN_CELLS),
        "data_arms": list(DATA_ARMS),
        "context_arms": {
            "full_history_replay_discard": {
                "training_context": "ordinary full-history causal attention",
                "decode_context": "ordinary full-history causal KV",
                "commit_action": (
                    "execute the same LF-triggered latest-epoch re-encode and discard "
                    "the fresh representation"
                ),
            },
            "commit_reencode_isolation": {
                "training_context": (
                    "replace stale context with a fresh latest-state representation "
                    "after every supervised LF"
                ),
                "decode_context": (
                    "replace stale KV after every model-emitted LF with fresh "
                    "latest-state KV"
                ),
            },
        },
        "scored_mechanism_name": "model-triggered external commit/re-encode runtime",
        "package_effect_only": {
            "allowed_label": "complete commit-reencode package context effect",
            "stale_source_specific_attribution": False,
            "component_attribution_deferred_to": (
                "separate SCERT extra-depth/no-retirement, mask-only, contaminated-replay, "
                "and fresh-host-prompt controls"
            ),
            "autonomous_base_model_reasoning_claim": False,
        },
        "implementation_boundary": {
            "trainer_consumption_present": False,
            "evaluator_scoring_present": False,
            "h100_authorized": False,
            "future_evaluator_rational_comparison": (
                "integer cross multiplication only; binary floating point forbidden"
            ),
        },
        "metadata_serialization": {
            "audit_only_fields_include": [
                "treatment_answer",
                "line_donor_episode_ids",
                "answer_donor_episode_id",
                "training_group",
            ],
            "serialized_into_binary_packs": False,
            "future_trainer_must_prove_metadata_exclusion": True,
        },
        "application_phases": {
            "teacher_forced_training": True,
            "autonomous_decode": True,
            "separate_fit_per_run_cell_and_seed": True,
            "same_weight_decode_ablation_required": True,
        },
        "delimiter": {
            "text": "LF",
            "activation_boundary": "model token ID after prompt prefill",
            "semantic_parser": False,
        },
        "resource_boundary": (
            "An external runtime compares generated IDs with one frozen LF token, tracks "
            "token spans, and performs re-encode forwards. It performs no DWS parsing, "
            "arithmetic, validation, repair, gold injection, schedule selection, or retry."
        ),
        "block_diagonal_context_filler": {
            "lane_roles": list(LANE_ROLES),
            "independent_batch_lanes": True,
            "cross_lane_attention": False,
            "all_arms_token_ids_identical": True,
            "all_arms_attention_masks_identical": True,
            "all_arms_state_epoch_ids_identical": True,
            "all_arms_position_ids_identical": True,
            "non_arm_blocks_loss_masked": True,
            "filler_can_affect_supervised_lane": False,
        },
        "equalization": {
            "parameters": "identical count and initialization within paired seed",
            "supervised_tokens_per_arm": supervised_tokens,
            "active_context_tokens_per_arm": active_tokens,
            "logical_packs_per_arm": pack_count,
            "physical_lanes_per_pack": LANES_PER_PACK,
            "lane_length": lane_length,
            "fixed_forward_positions_per_pack": LANES_PER_PACK * lane_length,
            "position_ids": {
                "normative_per_lane": "range(lane_length)",
                "serialized": True,
                "integer_encoding": "little-endian uint16",
                "padding_restarts_positions": False,
            },
            "binary_pack_layout": [
                "token_ids:little-endian uint32",
                "attention_mask:uint8",
                "loss_mask:uint8",
                "epoch_ids:little-endian uint16",
                "position_ids:little-endian uint16",
            ],
            "bytes_per_lane_position": PACK_ELEMENT_BYTES,
            "packs_per_update": PACKS_PER_UPDATE,
            "updates_per_arm": pack_count // PACKS_PER_UPDATE,
            "training_reencode_forwards": "exactly four per logical pack in every cell",
            "unpadding": "forbidden",
            "variable_length_kernel": "forbidden",
        },
        "optimizer_contract": OPTIMIZER_CONTRACT,
        "optimizer_contract_sha256": hash_json(OPTIMIZER_CONTRACT),
        "seed_schedules_sha256": seed_schedules_sha256,
        "decision_contract": _decision_contract(),
        "posthoc_kv_slicing": {
            "authorized_as_treatment": False,
            "negative_control": CACHE_PRUNING_PRIOR,
        },
        "authorization": {
            "gpu_job": False,
            "training": False,
            "production_corpus": False,
            "promotion": False,
            "capability_claim": False,
        },
    }


def _lane_receipt(lane: dict[str, list[int]]) -> dict[str, Any]:
    active_ids = [
        token_id
        for token_id, active in zip(
            lane["token_ids"], lane["attention_mask"], strict=True
        )
        if active
    ]
    supervised_ids = [
        token_id
        for token_id, supervised in zip(
            lane["token_ids"], lane["loss_mask"], strict=True
        )
        if supervised
    ]
    return {
        "active_tokens": sum(lane["attention_mask"]),
        "attention_ones": sum(lane["attention_mask"]),
        "loss_tokens": sum(lane["loss_mask"]),
        "active_token_ids_sha256": hash_json(active_ids),
        "supervised_token_ids_sha256": hash_json(supervised_ids),
        "token_ids_sha256": hash_json(lane["token_ids"]),
        "attention_mask_sha256": hash_json(lane["attention_mask"]),
        "loss_mask_sha256": hash_json(lane["loss_mask"]),
        "epoch_ids_sha256": hash_json(lane["epoch_ids"]),
        "position_ids_sha256": hash_json(lane["position_ids"]),
        "dense_position_ids_exact": lane["position_ids"]
        == list(range(len(lane["position_ids"]))),
    }


def _build_artifacts(
    *,
    tokenizer: FrozenTokenizer,
    replication_board: list[dict[str, Any]],
    source_receipts: dict[str, dict[str, Any]],
    parent_checkpoint_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
) -> dict[str, bytes]:
    cross_inventory = _aggregate_episode_inventory(replication_board)
    reserved_signatures = {episode_signature(row) for row in replication_board}
    train_rng = _new_bound_random(sha256_bytes((str(seed) + "\0train").encode("ascii")))
    train_episodes = generate_balanced_episodes(
        rng=train_rng,
        split="train",
        per_cell=train_per_cell,
        reserved_signatures=reserved_signatures,
        forbidden_inventory=cross_inventory,
        reserve_inventory_within_split=False,
        require_interventions=False,
    )
    if (
        mode == "production"
        and hash_json(train_episodes) != KNOWN_TRAIN_EPISODES_SHA256
    ):
        raise ContractError("production training episode reconstruction drifted")
    train_inventory = _aggregate_episode_inventory(train_episodes)
    development_forbidden = empty_overlap_inventory()
    merge_overlap_inventory(development_forbidden, cross_inventory)
    merge_overlap_inventory(development_forbidden, train_inventory)
    development_rng = _new_bound_random(
        sha256_bytes((str(seed) + "\0development").encode("ascii"))
    )
    development = generate_balanced_episodes(
        rng=development_rng,
        split="development",
        per_cell=development_per_cell,
        reserved_signatures=reserved_signatures,
        forbidden_inventory=development_forbidden,
        reserve_inventory_within_split=True,
        require_interventions=True,
    )
    development_inventory = _aggregate_episode_inventory(development)
    if not overlap_inventory_is_disjoint(train_inventory, development_inventory):
        raise ContractError("development inventory overlaps training")
    if not overlap_inventory_is_disjoint(train_inventory, cross_inventory):
        raise ContractError("cross-width inventory overlaps training")

    trace_rows = [_trace_row(episode) for episode in train_episodes]
    control_rows_by_episode = {
        episode["id"]: _control_rows(episode) for episode in train_episodes
    }
    control_rows = [
        row
        for episode in train_episodes
        for row in control_rows_by_episode[episode["id"]]
    ]
    sham_rows, donors = build_sham_rows(train_episodes, tokenizer)
    sham_by_episode = {row["episode_id"]: row for row in sham_rows}
    seed_schedules = build_seed_schedules([episode["id"] for episode in train_episodes])
    seed_schedules_payload = pretty_json_bytes(seed_schedules)

    binary_buffers = {arm: bytearray() for arm in DATA_ARMS}
    binary_digests = {arm: _BOUND_HASHLIB_SHA256() for arm in DATA_ARMS}
    pack_receipts = []
    supervised_totals = Counter()
    active_totals = Counter()
    global_supervised = {arm: [] for arm in DATA_ARMS}
    context_hashes = {arm: _BOUND_HASHLIB_SHA256() for arm in DATA_ARMS}
    for pack_index, episode in enumerate(train_episodes):
        arms = _pack_arms_for_episode(
            episode=episode,
            sham_row=sham_by_episode[episode["id"]],
            tokenizer=tokenizer,
            lane_length=lane_length,
        )
        reference = arms["full_trace"]
        for arm in DATA_ARMS[1:]:
            for lane_index, lane in enumerate(arms[arm]):
                if lane["token_ids"] != reference[lane_index]["token_ids"]:
                    raise ContractError("per-lane active token IDs are not exact")
                if lane["attention_mask"] != reference[lane_index]["attention_mask"]:
                    raise ContractError("per-lane attention masks are not exact")
                if lane["epoch_ids"] != reference[lane_index]["epoch_ids"]:
                    raise ContractError("per-lane epoch IDs are not exact")
                if lane["position_ids"] != reference[lane_index]["position_ids"]:
                    raise ContractError("per-lane position IDs are not exact")
        arm_receipts = {}
        for arm, lanes in arms.items():
            payload = _BOUND_PACK_PAYLOAD(lanes, lane_length)
            binary_buffers[arm].extend(payload)
            binary_digests[arm].update(payload)
            supervised = _supervised_ids(lanes)
            global_supervised[arm].extend(supervised)
            supervised_totals[arm] += len(supervised)
            active_totals[arm] += _active_tokens(lanes)
            active_ids = [
                token_id
                for lane in lanes
                for token_id, active in zip(
                    lane["token_ids"], lane["attention_mask"], strict=True
                )
                if active
            ]
            active_payload = _BOUND_STRUCT_PACK(
                "<{}I".format(len(active_ids)), *active_ids
            )
            context_hashes[arm].update(active_payload)
            arm_receipts[arm] = {
                "payload_bytes": len(payload),
                "payload_sha256": sha256_bytes(payload),
                "supervised_tokens": len(supervised),
                "active_tokens": _active_tokens(lanes),
                "fixed_forward_positions": LANES_PER_PACK * lane_length,
                "lanes": [
                    {"role": LANE_ROLES[index], **_lane_receipt(lane)}
                    for index, lane in enumerate(lanes)
                ],
            }
        if len({row["supervised_tokens"] for row in arm_receipts.values()}) != 1:
            raise ContractError("per-pack supervised token counts differ")
        if len({row["active_tokens"] for row in arm_receipts.values()}) != 1:
            raise ContractError("per-pack active context token counts differ")
        pack_receipts.append(
            {
                "schema": "shohin-dws-single-completion-pack-receipt-v3",
                "pack_index": pack_index,
                "source_episode_id": episode["id"],
                "operation": episode["operation"],
                "intermediate_carry_pattern": episode["intermediate_carry_pattern"],
                "lane_roles": list(LANE_ROLES),
                "attention_topology": "independent_batch_lanes_no_cross_lane_attention",
                "position_ids": {
                    "normative_per_lane": "range(lane_length)",
                    "serialized": True,
                    "integer_encoding": "little-endian uint16",
                },
                "arms": arm_receipts,
                "sham_donors": donors[episode["id"]],
            }
        )
    if len(set(supervised_totals.values())) != 1:
        raise ContractError("arm-level supervised token totals differ")
    if len(set(active_totals.values())) != 1:
        raise ContractError("arm-level active context token totals differ")
    if global_supervised["full_trace"] != global_supervised["decomposed_one_step"]:
        raise ContractError("trace/control global target-token sequence differs")
    if Counter(global_supervised["full_trace"]) != Counter(
        global_supervised["multiline_sham"]
    ):
        raise ContractError("trace/sham global target-token multiset differs")
    if len({digest.hexdigest() for digest in context_hashes.values()}) != 1:
        raise ContractError("active context token IDs differ across arms")

    pack_count = len(train_episodes)
    supervised_total = int(next(iter(supervised_totals.values())))
    active_total = int(next(iter(active_totals.values())))
    training_plan = _training_plan(
        pack_count=pack_count,
        lane_length=lane_length,
        supervised_tokens=supervised_total,
        active_tokens=active_total,
        seed_schedules_sha256=sha256_bytes(seed_schedules_payload),
    )
    training_plan["delimiter"]["token_id"] = tokenizer.commit_token_id
    training_plan["delimiter"]["single_token_roundtrip"] = True

    source_commitment = hash_json(source_receipts)
    development_payload = jsonl_bytes(development)
    cross_payload = jsonl_bytes(replication_board)
    development_commitment = {
        "schema": "shohin-dws-single-completion-development-commitment-v2",
        "protocol": PROTOCOL,
        "board_rows": len(development),
        "board_sha256": sha256_bytes(development_payload),
        "board_inventory": _inventory_receipt(development_inventory),
        "train_inventory": _inventory_receipt(train_inventory),
        "cross_width_board_sha256": sha256_bytes(cross_payload),
        "cross_width_inventory": _inventory_receipt(cross_inventory),
        "zero_train_overlap_all_inventory_fields": True,
        "source_bindings_sha256": source_commitment,
        "tokenizer_sha256": tokenizer.sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "seed_schedules_sha256": sha256_bytes(seed_schedules_payload),
        "confirmation_board_exists": False,
        "promotion_authority": False,
    }

    stratum_counts = Counter(
        (episode["operation"], tuple(episode["intermediate_carry_pattern"]))
        for episode in train_episodes
    )
    development_strata = Counter(
        (episode["operation"], tuple(episode["intermediate_carry_pattern"]))
        for episode in development
    )
    carry_cells = {
        operation: {
            str(position): Counter(
                episode["intermediate_carry_pattern"][position - 1]
                for episode in train_episodes
                if episode["operation"] == operation
            )
            for position in (1, 2, 3)
        }
        for operation in OPERATIONS
    }
    gates = {
        "train_episode_count": pack_count
        == len(OPERATIONS) * len(INTERMEDIATE_PATTERNS) * train_per_cell,
        "development_episode_count": len(development)
        == len(OPERATIONS) * len(INTERMEDIATE_PATTERNS) * development_per_cell,
        "all_terminal_carry_zero": all(
            episode["terminal_carry"] == 0 for episode in train_episodes + development
        ),
        "train_strata_exact": len(stratum_counts) == 16
        and set(stratum_counts.values()) == {train_per_cell},
        "development_strata_exact": len(development_strata) == 16
        and set(development_strata.values()) == {development_per_cell},
        "intermediate_carry_borrow_balanced": all(
            counts[0] == counts[1]
            for operation in carry_cells.values()
            for counts in operation.values()
        ),
        "train_development_inventory_disjoint": overlap_inventory_is_disjoint(
            train_inventory, development_inventory
        ),
        "train_cross_width_inventory_disjoint": overlap_inventory_is_disjoint(
            train_inventory, cross_inventory
        ),
        "unique_train_signatures": len(
            {episode_signature(row) for row in train_episodes}
        )
        == pack_count,
        "unique_development_signatures": len(
            {episode_signature(row) for row in development}
        )
        == len(development),
        "unique_development_prompts": len(
            {full_trace_prompt(row["initial_state"]) for row in development}
        )
        == len(development),
        "cross_width_board_exact": len(replication_board) == 12,
        "sham_continuity_broken": all(
            row["all_adjacent_transitions_broken"] and row["answer_relation_broken"]
            for row in sham_rows
        ),
        "supervised_tokens_equal": len(set(supervised_totals.values())) == 1,
        "active_context_tokens_equal": len(set(active_totals.values())) == 1,
        "active_context_token_ids_equal": len(
            {digest.hexdigest() for digest in context_hashes.values()}
        )
        == 1,
        "serialized_dense_positions_exact": all(
            lane["dense_position_ids_exact"]
            for receipt in pack_receipts
            for arm_receipt in receipt["arms"].values()
            for lane in arm_receipt["lanes"]
        ),
        "seed_schedules_genuinely_distinct": len(
            {row["pack_order_sha256"] for row in seed_schedules["schedules"]}
        )
        == len(PAIRED_TRAINING_SEEDS),
        "no_confirmation_artifact": all(
            "confirmation" not in name.lower() for name in ARTIFACT_NAMES
        ),
    }
    if not all(gates.values()):
        failures = [name for name, passed in gates.items() if not passed]
        raise ContractError(
            "artifact construction gates failed: " + ", ".join(failures)
        )
    audit = {
        "schema": "shohin-dws-single-completion-audit-v2",
        "protocol": PROTOCOL,
        "mode": mode,
        "counts": {
            "train_episodes": pack_count,
            "development_episodes": len(development),
            "full_trace_rows": len(trace_rows),
            "decomposed_rows": len(control_rows),
            "multiline_sham_rows": len(sham_rows),
            "cross_width_replication_rows": len(replication_board),
            "run_cells": len(RUN_CELLS),
        },
        "train_inventory": _inventory_receipt(train_inventory),
        "development_inventory": _inventory_receipt(development_inventory),
        "cross_width_inventory": _inventory_receipt(cross_inventory),
        "supervised_tokens": dict(supervised_totals),
        "active_context_tokens": dict(active_totals),
        "active_context_token_ids_sha256": {
            arm: digest.hexdigest() for arm, digest in context_hashes.items()
        },
        "fixed_forward_positions_per_arm": pack_count * LANES_PER_PACK * lane_length,
        "position_ids": {
            "normative_per_lane": "range(lane_length)",
            "serialized": True,
            "integer_encoding": "little-endian uint16",
            "per_lane_sha256": hash_json(list(range(lane_length))),
        },
        "binary_sha256": {
            arm: digest.hexdigest() for arm, digest in binary_digests.items()
        },
        "local_causal_prior": LOCAL_CAUSAL_PRIOR,
        "cross_width_replication_prior": REPLICATION_PRIOR,
        "cache_pruning_prior": CACHE_PRUNING_PRIOR,
        "primary_gates": PRIMARY_GATES,
        "source_retirement_gates": SOURCE_RETIREMENT_GATES,
        "carry_target_switch_is_noncompensatory_veto": True,
        "full_trace_training_already_justified_by_prior": False,
        "gates": gates,
        "all_gates_pass": True,
        "verification_rule": (
            "These booleans have no evidentiary authority; verify_bundle independently "
            "reconstructs every artifact and semantic invariant."
        ),
        "claim_boundary": (
            "CPU protocol only. Any future score is a model-triggered external "
            "commit/re-encode package result, not autonomous base-model reasoning or a "
            "stale-source-specific mechanism attribution."
        ),
    }

    artifacts = {
        "train_episodes.jsonl": jsonl_bytes(train_episodes),
        "full_trace_train.jsonl": jsonl_bytes(trace_rows),
        "decomposed_one_step_train.jsonl": jsonl_bytes(control_rows),
        "multiline_sham_train.jsonl": jsonl_bytes(sham_rows),
        "development_board.jsonl": development_payload,
        "development_commitment.json": pretty_json_bytes(development_commitment),
        "cross_width_replication_board.jsonl": cross_payload,
        "pack_receipts.jsonl": jsonl_bytes(pack_receipts),
        "seed_schedules.json": seed_schedules_payload,
        "full_trace_packs.bin": bytes(binary_buffers["full_trace"]),
        "decomposed_one_step_packs.bin": bytes(binary_buffers["decomposed_one_step"]),
        "multiline_sham_packs.bin": bytes(binary_buffers["multiline_sham"]),
        "training_plan.json": pretty_json_bytes(training_plan),
        "audit_report.json": pretty_json_bytes(audit),
    }
    if set(artifacts) != set(ARTIFACT_NAMES):
        raise AssertionError("constructed artifact inventory drifted")
    return artifacts


_BOUND_BUILD_ARTIFACTS = _build_artifacts


def _sealed_root(
    *,
    artifacts: dict[str, bytes],
    source_receipts: dict[str, dict[str, Any]],
    runtime_receipts: dict[str, Any],
    tokenizer: FrozenTokenizer,
    replication_receipt: dict[str, Any],
    parent_checkpoint_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
    staging_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "shohin-dws-single-completion-sealed-root-v1",
        "protocol": PROTOCOL,
        "mode": mode,
        "publication_layout": {
            "bundle_directory": BUNDLE_DIRECTORY_NAME,
            "sealed_root_file": SEALED_ROOT_NAME,
            "external_sha256_and_byte_receipt_required": True,
            "manifest_self_authentication_allowed": False,
            "staging_identity": staging_identity,
        },
        "frozen_constants": {
            "generation_seed": seed,
            "train_per_cell": train_per_cell,
            "development_per_cell": development_per_cell,
            "lane_length": lane_length,
            "lanes_per_pack": LANES_PER_PACK,
            "packs_per_update": PACKS_PER_UPDATE,
            "paired_training_seeds": list(PAIRED_TRAINING_SEEDS),
            "run_cells": list(RUN_CELLS),
            "optimizer_contract_sha256": hash_json(OPTIMIZER_CONTRACT),
            "replication_case_ids_sha256": REPLICATION_CASE_IDS_SHA256,
        },
        "inputs": {
            "location_contract": {
                "policy": _BOUND_INPUT_LOCATION_POLICY,
                "path_is_identity": False,
                "sha256_is_identity": True,
                "regular_non_symlink_file_required": True,
                "build_and_verify_locations_may_differ": True,
            },
            "tokenizer": {
                "bytes": tokenizer.size,
                "sha256": tokenizer.sha256,
                "eos_token_id": tokenizer.eos_id,
                "lf_commit_token_id": tokenizer.commit_token_id,
            },
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "cross_width_source": replication_receipt,
            "source_bindings": source_receipts,
            "source_bindings_sha256": hash_json(source_receipts),
            "runtime_bindings": runtime_receipts,
            "runtime_bindings_sha256": hash_json(runtime_receipts),
        },
        "artifacts": _payload_receipts(artifacts),
        "authorization": {
            "cpu_bundle_generation": True,
            "production_corpus": False,
            "training": False,
            "gpu_job": False,
            "promotion": False,
            "capability_claim": False,
        },
        "claim_boundary": (
            "This externally authenticated root binds a CPU experiment contract only. "
            "It grants no training, GPU, promotion, or reasoning claim authority."
        ),
    }


def _validate_directory_fd(descriptor: int, label: str, *, exact_mode: int) -> None:
    metadata = _BOUND_OS_FSTAT(descriptor)
    if not _BOUND_STAT_S_ISDIR(metadata.st_mode):
        raise ContractError("{} must be a directory".format(label))
    if _BOUND_STAT_S_IMODE(metadata.st_mode) != exact_mode:
        raise ContractError("{} mode must be {:04o}".format(label, exact_mode))


def _sealed_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        _BOUND_STAT_S_IMODE(metadata.st_mode),
    )


def _open_held_publication_file_at(
    directory_fd: int, name: str, label: str
) -> tuple[int, tuple[int, ...]]:
    _validate_entry_name(name)
    try:
        descriptor = _BOUND_OS_OPEN(
            name,
            _BOUND_OS_O_RDONLY | _BOUND_OS_O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ContractError("{} cannot be opened safely".format(label)) from error
    try:
        metadata = _assert_regular_file_entry_identity(
            directory_fd, name, descriptor, label
        )
        if not _BOUND_STAT_S_ISREG(metadata.st_mode):
            raise ContractError("{} must be a regular file".format(label))
        if metadata.st_nlink != 1:
            raise ContractError("{} must have exactly one hard link".format(label))
        if _BOUND_STAT_S_IMODE(metadata.st_mode) != 0o444:
            raise ContractError("{} mode must be 0444".format(label))
        return descriptor, _sealed_file_identity(metadata)
    except BaseException:
        _BOUND_OS_CLOSE(descriptor)
        raise


def _assert_held_publication_file(
    *,
    directory_fd: int,
    name: str,
    descriptor: int,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    metadata = _assert_regular_file_entry_identity(
        directory_fd, name, descriptor, label
    )
    if _sealed_file_identity(metadata) != expected_identity:
        raise ContractError(label + " held inode metadata changed")


def _read_publication(
    pinned_parent: _PinnedDirectory, publication_name: str
) -> dict[str, Any]:
    if ".partial" in publication_name:
        raise ContractError("a partial staging tree cannot be a publication")
    publication_fd = _open_directory_at(
        pinned_parent.descriptor, publication_name, "publication directory"
    )
    opened_descriptors = [publication_fd]
    try:
        publication_identity = _directory_descriptor_identity(
            publication_fd, "publication directory"
        )
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            publication_name,
            publication_identity,
            "publication directory",
        )
        _validate_directory_fd(
            publication_fd, "publication directory", exact_mode=0o555
        )
        release_inventory = set(_BOUND_OS_LISTDIR(publication_fd))
        if release_inventory != {BUNDLE_DIRECTORY_NAME, SEALED_ROOT_NAME}:
            raise ContractError("publication inventory is not closed")
        sealed_descriptor, sealed_identity = _open_held_publication_file_at(
            publication_fd,
            SEALED_ROOT_NAME,
            "external sealed manifest",
        )
        opened_descriptors.append(sealed_descriptor)
        sealed_payload, sealed_receipt = _stable_file_bytes_at(
            publication_fd,
            SEALED_ROOT_NAME,
            "external sealed manifest",
            exact_mode=0o444,
            require_nlink_one=True,
            expected_identity=sealed_identity[:2],
        )
        _assert_held_publication_file(
            directory_fd=publication_fd,
            name=SEALED_ROOT_NAME,
            descriptor=sealed_descriptor,
            expected_identity=sealed_identity,
            label="external sealed manifest",
        )
        bundle_fd = _open_directory_at(
            publication_fd, BUNDLE_DIRECTORY_NAME, "bundle directory"
        )
        opened_descriptors.append(bundle_fd)
        bundle_identity = _directory_descriptor_identity(bundle_fd, "bundle directory")
        _assert_directory_entry_identity(
            publication_fd,
            BUNDLE_DIRECTORY_NAME,
            bundle_identity,
            "bundle directory",
        )
        _validate_directory_fd(bundle_fd, "bundle directory", exact_mode=0o555)
        bundle_inventory = set(_BOUND_OS_LISTDIR(bundle_fd))
        if bundle_inventory != set(ARTIFACT_NAMES):
            raise ContractError("bundle artifact inventory is not closed")
        payloads = {}
        artifact_descriptors = []
        for name in ARTIFACT_NAMES:
            label = "published artifact {}".format(name)
            descriptor, identity = _open_held_publication_file_at(
                bundle_fd, name, label
            )
            opened_descriptors.append(descriptor)
            payloads[name], _ = _stable_file_bytes_at(
                bundle_fd,
                name,
                label,
                exact_mode=0o444,
                require_nlink_one=True,
                expected_identity=identity[:2],
            )
            _assert_held_publication_file(
                directory_fd=bundle_fd,
                name=name,
                descriptor=descriptor,
                expected_identity=identity,
                label=label,
            )
            artifact_descriptors.append((name, descriptor, identity))
        _assert_directory_entry_identity(
            publication_fd,
            BUNDLE_DIRECTORY_NAME,
            bundle_identity,
            "bundle directory",
        )
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            publication_name,
            publication_identity,
            "publication directory",
        )
        return {
            "publication_descriptor": publication_fd,
            "publication_identity": publication_identity,
            "bundle_descriptor": bundle_fd,
            "bundle_identity": bundle_identity,
            "sealed_descriptor": sealed_descriptor,
            "sealed_identity": sealed_identity,
            "sealed_payload": sealed_payload,
            "sealed_receipt": sealed_receipt,
            "artifact_payloads": payloads,
            "artifact_descriptors": tuple(artifact_descriptors),
            "opened_descriptors": tuple(opened_descriptors),
        }
    except BaseException:
        for descriptor in reversed(opened_descriptors):
            _BOUND_OS_CLOSE(descriptor)
        raise


def _close_held_publication(publication: dict[str, Any]) -> None:
    for descriptor in reversed(publication["opened_descriptors"]):
        _BOUND_OS_CLOSE(descriptor)


def _revalidate_held_publication(
    pinned_parent: _PinnedDirectory,
    publication_name: str,
    publication: dict[str, Any],
) -> None:
    publication_fd = publication["publication_descriptor"]
    bundle_fd = publication["bundle_descriptor"]
    sealed_payload, _ = _stable_file_descriptor_bytes(
        publication["sealed_descriptor"],
        "external sealed manifest",
        directory_fd=publication_fd,
        name=SEALED_ROOT_NAME,
        exact_mode=0o444,
        require_nlink_one=True,
        expected_identity=publication["sealed_identity"][:2],
    )
    if sealed_payload != publication["sealed_payload"]:
        raise ContractError("external sealed manifest changed after semantic replay")
    for name, descriptor, identity in publication["artifact_descriptors"]:
        label = "published artifact {}".format(name)
        payload, _ = _stable_file_descriptor_bytes(
            descriptor,
            label,
            directory_fd=bundle_fd,
            name=name,
            exact_mode=0o444,
            require_nlink_one=True,
            expected_identity=identity[:2],
        )
        if payload != publication["artifact_payloads"][name]:
            raise ContractError(label + " changed after semantic replay")

    _assert_pinned_directory(pinned_parent)
    release_inventory = set(_BOUND_OS_LISTDIR(publication_fd))
    bundle_inventory = set(_BOUND_OS_LISTDIR(bundle_fd))
    if release_inventory != {BUNDLE_DIRECTORY_NAME, SEALED_ROOT_NAME}:
        raise ContractError("publication inventory changed after semantic replay")
    if bundle_inventory != set(ARTIFACT_NAMES):
        raise ContractError("bundle inventory changed after semantic replay")
    _validate_directory_fd(publication_fd, "publication directory", exact_mode=0o555)
    _validate_directory_fd(bundle_fd, "bundle directory", exact_mode=0o555)
    _assert_held_publication_file(
        directory_fd=publication_fd,
        name=SEALED_ROOT_NAME,
        descriptor=publication["sealed_descriptor"],
        expected_identity=publication["sealed_identity"],
        label="external sealed manifest",
    )
    _assert_directory_entry_identity(
        publication_fd,
        BUNDLE_DIRECTORY_NAME,
        publication["bundle_identity"],
        "bundle directory",
    )
    for name, descriptor, identity in publication["artifact_descriptors"]:
        _assert_held_publication_file(
            directory_fd=bundle_fd,
            name=name,
            descriptor=descriptor,
            expected_identity=identity,
            label="published artifact {}".format(name),
        )
    _assert_directory_entry_identity(
        pinned_parent.descriptor,
        publication_name,
        publication["publication_identity"],
        "publication directory",
    )


def _verify_held_publication_semantics(
    publication: dict[str, Any],
    publication_dir: Path,
    *,
    pinned_parent: _PinnedDirectory,
    external_manifest_path: Path,
    expected_external_manifest_sha256: str,
    expected_external_manifest_bytes: int,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
    staging_identity: dict[str, Any],
    _construct: Any,
    _runtime_bindings: Any,
    _source_bindings: Any,
) -> dict[str, Any]:
    actual_artifacts = publication["artifact_payloads"]
    sealed_payload = publication["sealed_payload"]
    sealed_receipt = publication["sealed_receipt"]
    if sealed_receipt != {
        "bytes": expected_external_manifest_bytes,
        "sha256": expected_external_manifest_sha256,
    }:
        raise ContractError("external sealed-manifest receipt mismatch")
    sealed_root = strict_json_loads(sealed_payload, "external sealed manifest")

    current_source_receipts = _source_bindings()
    if hash_json(current_source_receipts) != expected_source_bindings_sha256:
        raise ContractError("source bytes changed after external commitment")
    current_runtime_receipts = _validated_runtime_bindings(
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
        source_receipts=current_source_receipts,
    )
    tokenizer = FrozenTokenizer(tokenizer_path, expected_tokenizer_sha256)
    replication_payload, replication_receipt = _stable_file_bytes(
        replication_source, "cross-width replication source"
    )
    if replication_receipt["sha256"] != expected_replication_source_sha256:
        raise ContractError("cross-width replication source receipt mismatch")
    replication_board = load_replication_board(
        replication_source, expected_replication_source_sha256
    )
    expected_artifacts = _construct(
        tokenizer=tokenizer,
        replication_board=replication_board,
        source_receipts=current_source_receipts,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
    )
    for name in ARTIFACT_NAMES:
        if actual_artifacts[name] != expected_artifacts[name]:
            raise ContractError(
                "independent semantic bundle replay mismatch: {}".format(name)
            )
    expected_root = _sealed_root(
        artifacts=expected_artifacts,
        source_receipts=current_source_receipts,
        runtime_receipts=current_runtime_receipts,
        tokenizer=tokenizer,
        replication_receipt=replication_receipt,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
        staging_identity=staging_identity,
    )
    expected_root_payload = pretty_json_bytes(expected_root)
    if sealed_root != expected_root or sealed_payload != expected_root_payload:
        raise ContractError("sealed root differs from external frozen constants")
    if sealed_root["authorization"] != {
        "cpu_bundle_generation": True,
        "production_corpus": False,
        "training": False,
        "gpu_job": False,
        "promotion": False,
        "capability_claim": False,
    }:
        raise ContractError("sealed root authorization was widened")

    _emit_runtime_phase(
        "after semantic replay before sealed publication revalidation",
        publication_dir,
    )
    final_replication_payload, _ = _stable_file_bytes(
        replication_source, "final cross-width replication source"
    )
    if replication_payload != final_replication_payload:
        raise ContractError("replication source changed after sealed read")
    if _source_bindings() != current_source_receipts:
        raise ContractError("source bytes changed during final replay")
    if _runtime_bindings() != current_runtime_receipts:
        raise ContractError("runtime bindings changed during final replay")
    _revalidate_held_publication(pinned_parent, publication_dir.name, publication)
    return {
        "schema": "shohin-dws-single-completion-independent-verification-v1",
        "verified": True,
        "semantic_replay": True,
        "external_manifest_path": str(external_manifest_path.absolute()),
        "external_manifest_sha256": sealed_receipt["sha256"],
        "external_manifest_bytes": sealed_receipt["bytes"],
        "artifact_count": len(actual_artifacts),
        "source_bindings_sha256": expected_source_bindings_sha256,
        "runtime_bindings_sha256": expected_runtime_bindings_sha256,
        "training_authorized": False,
        "gpu_authorized": False,
        "promotion_authorized": False,
    }


def _verify_bundle_pinned(
    publication_dir: Path,
    *,
    pinned_parent: _PinnedDirectory,
    external_manifest_path: Path,
    expected_external_manifest_sha256: str,
    expected_external_manifest_bytes: int,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int = GENERATION_SEED,
    train_per_cell: int = TRAIN_PER_CELL,
    development_per_cell: int = DEVELOPMENT_PER_CELL,
    lane_length: int = LANE_LENGTH,
    _construct: Any = _BOUND_BUILD_ARTIFACTS,
    _runtime_bindings: Any = runtime_bindings,
    _source_bindings: Any = source_bindings,
) -> dict[str, Any]:
    publication_dir = Path(_BOUND_OS_PATH_ABSPATH(_BOUND_OS_FSPATH(publication_dir)))
    if ".partial" in publication_dir.name:
        raise ContractError("a partial staging tree cannot be a publication")
    if publication_dir.parent != pinned_parent.path:
        raise ContractError("publication does not use the pinned parent")
    _assert_pinned_directory(pinned_parent)
    external_manifest_path = Path(external_manifest_path)
    expected_manifest_path = publication_dir / SEALED_ROOT_NAME
    if external_manifest_path.absolute() != expected_manifest_path.absolute():
        raise ContractError("external manifest path does not name the sealed root")
    if not HEX64_RE.fullmatch(expected_external_manifest_sha256):
        raise ContractError("external manifest SHA-256 receipt is malformed")
    if (
        type(expected_external_manifest_bytes) is not int
        or expected_external_manifest_bytes <= 0
    ):
        raise ContractError("external manifest byte receipt is malformed")
    _validate_generation_contract(
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_replication_source_sha256=expected_replication_source_sha256,
        expected_source_bindings_sha256=expected_source_bindings_sha256,
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
    )
    staging_identity = _staging_identity(
        out_dir=publication_dir,
        pinned_parent=pinned_parent,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_replication_source_sha256=expected_replication_source_sha256,
        expected_source_bindings_sha256=expected_source_bindings_sha256,
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
    )
    publication = _read_publication(pinned_parent, publication_dir.name)
    try:
        return _verify_held_publication_semantics(
            publication,
            publication_dir,
            pinned_parent=pinned_parent,
            external_manifest_path=external_manifest_path,
            expected_external_manifest_sha256=expected_external_manifest_sha256,
            expected_external_manifest_bytes=expected_external_manifest_bytes,
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            replication_source=replication_source,
            expected_replication_source_sha256=expected_replication_source_sha256,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            mode=mode,
            seed=seed,
            train_per_cell=train_per_cell,
            development_per_cell=development_per_cell,
            lane_length=lane_length,
            staging_identity=staging_identity,
            _construct=_construct,
            _runtime_bindings=_runtime_bindings,
            _source_bindings=_source_bindings,
        )
    finally:
        _close_held_publication(publication)


_BOUND_VERIFY_BUNDLE_PINNED = _verify_bundle_pinned


def _publication_success_receipt(
    *,
    out_dir: Path,
    sealed_receipt: dict[str, Any],
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    verification: dict[str, Any],
    disposition: str,
) -> dict[str, Any]:
    return {
        "schema": "shohin-dws-single-completion-publication-receipt-v2",
        "publication_disposition": disposition,
        "publication_dir": str(out_dir.absolute()),
        "bundle_dir": str((out_dir / BUNDLE_DIRECTORY_NAME).absolute()),
        "external_manifest_path": str((out_dir / SEALED_ROOT_NAME).absolute()),
        "external_manifest_sha256": sealed_receipt["sha256"],
        "external_manifest_bytes": sealed_receipt["bytes"],
        "source_bindings_sha256": expected_source_bindings_sha256,
        "runtime_bindings_sha256": expected_runtime_bindings_sha256,
        "verified": verification["verified"],
        "training_authorized": False,
        "gpu_authorized": False,
        "promotion_authorized": False,
    }


def _recover_existing_publication(
    *,
    pinned_parent: _PinnedDirectory,
    out_dir: Path,
    staging_identity: dict[str, Any],
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
    _verify_pinned: Any = _BOUND_VERIFY_BUNDLE_PINNED,
    _close: Any = _BOUND_OS_CLOSE,
) -> dict[str, Any]:
    publication_descriptor = _lock_existing_publication(
        pinned_parent.descriptor, out_dir.name
    )
    try:
        publication_identity = _directory_descriptor_identity(
            publication_descriptor, "existing publication"
        )
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            out_dir.name,
            publication_identity,
            "existing publication",
        )
        _validate_protocol_staging_tree(publication_descriptor, staging_identity)
        _validate_directory_fd(
            publication_descriptor, "existing publication directory", exact_mode=0o555
        )
        _, sealed_receipt = _stable_file_bytes_at(
            publication_descriptor,
            SEALED_ROOT_NAME,
            "existing publication sealed manifest",
            exact_mode=0o444,
            require_nlink_one=True,
        )
        _assert_pinned_directory(pinned_parent)
        verification = _verify_pinned(
            out_dir,
            pinned_parent=pinned_parent,
            external_manifest_path=out_dir / SEALED_ROOT_NAME,
            expected_external_manifest_sha256=sealed_receipt["sha256"],
            expected_external_manifest_bytes=sealed_receipt["bytes"],
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            replication_source=replication_source,
            expected_replication_source_sha256=expected_replication_source_sha256,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            mode=mode,
            seed=seed,
            train_per_cell=train_per_cell,
            development_per_cell=development_per_cell,
            lane_length=lane_length,
        )
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            out_dir.name,
            publication_identity,
            "recovered publication",
        )
        _assert_pinned_directory(pinned_parent)
        return _publication_success_receipt(
            out_dir=out_dir,
            sealed_receipt=sealed_receipt,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            verification=verification,
            disposition="recovered_existing_exact_publication",
        )
    finally:
        _close(publication_descriptor)


_BOUND_RECOVER_EXISTING_PUBLICATION = _recover_existing_publication


def verify_bundle(
    publication_dir: Path,
    *,
    external_manifest_path: Path,
    expected_external_manifest_sha256: str,
    expected_external_manifest_bytes: int,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int = GENERATION_SEED,
    train_per_cell: int = TRAIN_PER_CELL,
    development_per_cell: int = DEVELOPMENT_PER_CELL,
    lane_length: int = LANE_LENGTH,
    _verify_pinned: Any = _BOUND_VERIFY_BUNDLE_PINNED,
) -> dict[str, Any]:
    if _verify_pinned is not _BOUND_VERIFY_BUNDLE_PINNED:
        raise ContractError("verification dispatch identity changed")
    publication_dir = Path(_BOUND_OS_PATH_ABSPATH(_BOUND_OS_FSPATH(publication_dir)))
    pinned_parent = _pin_publication_parent(publication_dir.parent, create=False)
    try:
        return _verify_pinned(
            publication_dir,
            pinned_parent=pinned_parent,
            external_manifest_path=external_manifest_path,
            expected_external_manifest_sha256=expected_external_manifest_sha256,
            expected_external_manifest_bytes=expected_external_manifest_bytes,
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            replication_source=replication_source,
            expected_replication_source_sha256=expected_replication_source_sha256,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            mode=mode,
            seed=seed,
            train_per_cell=train_per_cell,
            development_per_cell=development_per_cell,
            lane_length=lane_length,
        )
    finally:
        pinned_parent.close()


def _build_and_publish_pinned(
    *,
    pinned_parent: _PinnedDirectory,
    out_dir: Path,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
    _construct: Any = _BOUND_BUILD_ARTIFACTS,
    _runtime_snapshot: Any = _assert_runtime_snapshot,
    _runtime_phase: Any = _emit_runtime_phase,
    _atomic_publish: Any = _BOUND_ATOMIC_RENAME_NOREPLACE_AT,
    _write_file: Any = _write_sealed_file_at,
    _fsync_directory: Any = _fsync_directory_fd,
    _verify_pinned: Any = _BOUND_VERIFY_BUNDLE_PINNED,
    _runtime_bindings: Any = runtime_bindings,
    _source_bindings: Any = source_bindings,
    _remove_staging: Any = _remove_protocol_staging_tree,
    _recover_existing: Any = _BOUND_RECOVER_EXISTING_PUBLICATION,
    _entry_metadata_bound: Any = _entry_metadata,
    _entry_has_identity: Any = _directory_entry_has_identity,
    _close: Any = _BOUND_OS_CLOSE,
) -> dict[str, Any]:
    out_dir = Path(_BOUND_OS_PATH_ABSPATH(_BOUND_OS_FSPATH(out_dir)))
    if (
        out_dir.name in ("", ".", "..")
        or out_dir.parent == out_dir
        or ".partial" in out_dir.name
    ):
        raise ContractError("invalid publication path")
    if out_dir.parent != pinned_parent.path:
        raise ContractError("publication target does not use the pinned parent")
    _assert_pinned_directory(pinned_parent)
    _validate_generation_contract(
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_replication_source_sha256=expected_replication_source_sha256,
        expected_source_bindings_sha256=expected_source_bindings_sha256,
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
    )
    source_receipts = _source_bindings()
    if hash_json(source_receipts) != expected_source_bindings_sha256:
        raise ContractError("source bindings changed before construction")
    runtime_receipts = _validated_runtime_bindings(
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
        source_receipts=source_receipts,
    )
    staging_identity = _staging_identity(
        out_dir=out_dir,
        pinned_parent=pinned_parent,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        expected_replication_source_sha256=expected_replication_source_sha256,
        expected_source_bindings_sha256=expected_source_bindings_sha256,
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
    )
    if _entry_metadata_bound(pinned_parent.descriptor, out_dir.name) is not None:
        return _recover_existing(
            pinned_parent=pinned_parent,
            out_dir=out_dir,
            staging_identity=staging_identity,
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            replication_source=replication_source,
            expected_replication_source_sha256=expected_replication_source_sha256,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            mode=mode,
            seed=seed,
            train_per_cell=train_per_cell,
            development_per_cell=development_per_cell,
            lane_length=lane_length,
        )
    stage_name = _partial_stage_path(out_dir).name
    _recover_protocol_staging_tree(pinned_parent, stage_name, staging_identity)
    _runtime_snapshot(runtime_receipts, "before construction")
    _runtime_phase("after initial runtime validation", out_dir)
    _runtime_snapshot(runtime_receipts, "after initial runtime validation")

    tokenizer = FrozenTokenizer(tokenizer_path, expected_tokenizer_sha256)
    replication_payload, replication_receipt = _stable_file_bytes(
        replication_source, "cross-width replication source"
    )
    if replication_receipt["sha256"] != expected_replication_source_sha256:
        raise ContractError("cross-width replication source receipt mismatch")
    replication_board = load_replication_board(
        replication_source, expected_replication_source_sha256
    )
    artifacts = _construct(
        tokenizer=tokenizer,
        replication_board=replication_board,
        source_receipts=source_receipts,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
    )
    _runtime_snapshot(runtime_receipts, "during construction")
    sealed_root = _sealed_root(
        artifacts=artifacts,
        source_receipts=source_receipts,
        runtime_receipts=runtime_receipts,
        tokenizer=tokenizer,
        replication_receipt=replication_receipt,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
        staging_identity=staging_identity,
    )
    sealed_payload = pretty_json_bytes(sealed_root)
    sealed_receipt = {
        "bytes": len(sealed_payload),
        "sha256": sha256_bytes(sealed_payload),
    }

    _runtime_snapshot(runtime_receipts, "before staging publication")
    _assert_pinned_directory(pinned_parent)
    try:
        _BOUND_OS_MKDIR(stage_name, mode=0o700, dir_fd=pinned_parent.descriptor)
    except FileExistsError as error:
        raise ContractError("staging path appeared during construction") from error
    except OSError as error:
        raise ContractError("staging directory cannot be created safely") from error
    stage_descriptor: int | None = None
    stage_directory_identity: tuple[int, int] | None = None
    bundle_descriptor: int | None = None
    renamed = False
    published = False
    try:
        stage_descriptor = _lock_stage_directory(pinned_parent.descriptor, stage_name)
        stage_directory_identity = _directory_descriptor_identity(
            stage_descriptor, "locked staging"
        )
        if stage_directory_identity[0] != pinned_parent.device:
            raise ContractError("staging directory is not on publication filesystem")
        _assert_pinned_directory(pinned_parent)
        owner_payload = pretty_json_bytes(
            {"schema": STAGING_OWNER_SCHEMA, "identity": staging_identity}
        )
        _write_file(stage_descriptor, SEALED_ROOT_NAME, owner_payload)
        _fsync_directory(stage_descriptor)
        _fsync_directory(pinned_parent.descriptor)
        _BOUND_OS_MKDIR(BUNDLE_DIRECTORY_NAME, mode=0o700, dir_fd=stage_descriptor)
        bundle_descriptor = _open_directory_at(
            stage_descriptor, BUNDLE_DIRECTORY_NAME, "staging bundle"
        )
        for name in ARTIFACT_NAMES:
            _write_file(bundle_descriptor, name, artifacts[name])
        _chmod_directory_fd(bundle_descriptor, "staging bundle", 0o555)
        _fsync_directory(bundle_descriptor)
        _close(bundle_descriptor)
        bundle_descriptor = None
        _write_file(stage_descriptor, STAGING_NEXT_MANIFEST_NAME, sealed_payload)
        _BOUND_OS_REPLACE(
            STAGING_NEXT_MANIFEST_NAME,
            SEALED_ROOT_NAME,
            src_dir_fd=stage_descriptor,
            dst_dir_fd=stage_descriptor,
        )
        completed_manifest_payload, _ = _stable_file_bytes_at(
            stage_descriptor,
            SEALED_ROOT_NAME,
            "completed staging manifest",
            exact_mode=0o444,
            require_nlink_one=True,
        )
        if completed_manifest_payload != sealed_payload:
            raise ContractError("completed staging manifest bytes changed")
        _fsync_directory(stage_descriptor)
        _chmod_directory_fd(stage_descriptor, "staging publication", 0o555)
        _fsync_directory(stage_descriptor)
        _fsync_directory(pinned_parent.descriptor)
        _assert_pinned_directory(pinned_parent)
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            stage_name,
            stage_directory_identity,
            "atomic publication source",
        )
        _runtime_snapshot(runtime_receipts, "before atomic publication")
        _runtime_phase("during publication after runtime validation", out_dir)
        _runtime_snapshot(runtime_receipts, "immediately before atomic publication")
        _assert_pinned_directory(pinned_parent)
        _runtime_phase("immediately before atomic no-replace rename", out_dir)
        _atomic_publish(
            pinned_parent.descriptor,
            stage_name,
            out_dir.name,
            stage_directory_identity,
        )
        renamed = True
        _runtime_phase("after destination rename before external receipt", out_dir)
        _assert_directory_entry_identity(
            pinned_parent.descriptor,
            out_dir.name,
            stage_directory_identity,
            "atomic publication destination",
        )
        _fsync_directory(pinned_parent.descriptor)
        _runtime_snapshot(runtime_receipts, "during atomic publication")
        _assert_pinned_directory(pinned_parent)
        verification = _verify_pinned(
            out_dir,
            pinned_parent=pinned_parent,
            external_manifest_path=out_dir / SEALED_ROOT_NAME,
            expected_external_manifest_sha256=sealed_receipt["sha256"],
            expected_external_manifest_bytes=sealed_receipt["bytes"],
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            replication_source=replication_source,
            expected_replication_source_sha256=expected_replication_source_sha256,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            mode=mode,
            seed=seed,
            train_per_cell=train_per_cell,
            development_per_cell=development_per_cell,
            lane_length=lane_length,
        )
        publication_descriptor = _open_directory_at(
            pinned_parent.descriptor, out_dir.name, "publication directory"
        )
        try:
            final_bundle_descriptor = _open_directory_at(
                publication_descriptor, BUNDLE_DIRECTORY_NAME, "bundle directory"
            )
            try:
                _fsync_directory(final_bundle_descriptor)
            finally:
                _close(final_bundle_descriptor)
            _fsync_directory(publication_descriptor)
        finally:
            _close(publication_descriptor)
        _fsync_directory(pinned_parent.descriptor)
        _assert_pinned_directory(pinned_parent)
        if _source_bindings() != source_receipts:
            raise ContractError("source bytes changed after final publication replay")
        if _runtime_bindings() != runtime_receipts:
            raise ContractError(
                "runtime bindings changed after final publication replay"
            )
        final_replication_payload, _ = _stable_file_bytes(
            replication_source, "final cross-width replication source"
        )
        if replication_payload != final_replication_payload:
            raise ContractError(
                "replication source changed after final publication replay"
            )
        published = True
        return _publication_success_receipt(
            out_dir=out_dir,
            sealed_receipt=sealed_receipt,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            verification=verification,
            disposition="created_new_publication",
        )
    finally:
        try:
            if bundle_descriptor is not None:
                _close(bundle_descriptor)
            if (
                not published
                and not renamed
                and _entry_metadata_bound(pinned_parent.descriptor, stage_name)
                is not None
                and (
                    stage_descriptor is None
                    or (
                        stage_directory_identity is not None
                        and _entry_has_identity(
                            pinned_parent.descriptor,
                            stage_name,
                            stage_directory_identity,
                        )
                    )
                )
            ):
                _remove_staging(
                    pinned_parent,
                    stage_name,
                    staging_identity,
                    locked_descriptor=stage_descriptor,
                )
        finally:
            if stage_descriptor is not None:
                _close(stage_descriptor)


_BOUND_BUILD_AND_PUBLISH_PINNED = _build_and_publish_pinned


def _build_and_publish(
    *,
    out_dir: Path,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int,
    train_per_cell: int,
    development_per_cell: int,
    lane_length: int,
    _publish_pinned: Any = _BOUND_BUILD_AND_PUBLISH_PINNED,
) -> dict[str, Any]:
    if _publish_pinned is not _BOUND_BUILD_AND_PUBLISH_PINNED:
        raise ContractError("pinned publication dispatch identity changed")
    out_dir = Path(_BOUND_OS_PATH_ABSPATH(_BOUND_OS_FSPATH(out_dir)))
    if (
        out_dir.name in ("", ".", "..")
        or out_dir.parent == out_dir
        or ".partial" in out_dir.name
    ):
        raise ContractError("invalid publication path")
    pinned_parent = _pin_publication_parent(out_dir.parent, create=True)
    try:
        return _publish_pinned(
            pinned_parent=pinned_parent,
            out_dir=out_dir,
            tokenizer_path=tokenizer_path,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            replication_source=replication_source,
            expected_replication_source_sha256=expected_replication_source_sha256,
            expected_source_bindings_sha256=expected_source_bindings_sha256,
            expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
            mode=mode,
            seed=seed,
            train_per_cell=train_per_cell,
            development_per_cell=development_per_cell,
            lane_length=lane_length,
        )
    finally:
        pinned_parent.close()


_BOUND_BUILD_AND_PUBLISH = _build_and_publish


def build_bundle(
    *,
    out_dir: Path,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    replication_source: Path,
    expected_replication_source_sha256: str,
    expected_source_bindings_sha256: str,
    expected_runtime_bindings_sha256: str,
    mode: str,
    seed: int = GENERATION_SEED,
    train_per_cell: int = TRAIN_PER_CELL,
    development_per_cell: int = DEVELOPMENT_PER_CELL,
    lane_length: int = LANE_LENGTH,
    _publish: Any = _BOUND_BUILD_AND_PUBLISH,
) -> dict[str, Any]:
    if _publish is not _BOUND_BUILD_AND_PUBLISH:
        raise ContractError("publication dispatch identity changed")
    return _publish(
        out_dir=out_dir,
        tokenizer_path=tokenizer_path,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        replication_source=replication_source,
        expected_replication_source_sha256=expected_replication_source_sha256,
        expected_source_bindings_sha256=expected_source_bindings_sha256,
        expected_runtime_bindings_sha256=expected_runtime_bindings_sha256,
        mode=mode,
        seed=seed,
        train_per_cell=train_per_cell,
        development_per_cell=development_per_cell,
        lane_length=lane_length,
    )


def parse_args() -> argparse.Namespace:
    parser = _BOUND_ARGUMENT_PARSER(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out-dir", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--print-bindings", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--parent-checkpoint-sha256")
    parser.add_argument("--replication-source", type=Path)
    parser.add_argument("--expected-replication-source-sha256")
    parser.add_argument("--expected-source-bindings-sha256")
    parser.add_argument("--expected-runtime-bindings-sha256")
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--expected-external-manifest-sha256")
    parser.add_argument("--expected-external-manifest-bytes", type=int)
    parser.add_argument("--mode", choices=("production", "test"))
    parser.add_argument("--seed", type=int, default=GENERATION_SEED)
    parser.add_argument("--train-per-cell", type=int, default=TRAIN_PER_CELL)
    parser.add_argument(
        "--development-per-cell", type=int, default=DEVELOPMENT_PER_CELL
    )
    parser.add_argument("--lane-length", type=int, default=LANE_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.print_bindings:
            result = {
                "schema": "shohin-dws-single-completion-cli-bindings-v1",
                "source_bindings_sha256": source_bindings_sha256(),
                "runtime_bindings_sha256": runtime_bindings_sha256(),
            }
            _assert_bound_runtime_exports()
            print(_BOUND_JSON_DUMPS(result, sort_keys=True))
            return
        required_arguments = {
            "--tokenizer": args.tokenizer,
            "--expected-tokenizer-sha256": args.expected_tokenizer_sha256,
            "--parent-checkpoint-sha256": args.parent_checkpoint_sha256,
            "--replication-source": args.replication_source,
            "--expected-replication-source-sha256": (
                args.expected_replication_source_sha256
            ),
            "--expected-source-bindings-sha256": args.expected_source_bindings_sha256,
            "--expected-runtime-bindings-sha256": (
                args.expected_runtime_bindings_sha256
            ),
            "--mode": args.mode,
        }
        missing_arguments = [
            name for name, value in required_arguments.items() if value is None
        ]
        if missing_arguments:
            raise ContractError(
                "missing required CLI arguments: " + ", ".join(missing_arguments)
            )
        common = {
            "tokenizer_path": args.tokenizer,
            "expected_tokenizer_sha256": args.expected_tokenizer_sha256,
            "parent_checkpoint_sha256": args.parent_checkpoint_sha256,
            "replication_source": args.replication_source,
            "expected_replication_source_sha256": args.expected_replication_source_sha256,
            "expected_source_bindings_sha256": args.expected_source_bindings_sha256,
            "expected_runtime_bindings_sha256": args.expected_runtime_bindings_sha256,
            "mode": args.mode,
            "seed": args.seed,
            "train_per_cell": args.train_per_cell,
            "development_per_cell": args.development_per_cell,
            "lane_length": args.lane_length,
        }
        if args.out_dir is not None:
            if any(
                value is not None
                for value in (
                    args.external_manifest,
                    args.expected_external_manifest_sha256,
                    args.expected_external_manifest_bytes,
                )
            ):
                raise ContractError("generation does not accept verification receipts")
            result = build_bundle(out_dir=args.out_dir, **common)
        else:
            if (
                args.external_manifest is None
                or args.expected_external_manifest_sha256 is None
                or args.expected_external_manifest_bytes is None
            ):
                raise ContractError(
                    "verification requires external manifest path, SHA-256, and bytes"
                )
            result = verify_bundle(
                args.verify,
                external_manifest_path=args.external_manifest,
                expected_external_manifest_sha256=args.expected_external_manifest_sha256,
                expected_external_manifest_bytes=args.expected_external_manifest_bytes,
                **common,
            )
    except ContractError as error:
        raise SystemExit("contract error: {}".format(error)) from error
    _assert_bound_runtime_exports()
    print(_BOUND_JSON_DUMPS(result, sort_keys=True))


def _iter_generator_class_callables() -> Iterable[tuple[str, Any, str, Any, Any]]:
    for class_name, class_value in sorted(globals().items()):
        if not (
            isinstance(class_value, type)
            and getattr(class_value, "__module__", None) == __name__
        ):
            continue
        for attribute, descriptor in sorted(class_value.__dict__.items()):
            if type(descriptor) is _PYTHON_FUNCTION_TYPE:
                yield (
                    class_name + "." + attribute,
                    class_value,
                    attribute,
                    descriptor,
                    descriptor,
                )
            elif isinstance(descriptor, property):
                for accessor_name in ("fget", "fset", "fdel"):
                    accessor = getattr(descriptor, accessor_name)
                    if type(accessor) is _PYTHON_FUNCTION_TYPE:
                        yield (
                            class_name + "." + attribute + "." + accessor_name,
                            class_value,
                            attribute,
                            descriptor,
                            accessor,
                        )


def _replace_generator_function_references(
    value: Any, replacements: dict[Any, Any]
) -> Any:
    if type(value) is _PYTHON_FUNCTION_TYPE:
        return replacements.get(value, value)
    if type(value) is tuple:
        return tuple(
            _replace_generator_function_references(item, replacements) for item in value
        )
    if type(value) is list:
        return [
            _replace_generator_function_references(item, replacements) for item in value
        ]
    if type(value) is dict:
        return {
            key: _replace_generator_function_references(item, replacements)
            for key, item in value.items()
        }
    return value


def _clone_generator_function(value: Any, module_globals: dict[str, Any]) -> Any:
    clone = _PYTHON_FUNCTION_TYPE(
        value.__code__,
        module_globals,
        value.__name__,
        value.__defaults__,
        value.__closure__,
    )
    clone.__kwdefaults__ = (
        None if value.__kwdefaults__ is None else dict(value.__kwdefaults__)
    )
    clone.__qualname__ = value.__qualname__
    clone.__module__ = value.__module__
    clone.__doc__ = value.__doc__
    clone.__annotations__ = dict(value.__annotations__)
    clone.__dict__.update(value.__dict__)
    return clone


def _freeze_generator_builtins(
    module_globals: dict[str, Any],
) -> tuple[Any, tuple[tuple[str, Any], ...]]:
    frozen_builtins = _FrozenReviewedGlobals(
        "pipeline.generate_dws_single_completion_v1 builtins",
        dict(builtins_module.__dict__),
    )
    frozen_builtins.seal()
    expected_items = tuple(frozen_builtins.items())
    module_globals["__builtins__"] = frozen_builtins

    originals = {
        value
        for value in module_globals.values()
        if type(value) is _PYTHON_FUNCTION_TYPE
        and getattr(value, "__module__", None) == __name__
    }
    replacements = {
        value: _clone_generator_function(value, module_globals) for value in originals
    }
    for clone in replacements.values():
        clone.__defaults__ = _replace_generator_function_references(
            clone.__defaults__, replacements
        )
        clone.__kwdefaults__ = _replace_generator_function_references(
            clone.__kwdefaults__, replacements
        )
    for name, value in tuple(module_globals.items()):
        if type(value) is _PYTHON_FUNCTION_TYPE and value in replacements:
            module_globals[name] = replacements[value]

    for class_value in (_SealedRuntimeClassMeta, FrozenTokenizer, _PinnedDirectory):
        for name, descriptor in tuple(class_value.__dict__.items()):
            if type(descriptor) is _PYTHON_FUNCTION_TYPE:
                clone = _clone_generator_function(descriptor, module_globals)
                clone.__defaults__ = _replace_generator_function_references(
                    clone.__defaults__, replacements
                )
                clone.__kwdefaults__ = _replace_generator_function_references(
                    clone.__kwdefaults__, replacements
                )
                type.__setattr__(class_value, name, clone)
            elif isinstance(descriptor, property) and descriptor.fget is not None:
                getter = _clone_generator_function(descriptor.fget, module_globals)
                getter.__defaults__ = _replace_generator_function_references(
                    getter.__defaults__, replacements
                )
                type.__setattr__(
                    class_value,
                    name,
                    property(
                        getter, descriptor.fset, descriptor.fdel, descriptor.__doc__
                    ),
                )
    for class_value in (FrozenTokenizer, _PinnedDirectory):
        type.__setattr__(class_value, "_runtime_descriptors_sealed", True)
    return frozen_builtins, expected_items


(
    _FROZEN_GENERATOR_BUILTINS,
    _BOUND_GENERATOR_BUILTIN_ITEMS,
) = _freeze_generator_builtins(globals())


_BOUND_MODULE_GETATTRIBUTE = type(_BOUND_GENERATOR_MODULE).__getattribute__
_BOUND_MODULE_SETATTR = type(_BOUND_GENERATOR_MODULE).__setattr__
_BOUND_MODULE_DELATTR = type(_BOUND_GENERATOR_MODULE).__delattr__
_PROTECTED_GENERATOR_GLOBAL_NAMES = frozenset(
    set(globals())
    | {
        name
        for name, value in globals().items()
        if (
            name.startswith(
                (
                    "_BOUND_",
                    "_FROZEN_",
                    "_REVIEWED_",
                    "_FILESYSTEM_",
                    "_GENERATOR_",
                    "_PACK_",
                    "_RUNTIME_",
                )
            )
            or (
                (type(value) is _PYTHON_FUNCTION_TYPE or isinstance(value, type))
                and getattr(value, "__module__", None) == __name__
            )
        )
    }
    | {
        "apply_microstep",
        "canonical_state",
        "initial_state",
        "parse_answer",
        "parse_state",
        "rows_from_episode",
        "state_answer",
        "_write_sealed_file_at",
        "_fsync_directory_fd",
        "_FrozenReviewedGlobals",
        "_FrozenReviewedGlobalsMeta",
        "_PROTECTED_GENERATOR_GLOBAL_NAMES",
        "_GENERATOR_RUNTIME_MUTATION_GUARD",
        "_SEALED_MODULE_MUTATION_GUARD",
        "_SealedGeneratorModule",
        "_assert_generator_runtime",
        "_generator_runtime_receipt",
        "__builtins__",
        "__file__",
        "__name__",
        "__spec__",
    }
)


class _SealedGeneratorModule(type(_BOUND_GENERATOR_MODULE)):
    def __getattribute__(
        self,
        name: str,
        _getattribute: Any = _BOUND_MODULE_GETATTRIBUTE,
        _mapping_proxy: Any = MappingProxyType,
    ) -> Any:
        value = _getattribute(self, name)
        if name == "__dict__":
            return _mapping_proxy(value)
        return value

    def __setattr__(
        self,
        name: str,
        value: Any,
        _protected: frozenset[str] = _PROTECTED_GENERATOR_GLOBAL_NAMES,
        _setattr: Any = _BOUND_MODULE_SETATTR,
        _error: Any = ContractError,
    ) -> None:
        if name in _protected:
            raise _error(
                "protected generator runtime binding mutation rejected: " + name
            )
        _setattr(self, name, value)

    def __delattr__(
        self,
        name: str,
        _protected: frozenset[str] = _PROTECTED_GENERATOR_GLOBAL_NAMES,
        _delattr: Any = _BOUND_MODULE_DELATTR,
        _error: Any = ContractError,
    ) -> None:
        if name in _protected:
            raise _error(
                "protected generator runtime binding mutation rejected: " + name
            )
        _delattr(self, name)


_BOUND_SEALED_MODULE_METHODS = tuple(
    (
        name,
        _SealedGeneratorModule.__dict__[name],
        _capture_callable_implementation(_SealedGeneratorModule.__dict__[name]),
    )
    for name in ("__getattribute__", "__setattr__", "__delattr__")
)


def _assert_sealed_generator_module(
    _module: Any = _BOUND_GENERATOR_MODULE,
    _module_type: Any = _SealedGeneratorModule,
    _methods: tuple[tuple[str, Any, tuple[Any, ...]], ...] = (
        _BOUND_SEALED_MODULE_METHODS
    ),
    _error: Any = ContractError,
) -> None:
    if type(_module) is not _module_type:
        raise _error("executing generator module mutation boundary changed")
    for name, expected_method, implementation in _methods:
        live_method = _module_type.__dict__.get(name)
        if live_method is not expected_method:
            raise _error("generator module mutation boundary method changed: " + name)
        _assert_callable_implementation(
            "generator module mutation boundary " + name,
            live_method,
            implementation,
        )


def _make_generator_runtime_boundary(
    module_functions: tuple[tuple[str, Any, tuple[Any, ...] | None, str], ...],
    class_methods: tuple[tuple[str, Any, str, Any, Any, Any, tuple[Any, ...]], ...],
    *,
    module_globals: dict[str, Any],
    generator_builtins: Any,
    assert_implementation: Any,
    callable_receipt: Any,
    error_type: Any,
) -> tuple[Any, Any]:
    def assert_runtime() -> None:
        for name, expected_function, implementation, _ in module_functions:
            if module_globals.get(name) is not expected_function:
                raise error_type("generator runtime callable changed: " + name)
            if expected_function.__builtins__ is not generator_builtins:
                raise error_type("generator runtime builtins changed: " + name)
            if implementation is not None:
                assert_implementation(
                    "generator runtime " + name,
                    expected_function,
                    implementation,
                )
        for (
            label,
            owner,
            attribute,
            expected_descriptor,
            expected_function,
            expected_builtins,
            implementation,
        ) in class_methods:
            if owner.__dict__.get(attribute) is not expected_descriptor:
                raise error_type("generator runtime class descriptor changed: " + label)
            assert_implementation(
                "generator runtime " + label,
                expected_function,
                implementation,
            )
            if expected_function.__builtins__ is not expected_builtins:
                raise error_type("generator class builtins changed: " + label)

    def receipt() -> dict[str, Any]:
        return {
            "module_functions": {
                name: {
                    **callable_receipt(name, value),
                    "code_sha256": stable_code_sha256,
                    "private_frozen_builtins": True,
                }
                for name, value, _, stable_code_sha256 in module_functions
            },
            "class_methods": {
                label: {
                    **callable_receipt(label, function),
                    "private_frozen_builtins": builtins_value is generator_builtins,
                }
                for label, _, _, _, function, builtins_value, _ in class_methods
            },
            "module_global_identity_required": True,
            "class_descriptor_identity_required": True,
            "live_code_defaults_and_closures_required": True,
        }

    return assert_runtime, receipt


_BOUND_GENERATOR_MODULE.__class__ = _SealedGeneratorModule
_BOUND_GENERATOR_MODULE_FUNCTIONS = tuple(
    (
        name,
        value,
        None if value is runtime_bindings else _capture_callable_implementation(value),
        sha256_bytes(_stable_code_bytes(value.__code__)),
    )
    for name, value in sorted(globals().items())
    if type(value) is _PYTHON_FUNCTION_TYPE
    and getattr(value, "__module__", None) == __name__
)


_BOUND_GENERATOR_CLASS_METHODS = tuple(
    (
        label,
        class_value,
        attribute,
        descriptor,
        function,
        function.__builtins__,
        _capture_callable_implementation(function),
    )
    for label, class_value, attribute, descriptor, function in (
        _iter_generator_class_callables()
    )
)
(
    _assert_generator_runtime,
    _generator_runtime_receipt,
) = _make_generator_runtime_boundary(
    _BOUND_GENERATOR_MODULE_FUNCTIONS,
    _BOUND_GENERATOR_CLASS_METHODS,
    module_globals=globals(),
    generator_builtins=_FROZEN_GENERATOR_BUILTINS,
    assert_implementation=_assert_callable_implementation,
    callable_receipt=_bound_callable_receipt,
    error_type=ContractError,
)
runtime_bindings.__defaults__ = (
    _assert_generator_runtime,
    _generator_runtime_receipt,
)


_GENERATOR_RUNTIME_MUTATION_GUARD = _install_callable_mutation_guard(
    "complete-generator-runtime-v1",
    (
        *(value for _, value, _, _ in _BOUND_GENERATOR_MODULE_FUNCTIONS),
        *(function for _, _, _, _, function, _, _ in _BOUND_GENERATOR_CLASS_METHODS),
        _assert_generator_runtime,
        _generator_runtime_receipt,
    ),
)
_SEALED_MODULE_MUTATION_GUARD = _install_callable_mutation_guard(
    "sealed-generator-module-v1",
    (
        _SealedGeneratorModule.__dict__["__setattr__"],
        _SealedGeneratorModule.__dict__["__delattr__"],
        _assert_sealed_generator_module,
    ),
)


if __name__ == "__main__":
    main()
