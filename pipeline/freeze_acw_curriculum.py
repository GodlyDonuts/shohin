"""Replay-verify ACW data and freeze reproducible public trainer curricula.

Canonical pilot and development datasets are regenerated from their registered
public seed material before use.  A pilot schedule is frozen only after two
isolated executions produce byte-identical reports and schedules.
"""

from __future__ import annotations

import sys

# Canonical jobs run with -S -P and execute this file directly. Remove CPython's
# optional stdlib ZIP entry before any non-builtin import can resolve through it.
if sys.flags.no_site and sys.flags.safe_path:
    sys.path[:] = [entry for entry in sys.path if not entry.endswith(".zip")]

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import secrets
import shlex
import shutil
import socket
import subprocess
import sysconfig
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.acw_hidden_basis_training import (
    ACW_SCIENTIFIC_PATHS,
    CANONICAL_BUNDLE_BLOCK,
    CONFIRMATION_COMMITMENTS,
    PUBLIC_ARRAYS,
    Curriculum,
    PublicTrainingData,
    canonical_json_bytes,
    curriculum_query_schedule_sha256,
    file_sha256,
    forward_logits,
    initialized_model_for_arm,
    load_public_training_data,
    recurrent_state,
    scientific_identity,
)
from pipeline.generate_acw_hidden_basis import (
    ADAPTATION_HISTORIES as GENERATOR_ADAPTATION_HISTORIES,
    DEVELOPMENT_SEEDS,
    EVALUATION_DEPTHS as GENERATOR_EVALUATION_DEPTHS,
    EVALUATION_HISTORIES as GENERATOR_EVALUATION_HISTORIES,
    GENERATOR_PROTOCOL,
    PILOT_SEED as GENERATOR_PILOT_SEED,
    TRAIN_HISTORIES as GENERATOR_TRAIN_HISTORIES,
    development_seed_material,
    generate_dataset,
)


_TORCH_GENERATED_MODULE_FILENAMES = {
    "_remote_module_non_scriptable": "_remote_module_non_scriptable.py",
}


def _detach_torch_generated_import_path() -> dict[str, Path]:
    """Close PyTorch's incidental RPC template directory as an import source."""
    instantiator = sys.modules.get("torch.distributed.nn.jit.instantiator")
    if instantiator is None:
        return {}
    raw_root = getattr(instantiator, "INSTANTIATED_TEMPLATE_DIR_PATH", None)
    if not isinstance(raw_root, str) or not raw_root:
        raise RuntimeError("PyTorch generated-module directory is unavailable")
    root = Path(raw_root).resolve()
    paths = {}
    for name, filename in _TORCH_GENERATED_MODULE_FILENAMES.items():
        module = sys.modules.get(name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise RuntimeError("PyTorch generated module is unavailable")
        path = Path(raw_path).resolve()
        if path.parent != root or path.name != filename or not path.is_file():
            raise RuntimeError("PyTorch generated module escaped its temporary root")
        paths[name] = path
    entry = str(root)
    if sys.path.count(entry) > 1:
        raise RuntimeError("PyTorch generated import path has unexpected multiplicity")
    if entry in sys.path:
        sys.path.remove(entry)
    return paths


_DETACHED_TORCH_GENERATED_MODULES: dict[str, Path] = {}


PILOT_PROTOCOL = "R12-ACW-CGBR-PILOT-v5"
SCHEDULE_PROTOCOL = "R12-ACW-QUERY-SCHEDULE-v3"
BUNDLE_PROTOCOL = "R12-ACW-TRAINER-BUNDLE-v4"
DATA_REPLAY_PROTOCOL = "R12-ACW-DATA-REPLAY-v1"
PILOT_EXECUTION_PROTOCOL = "R12-ACW-PILOT-REPLAY-EXECUTION-v5"
PILOT_COMPARISON_PROTOCOL = "R12-ACW-PILOT-REPLAY-COMPARISON-v5"
PILOT_ORCHESTRATION_PROTOCOL = "R12-ACW-PILOT-ORCHESTRATION-v2"
PILOT_INDEPENDENT_VERIFICATION_PROTOCOL = "R12-ACW-PILOT-INDEPENDENT-VERIFICATION-v2"
PILOT_ARTIFACT_REGISTRY_PROTOCOL = "R12-ACW-PILOT-ARTIFACT-REGISTRY-v1"
PILOT_SEED = 2026071600
UNIFORM_SEED = 2026071604
PUBLIC_QUERIES = 24
REFINEMENT_ROUNDS = 12
MAX_GROUPS_PER_ROUND = 512
MAX_CANDIDATE_EVALUATIONS = 147_456
CANONICAL_HISTORIES = 4096
CANONICAL_LABELS = 57_344
CANONICAL_UPDATES_PER_ROUND = 200
CANONICAL_FINAL_UPDATES = 800
CANONICAL_BATCH_SIZE = 256
CANONICAL_TOTAL_UPDATES = (
    REFINEMENT_ROUNDS + 1
) * CANONICAL_UPDATES_PER_ROUND + CANONICAL_FINAL_UPDATES
CANONICAL_PILOT_DATASET_PAYLOAD_SHA256 = (
    "3294a0d12d277f46ea8c0cbf50142be14816447c15bc3792f6e4df7e77e2ba33"
)
CANONICAL_PILOT_BASE = "/lustre/fs1/home/sa305415/shohin_acw"
CANONICAL_PILOT_SITE_PACKAGES = (
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages"
)
CANONICAL_PILOT_THREAD_ENV = {
    "ATEN_CPU_CAPABILITY": "avx2",
    "CUDA_VISIBLE_DEVICES": "",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "KMP_DETERMINISTIC_REDUCTION": "TRUE",
    "MKL_CBWR": "AVX2",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPYCACHEPREFIX": "/tmp/shohin-acw-pycache",
}
CANONICAL_PILOT_STATIC_ENV = {
    **CANONICAL_PILOT_THREAD_ENV,
    "HOME": "/home/sa305415",
    "LANG": "C",
    "LC_ALL": "C",
    "LOGNAME": "sa305415",
    "PATH": "/apps/slurm/current/bin:/usr/bin:/bin",
    "PYTHONPATH": f"{CANONICAL_PILOT_BASE}:{CANONICAL_PILOT_SITE_PACKAGES}",
    "PYTHONUTF8": "1",
    "SHELL": "/bin/bash",
    "TZ": "UTC",
    "USER": "sa305415",
}
CANONICAL_PILOT_DYNAMIC_ENV_KEYS = {
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURM_SUBMIT_DIR",
}
CANONICAL_PILOT_UID = 1_227_834_669
CANONICAL_PILOT_SCONTROL = "/apps/slurm/current/bin/scontrol"
CANONICAL_PILOT_ROLES = {
    "producer": {
        "command": f"{CANONICAL_PILOT_BASE}/pipeline/jobs/run_acw_pilot_stokes.sbatch",
        "hostname": "ec51.ucfarcc.org",
        "job_name": "shohin-acw-pilot",
        "node_list": "ec51",
        "scientific_path": "pipeline/jobs/run_acw_pilot_stokes.sbatch",
        "stdout_prefix": f"{CANONICAL_PILOT_BASE}/logs/acw_pilot_",
    },
    "verifier": {
        "command": f"{CANONICAL_PILOT_BASE}/pipeline/jobs/verify_acw_pilot_stokes.sbatch",
        "hostname": "ec52.ucfarcc.org",
        "job_name": "shohin-acw-verify",
        "node_list": "ec52",
        "scientific_path": "pipeline/jobs/verify_acw_pilot_stokes.sbatch",
        "stdout_prefix": f"{CANONICAL_PILOT_BASE}/logs/acw_pilot_verify_",
    },
}
CANONICAL_PILOT_NATIVE_FILES = {
    "/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13": {
        "bytes": 33010184,
        "sha256": "051a031d827eab9778e982571db754662809164c8a3ec01e9beea1e1088123e0",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libbz2.so.1.0.8": {
        "bytes": 241888,
        "sha256": "cc570bce44ed3ab1b0f480bdb95c04e8224432811bfb5a55b533135a6001a03b",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libcrypto.so.3": {
        "bytes": 7202968,
        "sha256": "2edb947628da5f5e7f1c4a6e11d38b3c136a71f5a4040f013d46eebbbe958f5f",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libffi.so.8.2.0": {
        "bytes": 50744,
        "sha256": "cff0ea6932fe2986f3a33410d2008692cc93caecfde282a2b2a961f7963a13df",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libgcc_s.so.1": {
        "bytes": 902640,
        "sha256": "e1e904051f77f9569c2ea53c83bb4083c26575e0fbd4010e46f1cb8b21037ad1",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/liblzma.so.5": {
        "bytes": 222712,
        "sha256": "07dceced575343c83860aedde6e7e2ac5deb7a0fa31b0f195c44388544817abc",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libmpdec.so.4.0.0": {
        "bytes": 232000,
        "sha256": "6be97fc47c07871b522613d575190b7e454e4f2ba7b85f4aabe812fb0ea93fc1",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libssl.so.3": {
        "bytes": 1202328,
        "sha256": "e5822ab6dd9aaa1712428f81757b44f26370c0b8ba9be07cb0aa95b96120b4fb",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libstdc++.so.6.0.34": {
        "bytes": 21295144,
        "sha256": "9581ad615b7c073423f57b69a3b148a89f8ea76fc909124211f9007909b807a6",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libuuid.so.1.3.0": {
        "bytes": 40720,
        "sha256": "fdf6282117d3443d7d3d3576a8bea9f53fd6d0cea6ac0141556f0cedd52b52d2",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/libz.so.1.3.2": {
        "bytes": 117128,
        "sha256": "22f1601237b86f0f48ed5b83071d1505167ae2e16365b33b4eed6e96dbf71ab0",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_asyncio.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 377568,
        "sha256": "409ed340ba17fa1233ed2b00502fa1ca37d580ce7d25320ef5f925b83b7d9240",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_bisect.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 80200,
        "sha256": "3765246a004bfd648effcd9cb9f17b575021b28c37053f1477f49389fb161ed5",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_blake2.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 398256,
        "sha256": "7c84c7c88da1b5037e4f2b457ec26b2a88768d5a78e5e565e0fe298efdc3b92e",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_bz2.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 98304,
        "sha256": "10e69ee5b149579b708701ae42e10950c441cbb52a65ca4062a9535b33dc4e0a",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_contextvars.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 27232,
        "sha256": "17e9acb56365c794841bb1ae965b626f49350633ba027ee7236d6e9d130f47b9",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_csv.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 144280,
        "sha256": "0226f428a17b659a69f11ee8bd004d7026cb972a44068874386cba60466560e7",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_ctypes.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 947056,
        "sha256": "48c09aba99ec50270d8045a4e520521ec9877d99efd02ca7e4260afb5a8c3dce",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_datetime.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 785552,
        "sha256": "945a66b3d688205f6e52d39b4d28cdf3eec456d3ff47a33472e356f411659959",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_decimal.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 910272,
        "sha256": "a3697fcabda8a9c1490ff71e5a5a3542ee2cfbeb4cfaa85b40b5e6ade153f43b",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_hashlib.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 235704,
        "sha256": "d3757cb76f2b7bde4159a7d8d8a3feb90589c774c6fcd61bb3d68835051cebae",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_heapq.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 112152,
        "sha256": "cfd2ed30a58d1a17c4a9ccb7e6d23110e834333a853a79622e78cbbfa4d5afcd",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_json.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 299240,
        "sha256": "daa1cba1d763921be9a0ffb483f5b1fba957147f30a3fd6607bf8e88b2ea07ca",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_lsprof.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 207240,
        "sha256": "bd05fb9d0049c30528696b54673bbc743e07b85a5100458e124bf3b5ffb36666",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_lzma.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 161608,
        "sha256": "a8e68ad473127c474ce64b0acd916b8f3c3af27f095432dd48e7a2d669bf39fc",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_multiprocessing.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 162736,
        "sha256": "78185767dc6f73f82c3c8bf68285a903b3275e0dbb0be7355e4e55f300b4f0e7",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_opcode.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 80784,
        "sha256": "0d362a76c66fb8e46f1f816fe6c3d3024b296873fea2f033fcdadda4c8891be2",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_pickle.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 686912,
        "sha256": "d866340b4f12a251675eea92cbb1daf241717e1b14ae6cf1f92b05f315b1d397",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_posixsubprocess.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 188024,
        "sha256": "268fd6911d48f97edd9f7347a4e3b61959b9219c147c8918161533d992a82029",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_queue.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 58576,
        "sha256": "a3f63050f9603b4964c06e9f9e6b39956ed908b53ea470a5d9e59563f6008890",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_random.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 62864,
        "sha256": "c3bd19bb9cf399ffacfbcb91677015e78fc487bf35ef9b845893c40312557b72",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_socket.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 316504,
        "sha256": "d98a78aabaec99de7e0834ca9253cf32a676655d02e388bb9a72a4006fc5d796",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_ssl.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 635584,
        "sha256": "7fe771e5feec21f35e840d142ed97e0c17b3036048c10123fdbbc00f130add1d",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_struct.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 221552,
        "sha256": "64f02c66a66d2a95841c52f9afc72915e6fba25a21f790deeedac305278d7967",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/_uuid.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 21592,
        "sha256": "f299b660471172b1c8095b999b085520ecd824592f6f7d5c3d8bd279f8b60bb5",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/array.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 267864,
        "sha256": "79adeb4256ee02446bc7c5752971f008e7a48816661ee27989efe8ac33f97fd8",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/binascii.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 193328,
        "sha256": "c4f35edd281989ea959472b122f3611a7bfb0ec3c634ffabde82b4d1cc7eee27",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/cmath.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 167864,
        "sha256": "c4b4780cc553740549e09befd9c071838d4121e9cb3933a8324d06de7c6f7c96",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/fcntl.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 53872,
        "sha256": "3ea9695d525a7a8e05ed1812390ef35c79f89e2a1b545fefb57f8814dda500c2",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/grp.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 56384,
        "sha256": "4bd43d35ab5640b8f23366a9d23d34e971021108be52c2f988e16d3688565ae3",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/math.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 397688,
        "sha256": "52f40e048493a0c3584991428e61b46302724cb94abe088420427aea8e82dc18",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/mmap.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 196416,
        "sha256": "761f0f1aac90252cad4b546cbae001756a05224fea6f2f49a53c64397249aa03",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/resource.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 62520,
        "sha256": "2fa41784614c4b8087071a3784d34cde9f5e9cf2e6848b8538dad5bb326dadec",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/select.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 210976,
        "sha256": "d291dba384ebe34603d937fd5b508351a69653bb8e84640714f9bef5411a1769",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/termios.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 64744,
        "sha256": "762963d0871875305b92787e8400cf615145842ff61187b66de15dadb495d2bf",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/unicodedata.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 820088,
        "sha256": "d2e692fdb89c39ca595581bfcd055d5431963640aa86a335dbb4122da1d6f720",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload/zlib.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 165040,
        "sha256": "e5a2756a813520d27aedc4507f2a097ef7470189410cef87fb4d2ed42da61480",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/cusparselt/lib/libcusparseLt.so.0": {
        "bytes": 212602113,
        "sha256": "42385f413fe87ce1c844b13890f920b66873dbd36030a3b0fd2c143d722f6879",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy.libs/libgfortran-83c28eba-b4027c22.so.5.0.0": {
        "bytes": 2862249,
        "sha256": "fd1eab02b28ddb97628ca82220f0141336e390990d1d92ae6a3475aaf4f51b03",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy.libs/libquadmath-2284e583-a9307bba.so.0.0.0": {
        "bytes": 275553,
        "sha256": "6bc3069003caec1f075be3c51ae8355332f081fa9465910c4b5724e7fe7d8ad4",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy.libs/libscipy_openblas64_-017048f4.so": {
        "bytes": 25128625,
        "sha256": "7f88e5cb3075a73d1610511f5de871c93e11701e862341c002ec8db936a60af8",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 10674025,
        "sha256": "4aa06b610683748bfd50b3ffbe42042c1d96cc138e0d2510f0e61cab642f5964",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/fft/_pocketfft_umath.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 539312,
        "sha256": "37eea6cf4de15ba3cc9487e704bbd306de65bf804310b1067f1b11d1e50a9de8",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/linalg/_umath_linalg.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 231801,
        "sha256": "9d2152c4c9600b5f7b93a7f07acd0a75cb0f4612604bab4e3041cbdd968712b5",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_bounded_integers.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 320016,
        "sha256": "0174ea23618432fdd71de80ae82c0032185a191beda037ae8b9b5753487953a4",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_common.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 257624,
        "sha256": "1df8f9da07e9b611f3cd5ba10802598661a0abba25740f1402e0364521ae7afc",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_generator.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 813560,
        "sha256": "7787a0c14a52342f5badc650e5c13a403096b426923ee2b4f1c80d15fc8629d9",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_mt19937.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 130232,
        "sha256": "f7507e9c958497ad07d7888da7dbb50b15ccd641df254fa0db497c4b29baa1d4",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_pcg64.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 141216,
        "sha256": "e598ffba681794207920c16edea155524f9ab1a3a717144de6fc25dcfb3e3e46",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_philox.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 117856,
        "sha256": "d9dc366f8b47ae0328e91e13d814c3eec5d4f5965e93455bcd2418c6f3c5009b",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/_sfc64.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 87344,
        "sha256": "f1404f43e749730641006e04f6cab20b8e39c46f50dad6cacf93f6467834e7a5",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/bit_generator.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 227192,
        "sha256": "17184894110ce0a6cf72f201a8ee13b9e9ff6a6e5ab7427f8961311d0d8bf0a8",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy/random/mtrand.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 631832,
        "sha256": "b241fe5be5ea2c61a0de11bfeec30dbddeb903cce85123d0a7790103713795cb",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cublas/lib/libcublas.so.12": {
        "bytes": 109604768,
        "sha256": "4cc45526449a95d3985389785e985b7832460ea3db178bc9e9963be2111fbee8",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cublas/lib/libcublasLt.so.12": {
        "bytes": 441938896,
        "sha256": "44a813aa2da08830f9083f81d0eb73f1ae4052a4d9b0b0de480a8f6cd9eb3078",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12": {
        "bytes": 7748112,
        "sha256": "fb2a7c5b15c84df9505dd47e553fe46f3121a57d30391fca24179d202f73f3f7",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12": {
        "bytes": 60418376,
        "sha256": "466d6f14d6cfde99838a132cfa7a23f90ba08acfadaa4d9e4c9defe1277f7bf1",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12": {
        "bytes": 707904,
        "sha256": "8774224f5b11a73b15d074a3fcce7327322c5c4cfdfd924d6a826779eec968fe",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cudnn/lib/libcudnn.so.9": {
        "bytes": 104664,
        "sha256": "74a4c495a26d5104150241e84a10842442c024c129cdc274ad8b6a7a8794980e",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cufft/lib/libcufft.so.11": {
        "bytes": 292889192,
        "sha256": "f3921c4133925242459ebfdbf7901db4ebfae6bbcfff9dc5ce53b1ff13fb42ff",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/curand/lib/libcurand.so.10": {
        "bytes": 96525744,
        "sha256": "dab8074b610b82a863a42eceda788e9b08364b545bab948509306b48c46018cf",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/cusparse/lib/libcusparse.so.12": {
        "bytes": 281313984,
        "sha256": "cb4288c965453a8020653c0fd92cdfcd12bb5a646843d663e9d4e5f520f8915c",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2": {
        "bytes": 251616632,
        "sha256": "78df2f31f6db8142ec546a1e5a31cb066f7892d12d2f665b448f8069a08ef807",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12": {
        "bytes": 53594512,
        "sha256": "cbd1488bbef82b8d64c2dd51a9b6aef3f6b0bc3a7a80a9821be48d1c07f700e7",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/_C.cpython-313-x86_64-linux-gnu.so": {
        "bytes": 33785,
        "sha256": "c16192409389020e9f456d4a8a3e29cc5d82d7af81608b1f41bcc25f3acc26bc",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libc10.so": {
        "bytes": 1455209,
        "sha256": "9e8b46af9a5b5f1bf3200f2aabc3376597f1fb4787a179428e8c91f7253af78d",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libc10_cuda.so": {
        "bytes": 720929,
        "sha256": "99caf7bd0183e53103a28a0d27d6c54e8be338dcf71e4444e51b1c9bf529ef11",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libgomp-a34b3233.so.1": {
        "bytes": 169113,
        "sha256": "aa9c09dd9d3b8f42b355048284f2894f1b4023f676843a69084276e754914206",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libshm.so": {
        "bytes": 52793,
        "sha256": "1bcc0868eaa8b49a3ae062d3dbb92f57e2fae521cbedf459a8cc0aeb887914a4",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libtorch.so": {
        "bytes": 196201,
        "sha256": "a5d97e6bd79ef3bed0d038f9847c89441b85f886d8ccce7e5ec96fa344f0381c",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so": {
        "bytes": 441856673,
        "sha256": "a7d7ca8ee0d1317d46a8e1491efe95fa898f7c9805436010748ed26f3b1e02d1",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libtorch_cuda.so": {
        "bytes": 902652937,
        "sha256": "06997c4af17a0f82b01b19c32b54da396f84ce165ec02c7589fab732e7bfc1d2",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libtorch_global_deps.so": {
        "bytes": 21193,
        "sha256": "4c08044712a9f1193d61f3d1449d0d82558c41baa2ec77aa4000d73be7c2e0ee",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch/lib/libtorch_python.so": {
        "bytes": 29557777,
        "sha256": "1edf4e5012432b7c4576ae6c4c31f1d3ff26eb5358b808628d1116d0df49643b",
    },
    "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/triton/_C/libtriton.so": {
        "bytes": 289921448,
        "sha256": "dbb29147681bfa6df112de722000d825b3bb9ff569081f23b5984b42815c6aa8",
    },
    "/usr/lib64/ld-2.28.so": {
        "bytes": 1104088,
        "sha256": "f54e08528da407525c1bf95d06b4b6426cedb835f018d1c596880fda4e308740",
    },
    "/usr/lib64/libc-2.28.so": {
        "bytes": 2164744,
        "sha256": "7d3b8e8cf41b2d8a63841469400a64c247135f5210a95c81c6e4993e4c736ffa",
    },
    "/usr/lib64/libdl-2.28.so": {
        "bytes": 19128,
        "sha256": "414cca30a2b3f41d64d1b67fc987552dc2cca649d73eac3a3a5099d76363834c",
    },
    "/usr/lib64/libm-2.28.so": {
        "bytes": 1599096,
        "sha256": "e27d5b1be7b305214bc91e78d41c55964365cb9d06cec814e6a0b61c79d8ddba",
    },
    "/usr/lib64/libpthread-2.28.so": {
        "bytes": 149936,
        "sha256": "0239064fff600cd4c9fe5bc8faaddfeb23dfd8444c8882ed0c95aa2edca5b985",
    },
    "/usr/lib64/librt-2.28.so": {
        "bytes": 42744,
        "sha256": "ab764837fc63ad7ff48249b9a736df3b37d36033b0a43af712f1ce87091329e0",
    },
    "/usr/lib64/libutil-2.28.so": {
        "bytes": 17032,
        "sha256": "e3f9a9ad55b94be376da0b7e9360821045f0a9d0339d95e615ace176176e0fe4",
    },
}
CANONICAL_PILOT_CODE_TREES = {
    "numpy": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy",
    "python_stdlib": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13",
    "torch": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch",
}
CANONICAL_PILOT_EXTERNAL_EXECUTABLE_PATHS = (
    "/bin/bash",
    "/usr/bin/chmod",
    "/usr/bin/env",
    "/usr/bin/git",
    "/usr/bin/install",
    "/usr/bin/mkdir",
    "/bin/ps",
    "/usr/bin/rm",
    CANONICAL_PILOT_SCONTROL,
)
# Filled only from byte-identical exact-runtime probes after this implementation is
# frozen.  Empty values deliberately make canonical execution fail until then.
CANONICAL_PILOT_CODE_TREE_SUMMARIES: dict[str, dict] = {
    "numpy": {
        "file_count": 1311,
        "payload_sha256": "76c2ffc9504c63288edcd43dec1ad037c082648690778b1b923f3fabdd8f679d",
        "root": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/numpy",
        "total_bytes": 40248116,
    },
    "python_stdlib": {
        "file_count": 2324,
        "payload_sha256": "b38c2de8355cb9828e88a610d190cf4b7a0aa3189ad2478a648b1741241e0343",
        "root": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13",
        "total_bytes": 63015347,
    },
    "torch": {
        "file_count": 12678,
        "payload_sha256": "0fe56bfe9da390770ecec50637fab6db8f9f79b93a59599b27bf7f60ec2ba1c1",
        "root": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/torch",
        "total_bytes": 1585247331,
    },
}
CANONICAL_PILOT_IMPORTED_CODE_SUMMARY: dict = {
    "file_count": 599,
    "payload_sha256": "7cdb8f02b82f8fd5a27b3a68df109731c54c32e9438a5b725ee92640cd8475ed",
    "total_bytes": 302774740,
}
CANONICAL_PILOT_STARTUP_IDENTITY: dict = {
    "purelib": "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages",
    "startup_files": {
        "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/conda-site.pth": {
            "bytes": 141,
            "sha256": "4b1c8e31d36ba334a6b8659b3bf7ca79a1a53f020f09ccf8ba0080a2d1904553",
        },
        "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/distutils-precedence.pth": {
            "bytes": 151,
            "sha256": "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224",
        },
    },
    "startup_files_payload_sha256": "615d8b7093ea3a68bb2391cc63b0119f8f87c9c161d6f9c89be5ff6d75e90227",
    "sys_path": [
        "/lustre/fs1/home/sa305415/shohin_acw",
        "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages",
        "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13",
        "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/lib-dynload",
        "/lustre/fs1/home/sa305415/shohin/miniforge3/lib/python3.13/site-packages/setuptools/_vendor",
    ],
}
CANONICAL_PILOT_EXTERNAL_EXECUTABLES: dict[str, dict] = {
    "/apps/slurm/current/bin/scontrol": {
        "bytes": 10942440,
        "resolved_path": "/apps/slurm/slurm-25.11.0-gcc-12.2.0/bin/scontrol",
        "sha256": "17f565e480394ceb7861b25083348931abb180f1edc1dbb65c39749d7243cf18",
    },
    "/bin/bash": {
        "bytes": 1154680,
        "resolved_path": "/usr/bin/bash",
        "sha256": "5dd8362955fbeb65199b630e9ac22900c9cddccab75adbdfb49812b28030fb34",
    },
    "/bin/ps": {
        "bytes": 137848,
        "resolved_path": "/usr/bin/ps",
        "sha256": "ffe6fb43df4cc59a3858e32fc85ce978eec20e54807d4e1b707fadc2fc1d3c34",
    },
    "/usr/bin/chmod": {
        "bytes": 63688,
        "resolved_path": "/usr/bin/chmod",
        "sha256": "6404e5284b285f9d173e83824ec7e5fac16a4312f15b70e5a37742a3bda27543",
    },
    "/usr/bin/env": {
        "bytes": 42344,
        "resolved_path": "/usr/bin/env",
        "sha256": "89ad78fb31764978187d23e81d4cbdb6840b26e84a44761c3be4fd40e7f652bc",
    },
    "/usr/bin/git": {
        "bytes": 3845928,
        "resolved_path": "/usr/bin/git",
        "sha256": "507917bbb5d24123c8e11df46df1d32483da1ce6420aa7ba7dd17de8ccd13a9e",
    },
    "/usr/bin/install": {
        "bytes": 159912,
        "resolved_path": "/usr/bin/install",
        "sha256": "fbebdb6f067903cf0cd849de8bff399f5b048f27dee5cbcd9ee77b15e03a8891",
    },
    "/usr/bin/mkdir": {
        "bytes": 84680,
        "resolved_path": "/usr/bin/mkdir",
        "sha256": "8e3ba8b0dc320ecfb8a44dabf2ec2a9826b633f13b2731f4becb9f64f8868112",
    },
    "/usr/bin/rm": {
        "bytes": 72064,
        "resolved_path": "/usr/bin/rm",
        "sha256": "8399188fff619a1032449861076002e5df4c0f07af0ff80e33533e22933a2e20",
    },
}
CANONICAL_PILOT_GENERATED_MODULES: dict[str, dict] = {
    "_remote_module_non_scriptable": {
        "bytes": 2355,
        "filename": "_remote_module_non_scriptable.py",
        "sha256": "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8",
    }
}
CANONICAL_PILOT_RUNTIME = {
    "cpu": {
        "available": True,
        "cpu_family": "6",
        "flags_sha256": (
            "493ad981446539efef86dd89fe12dd394fea32ce369d866b03365ad72f04c1a1"
        ),
        "microcode": "0x2007006",
        "model": "85",
        "model_name": "Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz",
        "stepping": "4",
        "vendor_id": "GenuineIntel",
    },
    "deterministic_algorithms": True,
    "code_trees": CANONICAL_PILOT_CODE_TREE_SUMMARIES,
    "imported_external_code": CANONICAL_PILOT_IMPORTED_CODE_SUMMARY,
    "python_startup": CANONICAL_PILOT_STARTUP_IDENTITY,
    "external_executables": CANONICAL_PILOT_EXTERNAL_EXECUTABLES,
    "generated_modules": CANONICAL_PILOT_GENERATED_MODULES,
    "libc": ["glibc", "2.28"],
    "machine": "x86_64",
    "native_files": CANONICAL_PILOT_NATIVE_FILES,
    "native_files_payload_sha256": "13c265e3f116beee105c883e6384595e5759f96419e790c064cd94e77f20425c",
    "numpy_config_sha256": (
        "6a202deb5035843d719b04dbfca97b3fe4191603e5884fac2f9af5659555419b"
    ),
    "numpy_cpu_features_enabled": [
        "AVX",
        "AVX2",
        "AVX512BW",
        "AVX512CD",
        "AVX512DQ",
        "AVX512F",
        "AVX512VL",
        "AVX512_SKX",
        "BMI",
        "BMI2",
        "CX16",
        "F16C",
        "FMA3",
        "LAHF",
        "LZCNT",
        "MMX",
        "MOVBE",
        "POPCNT",
        "SSE",
        "SSE2",
        "SSE3",
        "SSE41",
        "SSE42",
        "SSSE3",
        "X86_V2",
        "X86_V3",
        "X86_V4",
    ],
    "numpy_version": "2.5.0",
    "python_implementation": "CPython",
    "python_no_site": True,
    "python_safe_path": True,
    "python_executable": ("/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13"),
    "python_version": "3.13.13",
    "system": "Linux",
    "static_environment": CANONICAL_PILOT_STATIC_ENV,
    "thread_env": CANONICAL_PILOT_THREAD_ENV,
    "torch_cuda_available": False,
    "torch_cpu_capability": "AVX2",
    "torch_config_sha256": (
        "51bcbe59eb176362dc969b0341d85ca88416e37bd0f10de4b19350d07898e330"
    ),
    "torch_num_interop_threads": 32,
    "torch_num_threads": 1,
    "torch_version": "2.6.0+cu124",
}
CANONICAL_PILOT_DATASET = "artifacts/r12/acw_pilot_domain_v3_runtime_v1"
CANONICAL_PILOT_REPLAY_A = "artifacts/r12/acw_cgbr_pilot_v5_replay_a"
CANONICAL_PILOT_REPLAY_B = "artifacts/r12/acw_cgbr_pilot_v5_replay_b"
CANONICAL_PILOT_OUTPUT = "artifacts/r12/acw_cgbr_pilot_v5"
CANONICAL_PILOT_VERIFICATION = (
    "artifacts/r12/acw_cgbr_pilot_v5_independent_verification"
)
CANONICAL_PILOT_REGISTRY = "R12_ACW_PILOT_ARTIFACT_REGISTRY.json"
CANONICAL_PILOT_ARTIFACT_FILES = 80
CANONICAL_PILOT_ANCHORED_FILES = 81
CANONICAL_PILOT_ACTIVATION_ALLOWLIST = (
    "AGENT_RUNBOOK.md",
    "pipeline/acw_hidden_basis_training.py",
    "pipeline/adjudicate_acw_hidden_basis.py",
    "pipeline/freeze_acw_curriculum.py",
    "pipeline/test_acw_hidden_basis_training.py",
    "pipeline/test_adjudicate_acw_hidden_basis.py",
    "pipeline/test_freeze_acw_curriculum.py",
)
CANONICAL_PILOT_VERIFICATION_CLAIM = (
    "Different-node deterministic replay of the non-scored Track S pilot. "
    "This is not a scored architecture or reasoning result."
)
CANONICAL_PILOT_REGISTRY_CLAIM = (
    "Byte registry for one non-scored Track S pilot and its independent replay. "
    "It authorizes no scored arm, Shohin fit, or reasoning claim."
)

if GENERATOR_PILOT_SEED != PILOT_SEED:
    raise RuntimeError("freezer and generator pilot seed registries differ")


def _pilot_cpu_identity() -> dict:
    if platform.system() != "Linux":
        return {"available": False}
    blocks = Path("/proc/cpuinfo").read_text(errors="strict").split("\n\n")
    fields = {}
    for line in blocks[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    flags = sorted(fields.get("flags", "").split())
    required = ("vendor_id", "cpu family", "model", "model name", "stepping")
    if not flags or any(not fields.get(key) for key in required):
        raise RuntimeError("canonical ACW pilot cannot identify the CPU")
    return {
        "available": True,
        "cpu_family": fields["cpu family"],
        "flags_sha256": hashlib.sha256(" ".join(flags).encode("ascii")).hexdigest(),
        "microcode": fields.get("microcode"),
        "model": fields["model"],
        "model_name": fields["model name"],
        "stepping": fields["stepping"],
        "vendor_id": fields["vendor_id"],
    }


def _warm_pilot_runtime() -> None:
    """Load every numerical path used by the CPU pilot before fingerprinting."""
    numpy_sample = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.matmul(numpy_sample, numpy_sample)
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=0.003,
        weight_decay=0.0001,
    )
    parameter.square().sum().backward()
    optimizer.step()


def _tree_payload_summary(
    root: Path,
    *,
    excluded_top_levels: set[str] | None = None,
) -> dict:
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise RuntimeError("canonical ACW code tree root is not literal")
    excluded_top_levels = excluded_top_levels or set()
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    def tree_files(directory: Path, *, top_level: bool):
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            if top_level and entry.name in excluded_top_levels:
                continue
            if entry.is_symlink():
                raise RuntimeError("canonical ACW code tree contains a symlink")
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                yield from tree_files(path, top_level=False)
            elif entry.is_file(follow_symlinks=False):
                yield path, entry.stat(follow_symlinks=False).st_size
            else:
                raise RuntimeError("canonical ACW code tree contains a special file")

    def file_record(item: tuple[Path, int]) -> dict:
        path, size = item
        return {
            "bytes": size,
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = executor.map(file_record, tree_files(root, top_level=True))
        for record in records:
            size = record["bytes"]
            encoded = canonical_json_bytes(record)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            file_count += 1
            total_bytes += size
    if not file_count:
        raise RuntimeError("canonical ACW code tree is empty")
    return {
        "root": str(root),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "payload_sha256": digest.hexdigest(),
    }


def _pilot_code_tree_summaries() -> dict[str, dict]:
    if Path(sysconfig.get_path("stdlib")).resolve() != Path(
        CANONICAL_PILOT_CODE_TREES["python_stdlib"]
    ):
        raise RuntimeError("canonical ACW Python standard-library root mismatch")
    summaries = {}
    for name, raw_root in sorted(CANONICAL_PILOT_CODE_TREES.items()):
        summaries[name] = _tree_payload_summary(
            Path(raw_root),
            excluded_top_levels={"site-packages"} if name == "python_stdlib" else None,
        )
    return summaries


def _module_file(module: object) -> Path | None:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.suffix == ".pyc":
        try:
            source = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            source = None
        if source is not None and source.is_file():
            path = source
    if not path.is_absolute():
        path = Path(os.path.abspath(path))
    path = path.resolve()
    return path if path.is_file() else None


def _pilot_imported_external_code_summary() -> dict:
    repository_root = Path(__file__).resolve().parents[1]
    covered_roots = {
        name: Path(path).resolve() for name, path in CANONICAL_PILOT_CODE_TREES.items()
    }
    covered_scientific_files = {
        (repository_root / relative).resolve() for relative in ACW_SCIENTIFIC_PATHS
    }

    def covered_by_code_tree(path: Path) -> bool:
        for name, root in covered_roots.items():
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if name == "python_stdlib" and relative.parts[0] == "site-packages":
                continue
            return True
        return False

    files = set()
    for module in tuple(sys.modules.values()):
        path = _module_file(module)
        if (
            path is None
            or path in set(_DETACHED_TORCH_GENERATED_MODULES.values())
            or path in covered_scientific_files
            or covered_by_code_tree(path)
        ):
            continue
        files.add(path)
    records = {}
    for path in sorted(files):
        records[str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return {
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records.values()),
        "payload_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def _pilot_generated_module_summary() -> dict[str, dict]:
    if set(_DETACHED_TORCH_GENERATED_MODULES) != set(_TORCH_GENERATED_MODULE_FILENAMES):
        raise RuntimeError("canonical ACW generated-module registry is incomplete")
    records = {}
    roots = set()
    for name, filename in sorted(_TORCH_GENERATED_MODULE_FILENAMES.items()):
        path = _DETACHED_TORCH_GENERATED_MODULES[name]
        root = path.parent
        roots.add(root)
        if (
            root.is_symlink()
            or path.is_symlink()
            or path.name != filename
            or not path.is_file()
        ):
            raise RuntimeError("canonical ACW generated module is not literal")
        records[name] = {
            "bytes": path.stat().st_size,
            "filename": filename,
            "sha256": file_sha256(path),
        }
    for root in roots:
        entries = {entry.name for entry in os.scandir(root)}
        expected = {
            filename
            for name, filename in _TORCH_GENERATED_MODULE_FILENAMES.items()
            if _DETACHED_TORCH_GENERATED_MODULES[name].parent == root
        }
        if entries != expected or str(root) in sys.path:
            raise RuntimeError("canonical ACW generated-module import root is open")
    return records


def _pilot_python_startup_identity() -> dict:
    archive_paths = [entry for entry in sys.path if entry.endswith(".zip")]
    if archive_paths:
        raise RuntimeError("canonical ACW Python archive import paths are forbidden")
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    startup_files = set(purelib.glob("*.pth"))
    for raw_root in sys.path:
        if not raw_root:
            continue
        root = Path(raw_root)
        for name in ("sitecustomize.py", "usercustomize.py"):
            candidate = root / name
            if candidate.is_file():
                startup_files.add(candidate.resolve())
    registry = {
        str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(startup_files)
    }
    return {
        "purelib": str(purelib),
        "sys_path": list(sys.path),
        "startup_files": registry,
        "startup_files_payload_sha256": hashlib.sha256(
            canonical_json_bytes(registry)
        ).hexdigest(),
    }


def _pilot_external_executable_registry() -> dict[str, dict]:
    registry = {}
    for raw_path in CANONICAL_PILOT_EXTERNAL_EXECUTABLE_PATHS:
        lexical = Path(raw_path)
        resolved = lexical.resolve()
        if not lexical.is_file() or not resolved.is_file():
            raise RuntimeError("canonical ACW external executable is missing")
        registry[raw_path] = {
            "bytes": resolved.stat().st_size,
            "resolved_path": str(resolved),
            "sha256": file_sha256(resolved),
        }
    return registry


def _pilot_native_file_registry() -> dict[str, dict]:
    """Hash every file-backed executable mapping after numerical warmup."""
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1] or not fields[5].startswith("/"):
            continue
        if fields[5].endswith(" (deleted)"):
            raise RuntimeError("canonical ACW runtime has a deleted executable mapping")
        path = Path(fields[5]).resolve()
        if not path.is_file():
            raise RuntimeError("canonical ACW runtime mapping is not a regular file")
        mapped.add(path)
    executable = Path(sys.executable).resolve()
    mapped.add(executable)
    if not mapped:
        raise RuntimeError("canonical ACW runtime has no executable file mappings")
    return {
        str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(mapped)
    }


def pilot_runtime_identity() -> dict:
    global _DETACHED_TORCH_GENERATED_MODULES
    _warm_pilot_runtime()
    generated_paths = _detach_torch_generated_import_path()
    if (
        _DETACHED_TORCH_GENERATED_MODULES
        and generated_paths != _DETACHED_TORCH_GENERATED_MODULES
    ):
        raise RuntimeError("PyTorch generated module changed within one process")
    _DETACHED_TORCH_GENERATED_MODULES = generated_paths
    try:
        numpy_config = np.__config__.show(mode="dicts")
    except TypeError as error:
        raise RuntimeError(
            "NumPy cannot expose the canonical build identity"
        ) from error
    numpy_config_bytes = json.dumps(
        numpy_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("ascii")
    code_trees = _pilot_code_tree_summaries()
    external_executables = _pilot_external_executable_registry()
    generated_modules = _pilot_generated_module_summary()
    imported_external_code = _pilot_imported_external_code_summary()
    python_startup = _pilot_python_startup_identity()
    native_files = _pilot_native_file_registry()
    numpy_cpu_features = getattr(np._core._multiarray_umath, "__cpu_features__")
    return {
        "cpu": _pilot_cpu_identity(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "code_trees": code_trees,
        "external_executables": external_executables,
        "generated_modules": generated_modules,
        "imported_external_code": imported_external_code,
        "libc": list(platform.libc_ver()),
        "machine": platform.machine(),
        "native_files": native_files,
        "native_files_payload_sha256": hashlib.sha256(
            canonical_json_bytes(native_files)
        ).hexdigest(),
        "numpy_cpu_features_enabled": sorted(
            key for key, enabled in numpy_cpu_features.items() if enabled
        ),
        "numpy_config_sha256": hashlib.sha256(numpy_config_bytes).hexdigest(),
        "numpy_version": np.__version__,
        "python_implementation": platform.python_implementation(),
        "python_no_site": bool(sys.flags.no_site),
        "python_safe_path": bool(sys.flags.safe_path),
        "python_startup": python_startup,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "static_environment": {
            key: os.environ.get(key) for key in sorted(CANONICAL_PILOT_STATIC_ENV)
        },
        "system": platform.system(),
        "thread_env": {
            key: os.environ.get(key) for key in sorted(CANONICAL_PILOT_THREAD_ENV)
        },
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
    }


def _canonical_pilot_role() -> str:
    observed = {
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }
    roles = [
        role
        for role, expected in CANONICAL_PILOT_ROLES.items()
        if observed
        == {"job_name": expected["job_name"], "node_list": expected["node_list"]}
    ]
    if len(roles) != 1:
        raise RuntimeError("canonical ACW pilot role environment mismatch")
    return roles[0]


def _cpu_list_members(raw: str) -> set[int]:
    members = set()
    for item in raw.split(","):
        if not item:
            raise ValueError("canonical ACW cpuset is malformed")
        bounds = item.split("-", 1)
        if len(bounds) == 1:
            start = finish = int(bounds[0])
        else:
            start, finish = (int(value) for value in bounds)
        if start < 0 or finish < start:
            raise ValueError("canonical ACW cpuset is malformed")
        members.update(range(start, finish + 1))
    return members


def _validate_canonical_pilot_process_membership(
    cgroup_text: str,
    status_text: str,
    *,
    job_id: str,
    user_id: int,
) -> dict:
    controller_paths = {}
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise RuntimeError("canonical ACW process cgroup is malformed")
        for controller in fields[1].split(","):
            if controller:
                controller_paths[controller] = fields[2]
    task_path = f"/slurm/uid_{user_id}/job_{job_id}/step_batch/task_0"
    step_path = f"/slurm/uid_{user_id}/job_{job_id}/step_batch"
    if (
        any(
            controller_paths.get(controller) != task_path
            for controller in ("cpu", "cpuacct", "cpuset", "memory")
        )
        or controller_paths.get("freezer") != step_path
    ):
        raise RuntimeError("canonical ACW process is outside its Slurm cgroup")
    status = {}
    for line in status_text.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            status[name] = value.strip()
    cpu_list = status.get("Cpus_allowed_list", "")
    memory_list = status.get("Mems_allowed_list", "")
    try:
        cpu_members = _cpu_list_members(cpu_list)
    except (TypeError, ValueError) as error:
        raise RuntimeError("canonical ACW process cpuset is malformed") from error
    if len(cpu_members) != 4 or not memory_list:
        raise RuntimeError("canonical ACW process allocation differs from Slurm")
    return {
        "cpu_list": cpu_list,
        "memory_list": memory_list,
        "task_cgroup": task_path,
    }


def _validate_canonical_pilot_batch_script(
    spooled: bytes,
    committed: bytes,
) -> str:
    if spooled != committed:
        raise RuntimeError("canonical ACW spooled batch script differs from Git bytes")
    return hashlib.sha256(spooled).hexdigest()


def _canonical_pilot_process_membership(job_id: str, *, role: str) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("canonical ACW process membership requires Linux")
    if os.getuid() != CANONICAL_PILOT_UID:
        raise RuntimeError("canonical ACW process user ID mismatch")
    membership = _validate_canonical_pilot_process_membership(
        Path("/proc/self/cgroup").read_text(errors="strict"),
        Path("/proc/self/status").read_text(errors="strict"),
        job_id=job_id,
        user_id=CANONICAL_PILOT_UID,
    )
    script_path = Path(f"/var/spool/slurmd/job{job_id}/slurm_script")
    committed_path = Path(CANONICAL_PILOT_ROLES[role]["command"])
    if (
        script_path.is_symlink()
        or not script_path.is_file()
        or committed_path.is_symlink()
        or not committed_path.is_file()
    ):
        raise RuntimeError("canonical ACW batch script is not a literal file")
    repository_root = Path(__file__).resolve().parents[1]
    committed = subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "show",
            f"HEAD:{CANONICAL_PILOT_ROLES[role]['scientific_path']}",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    if committed_path.read_bytes() != committed:
        raise RuntimeError("canonical ACW batch worktree differs from its Git blob")
    membership["batch_script_sha256"] = _validate_canonical_pilot_batch_script(
        script_path.read_bytes(),
        committed,
    )
    return membership


def _canonical_pilot_environment(
    *,
    require_committed_batch_script: bool = True,
) -> dict[str, str]:
    expected_keys = set(CANONICAL_PILOT_STATIC_ENV) | CANONICAL_PILOT_DYNAMIC_ENV_KEYS
    if set(os.environ) != expected_keys:
        raise RuntimeError("canonical ACW pilot environment allowlist mismatch")
    observed_static = {
        key: os.environ.get(key) for key in sorted(CANONICAL_PILOT_STATIC_ENV)
    }
    if observed_static != CANONICAL_PILOT_STATIC_ENV:
        raise RuntimeError("canonical ACW pilot static environment mismatch")
    role = _canonical_pilot_role()
    expected_role = CANONICAL_PILOT_ROLES[role]
    dynamic = {
        key: os.environ.get(key) for key in sorted(CANONICAL_PILOT_DYNAMIC_ENV_KEYS)
    }
    if (
        not dynamic["SLURM_JOB_ID"]
        or not dynamic["SLURM_JOB_ID"].isdigit()
        or dynamic["SLURM_CPUS_PER_TASK"] != "4"
        or dynamic["SLURM_JOB_NAME"] != expected_role["job_name"]
        or dynamic["SLURM_JOB_NODELIST"] != expected_role["node_list"]
        or dynamic["SLURM_NODELIST"] != expected_role["node_list"]
        or dynamic["SLURM_SUBMIT_DIR"] != CANONICAL_PILOT_BASE
    ):
        raise RuntimeError("canonical ACW pilot dynamic environment mismatch")
    if platform.system() == "Linux":
        if os.getuid() != CANONICAL_PILOT_UID:
            raise RuntimeError("canonical ACW process user ID mismatch")
        if require_committed_batch_script:
            _canonical_pilot_process_membership(
                str(dynamic["SLURM_JOB_ID"]),
                role=role,
            )
        else:
            _validate_canonical_pilot_process_membership(
                Path("/proc/self/cgroup").read_text(errors="strict"),
                Path("/proc/self/status").read_text(errors="strict"),
                job_id=str(dynamic["SLURM_JOB_ID"]),
                user_id=os.getuid(),
            )
    return {key: str(os.environ[key]) for key in sorted(expected_keys)}


def require_canonical_pilot_runtime() -> dict:
    _canonical_pilot_environment()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observed = pilot_runtime_identity()
    if observed != CANONICAL_PILOT_RUNTIME:
        raise RuntimeError("canonical ACW pilot runtime identity mismatch")
    return observed


def _load_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    recorded = payload.pop("payload_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != recorded:
        raise ValueError("dataset manifest payload hash mismatch")
    if manifest.get("protocol") != GENERATOR_PROTOCOL:
        raise ValueError("wrong ACW generator manifest protocol")
    return manifest


def _load_bound_array(root: Path, manifest: dict, relative: str) -> np.ndarray:
    record = manifest.get("arrays", {}).get(relative)
    if not isinstance(record, dict):
        raise ValueError(f"manifest lacks array: {relative}")
    path = root / relative
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"array hash mismatch: {relative}")
    with path.open("rb") as handle:
        array = np.load(handle, allow_pickle=False)
    if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
        "dtype"
    ):
        raise ValueError(f"array schema mismatch: {relative}")
    return array


def _registered_public_seed_material(
    seed_identity: dict,
    *,
    allowed_kinds: set[str],
) -> bytes:
    kind = seed_identity.get("kind")
    if kind not in allowed_kinds:
        raise ValueError(f"registered deterministic replay forbids {kind!r} data")
    if kind == "pilot":
        if seed_identity != {"kind": "pilot", "seed": PILOT_SEED}:
            raise ValueError("pilot identity is not the registered public seed")
        return development_seed_material(PILOT_SEED)
    if kind == "development":
        if set(seed_identity) != {"kind", "seed"}:
            raise ValueError("development seed identity has the wrong schema")
        seed = int(seed_identity["seed"])
        if seed not in DEVELOPMENT_SEEDS:
            raise ValueError("development seed is outside the public registry")
        return development_seed_material(seed)
    raise ValueError("only pilot and development domains have public replay material")


def _regenerate_registered_dataset(
    out: Path,
    seed_material: bytes,
    seed_identity: dict,
) -> dict:
    return generate_dataset(
        out,
        seed_material,
        seed_identity=seed_identity,
        train_count=GENERATOR_TRAIN_HISTORIES,
        adaptation_count=GENERATOR_ADAPTATION_HISTORIES,
        evaluation_count=GENERATOR_EVALUATION_HISTORIES,
        evaluation_depths=GENERATOR_EVALUATION_DEPTHS,
    )


def _data_files(root: Path) -> set[str]:
    files = set()
    for prefix in ("public", "oracle"):
        directory = root / prefix
        if directory.exists():
            files.update(
                str(path.relative_to(root))
                for path in directory.rglob("*")
                if path.is_file()
            )
    return files


def _dataset_tree_entries(root: Path) -> tuple[set[str], set[str]]:
    files = set()
    directories = set()
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_file():
            files.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise ValueError("dataset contains a non-file, non-directory entry")
    return files, directories


def verify_registered_dataset(
    root: Path,
    *,
    allowed_kinds: set[str],
) -> dict:
    """Regenerate a public-seed domain and compare every public/oracle array."""
    _require_tree_without_symlinks(root)
    root = root.resolve()
    manifest = _load_manifest(root)
    if manifest.get("protocol") != GENERATOR_PROTOCOL:
        raise ValueError("dataset does not use the registered generator protocol")
    seed_identity = manifest.get("seed_identity")
    if not isinstance(seed_identity, dict):
        raise ValueError("dataset lacks a registered seed identity")
    seed_material = _registered_public_seed_material(
        seed_identity,
        allowed_kinds=allowed_kinds,
    )
    expected_fingerprint = hashlib.sha256(seed_material).hexdigest()
    if manifest.get("seed_fingerprint") != expected_fingerprint:
        raise ValueError("dataset seed fingerprint fails deterministic replay")

    with tempfile.TemporaryDirectory(prefix="acw-data-replay-") as temporary:
        replay_root = Path(temporary) / "dataset"
        expected = _regenerate_registered_dataset(
            replay_root,
            seed_material,
            dict(seed_identity),
        )
        observed_arrays = manifest.get("arrays")
        expected_arrays = expected.get("arrays")
        if not isinstance(observed_arrays, dict) or set(observed_arrays) != set(
            expected_arrays or {}
        ):
            raise ValueError("dataset array registry fails deterministic replay")
        expected_files = set(expected_arrays)
        allowed_files = {"manifest.json", *expected_files}
        allowed_directories = {
            str(parent)
            for relative in expected_files
            for parent in Path(relative).parents
            if str(parent) != "."
        }
        observed_files, observed_directories = _dataset_tree_entries(root)
        if (
            observed_files != allowed_files
            or observed_directories != allowed_directories
        ):
            raise ValueError("dataset tree registry fails deterministic replay")
        if _data_files(replay_root) != expected_files:
            raise RuntimeError(
                "registered generator emitted an unexpected file registry"
            )

        public_count = 0
        oracle_count = 0
        for relative in sorted(expected_files):
            observed = _load_bound_array(root, manifest, relative)
            regenerated = _load_bound_array(replay_root, expected, relative)
            if not np.array_equal(observed, regenerated):
                raise ValueError(
                    f"dataset array differs from deterministic replay: {relative}"
                )
            public_count += relative.startswith("public/")
            oracle_count += relative.startswith("oracle/")

        if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
            raise ValueError("dataset manifest differs from deterministic replay")

    return {
        "protocol": DATA_REPLAY_PROTOCOL,
        "seed_identity": dict(seed_identity),
        "seed_fingerprint": expected_fingerprint,
        "source_manifest_payload_sha256": manifest["payload_sha256"],
        "regenerated_manifest_payload_sha256": expected["payload_sha256"],
        "array_registry_sha256": hashlib.sha256(
            canonical_json_bytes(expected["arrays"])
        ).hexdigest(),
        "arrays_verified": len(expected["arrays"]),
        "public_arrays_verified": public_count,
        "oracle_arrays_verified": oracle_count,
    }


def load_oracle_truth(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    _require_tree_without_symlinks(root)
    root = root.resolve()
    manifest = _load_manifest(root)
    states = _load_bound_array(root, manifest, "oracle/train/final_states.npy")
    answers = _load_bound_array(root, manifest, "oracle/train/public_answers.npy")
    if states.shape != (answers.shape[0], 3) or answers.shape[1] != PUBLIC_QUERIES:
        raise ValueError("oracle training truth has the wrong shape")
    return states, answers, manifest


def _initial_rows(data: PublicTrainingData, oracle_answers: np.ndarray) -> list[dict]:
    rows = []
    for history_id in range(data.histories):
        for query_id, answer in zip(
            data.initial_queries[history_id].tolist(),
            data.initial_answers[history_id].tolist(),
            strict=True,
        ):
            if int(oracle_answers[history_id, query_id]) != answer:
                raise ValueError("public initial answer disagrees with oracle truth")
            rows.append(
                {
                    "history_id": history_id,
                    "query_id": int(query_id),
                    "round": 0,
                }
            )
    return rows


def _rows_to_curriculum(rows: list[dict], oracle_answers: np.ndarray) -> Curriculum:
    return Curriculum(
        history_ids=torch.tensor([row["history_id"] for row in rows], dtype=torch.long),
        query_ids=torch.tensor([row["query_id"] for row in rows], dtype=torch.long),
        answers=torch.tensor(
            [oracle_answers[row["history_id"], row["query_id"]] for row in rows],
            dtype=torch.long,
        ),
        rounds=torch.tensor([row["round"] for row in rows], dtype=torch.long),
    )


def validate_query_schedule(
    rows: list[dict],
    histories: int,
    *,
    refinement_rounds: int,
    canonical: bool,
) -> None:
    if any(set(row) != {"history_id", "query_id", "round"} for row in rows):
        raise ValueError("query schedule has the wrong schema")
    pairs = set()
    per_history = [0] * histories
    round_counts = [0] * (refinement_rounds + 1)
    for row in rows:
        history_id = row["history_id"]
        query_id = row["query_id"]
        round_index = row["round"]
        if not 0 <= history_id < histories or not 0 <= query_id < PUBLIC_QUERIES:
            raise ValueError("query schedule index is outside the public domain")
        if not 0 <= round_index <= refinement_rounds:
            raise ValueError("query schedule round is outside the configured range")
        pair = (history_id, query_id)
        if pair in pairs:
            raise ValueError("query schedule repeats a history/query pair")
        pairs.add(pair)
        per_history[history_id] += 1
        round_counts[round_index] += 1
    expected_per_history = 2 + refinement_rounds
    if per_history != [expected_per_history] * histories:
        raise ValueError("query schedule multiplicity differs across histories")
    expected_rounds = [2 * histories] + [histories] * refinement_rounds
    if round_counts != expected_rounds:
        raise ValueError("query schedule round multiplicity is invalid")
    if canonical and (
        histories != CANONICAL_HISTORIES
        or len(rows) != CANONICAL_LABELS
        or refinement_rounds != REFINEMENT_ROUNDS
    ):
        raise ValueError("canonical query schedule dimensions are invalid")


def _unused_query(
    used: set[int],
    *,
    seed: int,
    round_index: int,
    history_id: int,
    domain: str,
) -> int:
    available = sorted(set(range(PUBLIC_QUERIES)) - used)
    if not available:
        raise RuntimeError("history exhausted the public query bank")
    material = (
        b"R12-ACW-UNUSED-QUERY-v1\x00"
        + domain.encode("ascii")
        + b"\x00"
        + seed.to_bytes(8, "big")
        + round_index.to_bytes(2, "big")
        + history_id.to_bytes(8, "big")
    )
    rank = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return available[rank % len(available)]


def select_refinement_round(
    packets: np.ndarray,
    final_states: np.ndarray,
    oracle_answers: np.ndarray,
    rows: list[dict],
    *,
    round_index: int,
    max_groups: int = MAX_GROUPS_PER_ROUND,
) -> tuple[list[dict], dict]:
    histories = len(packets)
    if packets.shape != (histories, 3) or packets.dtype != np.uint8:
        raise ValueError("pilot packets must be literal uint8 triples")
    if final_states.shape != (histories, 3):
        raise ValueError("final states do not match pilot packets")
    used = [set() for _ in range(histories)]
    for row in rows:
        used[row["history_id"]].add(row["query_id"])

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for history_id, packet in enumerate(packets):
        grouped.setdefault(tuple(int(value) for value in packet), []).append(history_id)
    collisions = []
    for packet, members in grouped.items():
        residual_classes = len(
            {tuple(int(value) for value in final_states[h]) for h in members}
        )
        if residual_classes > 1:
            collisions.append((residual_classes, packet, min(members), members))
    collisions.sort(key=lambda item: (-item[0], item[1], item[2]))

    assigned: dict[int, int] = {}
    candidate_evaluations = 0
    selected_witnesses = 0
    witness_exhausted = 0
    query_bank_unresolved = 0
    for residual_classes, packet, _, members in collisions[:max_groups]:
        del residual_classes, packet
        common_unused = set(range(PUBLIC_QUERIES))
        for history_id in members:
            common_unused.difference_update(used[history_id])
        if not common_unused:
            witness_exhausted += 1
            continue
        best_query = None
        best_distinct = -1
        for query_id in sorted(common_unused):
            distinct = len({int(oracle_answers[h, query_id]) for h in members})
            candidate_evaluations += 1
            if distinct > best_distinct:
                best_query = query_id
                best_distinct = distinct
        if best_distinct < 2:
            query_bank_unresolved += 1
            continue
        selected_witnesses += 1
        for history_id in members:
            assigned[history_id] = int(best_query)

    new_rows = []
    filler_histories = 0
    for history_id in range(histories):
        query_id = assigned.get(history_id)
        if query_id is None:
            filler_histories += 1
            query_id = _unused_query(
                used[history_id],
                seed=PILOT_SEED,
                round_index=round_index,
                history_id=history_id,
                domain="CGBR-FILLER",
            )
        if query_id in used[history_id]:
            raise RuntimeError("refinement selected an already-used query")
        new_rows.append(
            {
                "history_id": history_id,
                "query_id": query_id,
                "round": round_index,
            }
        )
    report = {
        "round": round_index,
        "packet_classes": len(grouped),
        "cross_residual_collision_groups": len(collisions),
        "groups_scanned": min(len(collisions), max_groups),
        "selected_witnesses": selected_witnesses,
        "witness_exhausted_groups": witness_exhausted,
        "query_bank_unresolved_groups": query_bank_unresolved,
        "candidate_evaluations": candidate_evaluations,
        "filler_histories": filler_histories,
    }
    return new_rows, report


def build_uniform_schedule(
    initial_rows: list[dict],
    histories: int,
    *,
    refinement_rounds: int,
) -> list[dict]:
    rows = list(initial_rows)
    used = [set() for _ in range(histories)]
    for row in rows:
        used[row["history_id"]].add(row["query_id"])
    for round_index in range(1, refinement_rounds + 1):
        for history_id in range(histories):
            query_id = _unused_query(
                used[history_id],
                seed=UNIFORM_SEED,
                round_index=round_index,
                history_id=history_id,
                domain="UNIFORM",
            )
            used[history_id].add(query_id)
            rows.append(
                {
                    "history_id": history_id,
                    "query_id": query_id,
                    "round": round_index,
                }
            )
    return sorted(rows, key=lambda row: (row["history_id"], row["query_id"]))


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        encoded = canonical_json_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _hard_packets(
    model: torch.nn.Module,
    data: PublicTrainingData,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    packets = []
    with torch.no_grad():
        for start in range(0, data.histories, batch_size):
            history_ids = torch.arange(start, min(start + batch_size, data.histories))
            packets.append(
                recurrent_state(
                    model,
                    "acw",
                    data,
                    history_ids,
                    training=False,
                    literal_symbols=True,
                )
                .cpu()
                .numpy()
            )
    model.train()
    result = np.concatenate(packets, axis=0)
    if result.dtype != np.uint8:
        raise RuntimeError("pilot packet extraction lost literal uint8 persistence")
    return result


def run_pilot(
    root: Path,
    *,
    seed: int = PILOT_SEED,
    refinement_rounds: int = REFINEMENT_ROUNDS,
    updates_per_round: int = CANONICAL_UPDATES_PER_ROUND,
    final_updates: int = CANONICAL_FINAL_UPDATES,
    batch_size: int = CANONICAL_BATCH_SIZE,
    max_groups: int = MAX_GROUPS_PER_ROUND,
    canonical: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    replay_verification = None
    canonical_runtime = None
    if canonical:
        canonical_values = (
            seed == PILOT_SEED,
            refinement_rounds == REFINEMENT_ROUNDS,
            updates_per_round == CANONICAL_UPDATES_PER_ROUND,
            final_updates == CANONICAL_FINAL_UPDATES,
            batch_size == CANONICAL_BATCH_SIZE,
            max_groups == MAX_GROUPS_PER_ROUND,
        )
        if not all(canonical_values):
            raise ValueError("canonical pilot hyperparameters are frozen internally")
        canonical_runtime = require_canonical_pilot_runtime()
        replay_verification = verify_registered_dataset(
            root,
            allowed_kinds={"pilot"},
        )
    data = load_public_training_data(root, reject_oracle=False)
    final_states, oracle_answers, manifest = load_oracle_truth(root)
    if canonical and manifest.get("seed_identity") != {
        "kind": "pilot",
        "seed": PILOT_SEED,
    }:
        raise ValueError("canonical curriculum pilot requires the frozen pilot dataset")
    if (
        canonical
        and manifest.get("payload_sha256") != CANONICAL_PILOT_DATASET_PAYLOAD_SHA256
    ):
        raise ValueError("canonical pilot dataset differs from its cross-runtime pin")
    if len(final_states) != data.histories:
        raise ValueError("public and oracle training history counts differ")
    rows = _initial_rows(data, oracle_answers)
    model = initialized_model_for_arm("acw", seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.0001,
    )
    round_reports = []
    total_updates = 0

    def optimize(count: int) -> list[float]:
        nonlocal total_updates
        curriculum = _rows_to_curriculum(rows, oracle_answers)
        losses = []
        for _ in range(count):
            selected = torch.randint(
                len(curriculum.history_ids),
                (batch_size,),
                generator=generator,
            )
            logits = forward_logits(
                model,
                "acw",
                data,
                curriculum.history_ids[selected],
                curriculum.query_ids[selected],
                training=True,
            )
            loss = F.cross_entropy(logits.float(), curriculum.answers[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            total_updates += 1
        return losses

    for round_index in range(refinement_rounds + 1):
        losses = optimize(updates_per_round)
        report = {
            "round": round_index,
            "labels_before_update": len(rows),
            "optimizer_updates": updates_per_round,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        }
        if round_index < refinement_rounds:
            packets = _hard_packets(model, data, batch_size=batch_size)
            report["packet_sha256"] = hashlib.sha256(packets.tobytes()).hexdigest()
            additions, selection = select_refinement_round(
                packets,
                final_states,
                oracle_answers,
                rows,
                round_index=round_index + 1,
                max_groups=max_groups,
            )
            rows.extend(additions)
            report["selection"] = selection
        round_reports.append(report)
    final_losses = optimize(final_updates)
    cgb_schedule = sorted(rows, key=lambda row: (row["history_id"], row["query_id"]))
    uniform_schedule = build_uniform_schedule(
        _initial_rows(data, oracle_answers),
        data.histories,
        refinement_rounds=refinement_rounds,
    )
    validate_query_schedule(
        cgb_schedule,
        data.histories,
        refinement_rounds=refinement_rounds,
        canonical=canonical,
    )
    validate_query_schedule(
        uniform_schedule,
        data.histories,
        refinement_rounds=refinement_rounds,
        canonical=canonical,
    )
    candidate_evaluations = sum(
        item.get("selection", {}).get("candidate_evaluations", 0)
        for item in round_reports
    )
    if canonical and candidate_evaluations > MAX_CANDIDATE_EVALUATIONS:
        raise RuntimeError("pilot exceeded its preregistered oracle-query cap")
    report = {
        "protocol": PILOT_PROTOCOL,
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "model_arm": "acw",
        "deterministic_algorithms": canonical,
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
        },
        "pilot_seed": seed,
        "uniform_seed": UNIFORM_SEED,
        "dataset_manifest_payload_sha256": manifest["payload_sha256"],
        "histories": data.histories,
        "refinement_rounds": refinement_rounds,
        "updates_per_round": updates_per_round,
        "final_updates": final_updates,
        "total_updates": total_updates,
        "batch_size": batch_size,
        "max_groups_per_round": max_groups,
        "labels": len(cgb_schedule),
        "candidate_evaluations": candidate_evaluations,
        "rounds": round_reports,
        "final_loss_first": final_losses[0] if final_losses else None,
        "final_loss_last": final_losses[-1] if final_losses else None,
        "model_tensor_sha256": _tensor_state_sha256(model),
        "canonical_runtime": canonical_runtime,
        "dataset_replay_verification": replay_verification,
        "claim_boundary": (
            "Non-scored curriculum pilot only. This is neither a scored architecture "
            "result nor evidence of reasoning."
        ),
    }
    return cgb_schedule, uniform_schedule, report


def _schedule_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _published_pilot_files(
    cgb_schedule: list[dict],
    uniform_schedule: list[dict],
    report: dict,
    *,
    identity: dict,
) -> tuple[dict[str, bytes], dict]:
    files = {
        "cgb_schedule.jsonl": _schedule_bytes(cgb_schedule),
        "uniform_schedule.jsonl": _schedule_bytes(uniform_schedule),
    }
    published = dict(report)
    published["scientific_identity"] = identity
    published["schedules"] = {
        name: {
            "bytes": len(payload),
            "rows": len(cgb_schedule)
            if name.startswith("cgb_")
            else len(uniform_schedule),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in files.items()
    }
    published["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(published)
    ).hexdigest()
    files["report.json"] = canonical_json_bytes(published) + b"\n"
    return files, published


def _parse_scontrol_fields(snapshot: str) -> dict[str, str]:
    fields = {}
    for token in shlex.split(snapshot):
        if "=" in token:
            name, value = token.split("=", 1)
            fields[name] = value
    return fields


def _validate_slurm_snapshot_record(
    snapshot: dict,
    *,
    role: str,
) -> dict:
    if role not in CANONICAL_PILOT_ROLES or set(snapshot) != {
        "command",
        "stdout",
        "stdout_sha256",
        "allocation",
    }:
        raise ValueError("canonical ACW Slurm snapshot has the wrong schema")
    stdout = snapshot.get("stdout")
    allocation = snapshot.get("allocation")
    if not isinstance(stdout, str) or not isinstance(allocation, dict):
        raise ValueError("canonical ACW Slurm snapshot is malformed")
    if set(allocation) != {
        "batch_host",
        "command",
        "job_id",
        "job_name",
        "job_state",
        "node_list",
        "num_cpus",
        "num_nodes",
        "partition",
        "stdout",
        "work_dir",
    }:
        raise ValueError("canonical ACW Slurm allocation has the wrong schema")
    job_id = allocation.get("job_id")
    expected = CANONICAL_PILOT_ROLES[role]
    expected_command = [CANONICAL_PILOT_SCONTROL, "show", "job", "-o", job_id]
    expected_stdout = f"{expected['stdout_prefix']}{job_id}.out"
    fields = _parse_scontrol_fields(stdout)
    if (
        not isinstance(job_id, str)
        or not job_id.isdigit()
        or snapshot.get("command") != expected_command
        or snapshot.get("stdout_sha256")
        != hashlib.sha256(stdout.encode("utf-8")).hexdigest()
        or fields.get("JobId") != job_id
        or fields.get("JobState") != "RUNNING"
        or fields.get("JobName") != expected["job_name"]
        or fields.get("Command") != expected["command"]
        or fields.get("WorkDir") != CANONICAL_PILOT_BASE
        or fields.get("NodeList") != expected["node_list"]
        or fields.get("BatchHost") != expected["node_list"]
        or fields.get("Partition") != "normal"
        or fields.get("StdOut") != expected_stdout
        or fields.get("NumCPUs") != "4"
        or fields.get("NumNodes") != "1"
        or allocation
        != {
            "batch_host": fields.get("BatchHost"),
            "command": fields.get("Command"),
            "job_id": fields.get("JobId"),
            "job_name": fields.get("JobName"),
            "job_state": fields.get("JobState"),
            "node_list": fields.get("NodeList"),
            "num_cpus": int(fields.get("NumCPUs", "0")),
            "num_nodes": int(fields.get("NumNodes", "0")),
            "partition": fields.get("Partition"),
            "stdout": fields.get("StdOut"),
            "work_dir": fields.get("WorkDir"),
        }
    ):
        raise ValueError("canonical ACW Slurm allocation differs from its role")
    return allocation


def _slurm_snapshot(*, required: bool, role: str | None = None) -> dict | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        if required:
            raise RuntimeError("canonical pilot execution requires SLURM_JOB_ID")
        return None
    if not job_id.isdigit():
        raise RuntimeError("SLURM_JOB_ID must be numeric")
    if required and role not in CANONICAL_PILOT_ROLES:
        raise RuntimeError("canonical pilot Slurm role is required")
    command = [CANONICAL_PILOT_SCONTROL, "show", "job", "-o", job_id]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if required:
            raise RuntimeError(
                "canonical pilot could not query its Slurm job"
            ) from error
        return None
    snapshot = result.stdout.strip()
    if not snapshot or f"JobId={job_id}" not in snapshot:
        raise RuntimeError("Slurm job snapshot does not bind the active job ID")
    fields = _parse_scontrol_fields(snapshot)
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    node_list = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if required and (
        fields.get("JobId") != job_id
        or fields.get("JobState") != "RUNNING"
        or not cpus
        or not cpus.isdigit()
        or int(fields.get("NumCPUs", "0")) != int(cpus)
        or not node_list
        or fields.get("NodeList") != node_list
    ):
        raise RuntimeError("live Slurm allocation differs from the pilot environment")
    record = {
        "command": command,
        "stdout": snapshot,
        "stdout_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        "allocation": {
            "batch_host": fields.get("BatchHost"),
            "command": fields.get("Command"),
            "job_id": fields.get("JobId"),
            "job_name": fields.get("JobName"),
            "job_state": fields.get("JobState"),
            "node_list": fields.get("NodeList"),
            "num_cpus": int(fields.get("NumCPUs", "0")),
            "num_nodes": int(fields.get("NumNodes", "0")),
            "partition": fields.get("Partition"),
            "stdout": fields.get("StdOut"),
            "work_dir": fields.get("WorkDir"),
        },
    }
    if required:
        try:
            _validate_slurm_snapshot_record(record, role=role)
        except ValueError as error:
            raise RuntimeError("canonical pilot Slurm role binding failed") from error
        if socket.getfqdn() != CANONICAL_PILOT_ROLES[role]["hostname"]:
            raise RuntimeError("canonical pilot hostname differs from its role")
    return record


def execute_pilot_replay(
    root: Path,
    out: Path,
    *,
    replay_id: str,
    canonical: bool,
    pilot_kwargs: dict | None = None,
    hold_fd: int | None = None,
) -> dict:
    """Run and publish one replay while measuring the process that did the work."""
    if replay_id not in {"a", "b"}:
        raise ValueError("pilot replay ID must be 'a' or 'b'")
    if canonical and hold_fd is None:
        raise RuntimeError("canonical pilot child requires a parent-held release pipe")
    if canonical:
        expected_out = (
            CANONICAL_PILOT_REPLAY_A if replay_id == "a" else CANONICAL_PILOT_REPLAY_B
        )
        if _lexical_absolute(root) != _canonical_path(
            CANONICAL_PILOT_DATASET
        ) or _lexical_absolute(out) != _canonical_path(expected_out):
            raise ValueError("canonical pilot replay paths are frozen")
    if hold_fd is not None:
        os.fstat(hold_fd)
    started_time_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    identity = (
        scientific_identity(require_clean=True)
        if canonical
        else {
            "scientific_commit": "noncanonical-test",
            "scientific_path_sha256": {"noncanonical-test": "0" * 64},
        }
    )
    cgb, uniform, report = run_pilot(
        root,
        canonical=canonical,
        **(pilot_kwargs or {}),
    )
    if canonical and require_canonical_pilot_runtime() != report["canonical_runtime"]:
        raise RuntimeError("canonical ACW replay runtime changed during execution")
    finished_monotonic_ns = time.monotonic_ns()
    finished_time_ns = time.time_ns()
    files, report = _published_pilot_files(
        cgb,
        uniform,
        report,
        identity=identity,
    )
    execution = {
        "protocol": PILOT_EXECUTION_PROTOCOL,
        "replay_id": replay_id,
        "execution_nonce": secrets.token_hex(32),
        "process_id": os.getpid(),
        "hostname": socket.getfqdn(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "canonical_runtime": report["canonical_runtime"],
        "started_time_ns": started_time_ns,
        "finished_time_ns": finished_time_ns,
        "elapsed_wall_ns": finished_time_ns - started_time_ns,
        "elapsed_monotonic_ns": finished_monotonic_ns - started_monotonic_ns,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cpu_count": int(os.cpu_count() or 0),
        "slurm_cpus_per_task": (
            int(os.environ["SLURM_CPUS_PER_TASK"])
            if os.environ.get("SLURM_CPUS_PER_TASK")
            else None
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_snapshot": _slurm_snapshot(
            required=canonical,
            role="producer" if canonical else None,
        ),
        "dataset_manifest_payload_sha256": report["dataset_manifest_payload_sha256"],
        "scientific_identity": identity,
        "report_sha256": hashlib.sha256(files["report.json"]).hexdigest(),
        "schedule_sha256": {
            name: record["sha256"] for name, record in report["schedules"].items()
        },
    }
    execution["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(execution)
    ).hexdigest()
    files["execution.json"] = canonical_json_bytes(execution) + b"\n"
    _write_pilot_output(out, files, canonical=canonical)
    if hold_fd is not None:
        try:
            if os.read(hold_fd, 1) != b"1":
                raise RuntimeError("pilot child received an invalid parent release")
        finally:
            os.close(hold_fd)
    return report


def _write_pilot_output(
    out: Path,
    files: dict[str, bytes],
    *,
    canonical: bool = False,
) -> None:
    out = _lexical_absolute(out) if canonical else out.resolve()
    canonical_relative = None
    if canonical:
        for relative in (CANONICAL_PILOT_REPLAY_A, CANONICAL_PILOT_REPLAY_B):
            if out == _canonical_path(relative):
                canonical_relative = relative
                break
        if canonical_relative is None:
            raise ValueError("canonical pilot output path is frozen")
    if out.exists():
        raise FileExistsError(out)
    partial = out.with_name(out.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        for name, payload in files.items():
            (partial / name).write_bytes(payload)
        for path in partial.iterdir():
            path.chmod(0o444)
        if canonical and out != _canonical_path(canonical_relative):
            raise RuntimeError("canonical pilot output path changed during publication")
        partial.replace(out)
        if canonical and out != _canonical_path(canonical_relative):
            raise RuntimeError("canonical pilot output became a symlink")
        out.chmod(0o555)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def load_query_schedule(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed query-schedule row {line_number}"
                ) from error
            rows.append(row)
    return rows


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("hash-bound JSON contains a duplicate key")
        record[key] = value
    return record


def _load_hash_bound_json_bytes(
    raw: bytes,
    *,
    label: str,
    require_canonical_bytes: bool = False,
) -> dict:
    record = json.loads(raw, object_pairs_hook=_unique_json_object)
    if require_canonical_bytes and raw != canonical_json_bytes(record) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    payload = dict(record)
    recorded = payload.pop("payload_sha256", None)
    if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != recorded:
        raise ValueError(f"{label} payload hash mismatch")
    return record


def _load_hash_bound_json(
    path: Path,
    *,
    label: str,
    require_canonical_bytes: bool = False,
) -> dict:
    return _load_hash_bound_json_bytes(
        path.read_bytes(),
        label=label,
        require_canonical_bytes=require_canonical_bytes,
    )


def _validate_replay_report(path: Path, *, canonical: bool) -> dict:
    path = _lexical_absolute(path) if canonical else path.resolve()
    if canonical and path.is_symlink():
        raise ValueError("canonical pilot report may not be a symlink")
    report = _load_hash_bound_json(path, label="pilot report")
    if report.get("protocol") != PILOT_PROTOCOL:
        raise ValueError("wrong pilot report protocol")
    identity = report.get("scientific_identity")
    if not isinstance(identity, dict) or set(identity) != {
        "scientific_commit",
        "scientific_path_sha256",
    }:
        raise ValueError("pilot report lacks a scientific identity")
    if canonical and identity != scientific_identity(require_clean=True):
        raise ValueError(
            "pilot report scientific identity differs from the executing code"
        )
    schedules = report.get("schedules")
    if set(schedules or {}) != {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}:
        raise ValueError("pilot report schedule registry is incomplete")
    refinement_rounds = int(report.get("refinement_rounds", -1))
    for name, record in schedules.items():
        if set(record) != {"bytes", "rows", "sha256"}:
            raise ValueError("pilot schedule binding has the wrong schema")
        schedule_path = path.parent / name
        if (
            not schedule_path.is_file()
            or (canonical and schedule_path.is_symlink())
            or schedule_path.stat().st_size != record["bytes"]
            or file_sha256(schedule_path) != record["sha256"]
        ):
            raise ValueError("pilot schedule file differs from its report binding")
        rows = load_query_schedule(schedule_path)
        if len(rows) != record["rows"]:
            raise ValueError("pilot schedule row count differs from its report binding")
        validate_query_schedule(
            rows,
            int(report.get("histories", -1)),
            refinement_rounds=refinement_rounds,
            canonical=canonical,
        )
    if report.get("labels") != schedules["cgb_schedule.jsonl"]["rows"]:
        raise ValueError("pilot label count differs from its schedule")
    if canonical:
        expected = {
            "schedule_protocol": SCHEDULE_PROTOCOL,
            "model_arm": "acw",
            "deterministic_algorithms": True,
            "optimizer": {
                "kind": "AdamW",
                "learning_rate": 0.003,
                "weight_decay": 0.0001,
            },
            "pilot_seed": PILOT_SEED,
            "uniform_seed": UNIFORM_SEED,
            "canonical_runtime": CANONICAL_PILOT_RUNTIME,
            "dataset_manifest_payload_sha256": (CANONICAL_PILOT_DATASET_PAYLOAD_SHA256),
            "histories": CANONICAL_HISTORIES,
            "refinement_rounds": REFINEMENT_ROUNDS,
            "updates_per_round": CANONICAL_UPDATES_PER_ROUND,
            "final_updates": CANONICAL_FINAL_UPDATES,
            "total_updates": CANONICAL_TOTAL_UPDATES,
            "batch_size": CANONICAL_BATCH_SIZE,
            "max_groups_per_round": MAX_GROUPS_PER_ROUND,
            "labels": CANONICAL_LABELS,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "pilot report differs from frozen canonical hyperparameters"
            )
        if int(report.get("candidate_evaluations", -1)) > MAX_CANDIDATE_EVALUATIONS:
            raise ValueError("pilot report exceeds the frozen oracle-query cap")
        replay = report.get("dataset_replay_verification")
        expected_keys = {
            "protocol",
            "seed_identity",
            "seed_fingerprint",
            "source_manifest_payload_sha256",
            "regenerated_manifest_payload_sha256",
            "array_registry_sha256",
            "arrays_verified",
            "public_arrays_verified",
            "oracle_arrays_verified",
        }
        if not isinstance(replay, dict) or set(replay) != expected_keys:
            raise ValueError("pilot report lacks complete data replay verification")
        if (
            replay["protocol"] != DATA_REPLAY_PROTOCOL
            or replay["seed_identity"] != {"kind": "pilot", "seed": PILOT_SEED}
            or replay["source_manifest_payload_sha256"]
            != report.get("dataset_manifest_payload_sha256")
            or replay["regenerated_manifest_payload_sha256"]
            != report.get("dataset_manifest_payload_sha256")
            or min(
                int(replay["arrays_verified"]),
                int(replay["public_arrays_verified"]),
                int(replay["oracle_arrays_verified"]),
            )
            <= 0
        ):
            raise ValueError("pilot report data replay verification is inconsistent")
    return report


def _validate_execution(
    root: Path,
    report: dict,
    *,
    replay_id: str,
    canonical: bool,
    require_live_scheduler: bool = False,
) -> dict:
    execution_path = root / "execution.json"
    if canonical and execution_path.is_symlink():
        raise ValueError("canonical pilot execution may not be a symlink")
    execution = _load_hash_bound_json(execution_path, label="pilot execution")
    required = {
        "protocol",
        "replay_id",
        "execution_nonce",
        "process_id",
        "hostname",
        "python_executable",
        "python_version",
        "torch_version",
        "numpy_version",
        "canonical_runtime",
        "started_time_ns",
        "finished_time_ns",
        "elapsed_wall_ns",
        "elapsed_monotonic_ns",
        "peak_rss_kib",
        "cpu_count",
        "slurm_cpus_per_task",
        "slurm_job_id",
        "slurm_snapshot",
        "dataset_manifest_payload_sha256",
        "scientific_identity",
        "report_sha256",
        "schedule_sha256",
        "payload_sha256",
    }
    nonce = execution.get("execution_nonce")
    if (
        set(execution) != required
        or execution.get("protocol") != PILOT_EXECUTION_PROTOCOL
    ):
        raise ValueError("pilot execution receipt has the wrong schema")
    if (
        execution.get("replay_id") != replay_id
        or not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or int(execution.get("process_id", 0)) <= 0
        or not isinstance(execution.get("hostname"), str)
        or not execution["hostname"]
        or not isinstance(execution.get("python_executable"), str)
        or not execution["python_executable"]
        or not isinstance(execution.get("python_version"), str)
        or not execution["python_version"]
        or not isinstance(execution.get("torch_version"), str)
        or not execution["torch_version"]
        or not isinstance(execution.get("numpy_version"), str)
        or not execution["numpy_version"]
        or int(execution.get("started_time_ns", 0)) <= 0
        or int(execution.get("finished_time_ns", 0)) <= 0
        or int(execution.get("finished_time_ns", 0))
        <= int(execution.get("started_time_ns", 0))
        or int(execution.get("elapsed_wall_ns", 0))
        != int(execution.get("finished_time_ns", 0))
        - int(execution.get("started_time_ns", 0))
        or int(execution.get("elapsed_monotonic_ns", 0)) <= 0
        or int(execution.get("peak_rss_kib", 0)) <= 0
        or int(execution.get("cpu_count", 0)) <= 0
    ):
        raise ValueError("pilot execution receipt is invalid")
    if canonical:
        job_id = execution.get("slurm_job_id")
        cpus = execution.get("slurm_cpus_per_task")
        snapshot = execution.get("slurm_snapshot")
        try:
            allocation = _validate_slurm_snapshot_record(snapshot, role="producer")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "canonical pilot lacks measured Slurm execution evidence"
            ) from error
        if (
            not isinstance(job_id, str)
            or not job_id.isdigit()
            or cpus != 4
            or allocation["job_id"] != job_id
            or allocation["num_cpus"] != cpus
            or execution["hostname"] != CANONICAL_PILOT_ROLES["producer"]["hostname"]
            or execution["elapsed_wall_ns"] <= 0
            or abs(
                int(execution["elapsed_wall_ns"])
                - int(execution["elapsed_monotonic_ns"])
            )
            > max(5_000_000_000, int(execution["elapsed_monotonic_ns"]) // 20)
        ):
            raise ValueError("canonical pilot lacks measured Slurm execution evidence")
        if execution.get("canonical_runtime") != CANONICAL_PILOT_RUNTIME:
            raise ValueError("canonical pilot execution has the wrong runtime identity")
        if require_live_scheduler:
            current = _slurm_snapshot(required=True, role="producer")
            if (
                current is None
                or current["allocation"] != allocation
                or os.environ.get("SLURM_JOB_ID") != job_id
                or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != cpus
                or execution["hostname"] != socket.getfqdn()
            ):
                raise ValueError(
                    "canonical pilot execution differs from the live Slurm allocation"
                )
    runtime = execution.get("canonical_runtime")
    runtime_mismatch = (
        canonical
        and CANONICAL_PILOT_RUNTIME is not None
        and (
            not isinstance(runtime, dict)
            or execution["python_executable"] != runtime.get("python_executable")
            or execution["python_version"] != runtime.get("python_version")
            or execution["torch_version"] != runtime.get("torch_version")
            or execution["numpy_version"] != runtime.get("numpy_version")
        )
    )
    if runtime_mismatch or (
        execution["dataset_manifest_payload_sha256"]
        != report["dataset_manifest_payload_sha256"]
        or execution["canonical_runtime"] != report.get("canonical_runtime")
        or execution["scientific_identity"] != report["scientific_identity"]
        or execution["report_sha256"] != file_sha256(root / "report.json")
        or execution["schedule_sha256"]
        != {name: record["sha256"] for name, record in report["schedules"].items()}
    ):
        raise ValueError("pilot execution receipt differs from replay output")
    return execution


def _validate_replay_output(
    root: Path,
    *,
    canonical: bool,
    replay_id: str,
    require_live_scheduler: bool = False,
) -> tuple[dict, dict]:
    root = _lexical_absolute(root) if canonical else root.resolve()
    if canonical:
        _require_tree_without_symlinks(root)
    expected_files = {
        "cgb_schedule.jsonl",
        "uniform_schedule.jsonl",
        "report.json",
        "execution.json",
    }
    if {path.name for path in root.iterdir() if path.is_file()} != expected_files:
        raise ValueError("pilot replay output file registry is incomplete")
    report = _validate_replay_report(root / "report.json", canonical=canonical)
    execution = _validate_execution(
        root,
        report,
        replay_id=replay_id,
        canonical=canonical,
        require_live_scheduler=require_live_scheduler,
    )
    return report, execution


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_tree_without_symlinks(root: Path) -> None:
    root = _lexical_absolute(root)
    if root.is_symlink() or (
        root.exists() and any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError("canonical ACW artifact tree contains a symlink")


def _canonical_path(relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("canonical ACW path must be repository-relative")
    root = Path(__file__).resolve().parents[1]
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("canonical ACW path contains a symlink")
    return _lexical_absolute(candidate)


def _recompute_pilot_files(
    dataset_root: Path,
    report: dict,
    *,
    canonical: bool,
) -> tuple[dict[str, bytes], dict]:
    replay_kwargs = {
        "seed": int(report["pilot_seed"]),
        "refinement_rounds": int(report["refinement_rounds"]),
        "updates_per_round": int(report["updates_per_round"]),
        "final_updates": int(report["final_updates"]),
        "batch_size": int(report["batch_size"]),
        "max_groups": int(report["max_groups_per_round"]),
    }
    cgb, uniform, recomputed_report = run_pilot(
        dataset_root,
        canonical=canonical,
        **replay_kwargs,
    )
    return _published_pilot_files(
        cgb,
        uniform,
        recomputed_report,
        identity=report["scientific_identity"],
    )


def _process_parent_pid(process_id: int) -> int:
    status = Path(f"/proc/{process_id}/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", 1)[1].strip())
    result = subprocess.run(
        ["/bin/ps", "-o", "ppid=", "-p", str(process_id)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _validate_live_child_processes(
    children: list[dict],
    executions: tuple[dict, dict],
    replay_roots: tuple[Path, Path],
) -> list[dict]:
    if len(children) != 2:
        raise RuntimeError("canonical pilot requires two parent-owned children")
    records = []
    for expected_id, child, execution, replay_root in zip(
        ("a", "b"),
        children,
        executions,
        replay_roots,
        strict=True,
    ):
        process = child.get("process")
        if (
            child.get("replay_id") != expected_id
            or not isinstance(process, subprocess.Popen)
            or process.poll() is not None
            or process.pid != execution["process_id"]
            or _process_parent_pid(process.pid) != os.getpid()
            or child.get("command") != process.args
            or int(child.get("started_time_ns", 0)) <= 0
            or int(child.get("ready_time_ns", 0)) < int(execution["finished_time_ns"])
            or child.get("execution_sha256")
            != file_sha256(replay_root / "execution.json")
            or execution["slurm_job_id"] != os.environ.get("SLURM_JOB_ID")
        ):
            raise RuntimeError(
                "canonical replay is not a live parent-observed child execution"
            )
        records.append(
            {
                "replay_id": expected_id,
                "command": list(child["command"]),
                "observed_process_id": process.pid,
                "observed_parent_process_id": os.getpid(),
                "started_time_ns": int(child["started_time_ns"]),
                "ready_time_ns": int(child["ready_time_ns"]),
                "execution_sha256": child["execution_sha256"],
            }
        )
    return records


def _release_child_processes(
    children: list[dict],
    records: list[dict],
) -> None:
    for child in children:
        release_fd = int(child["release_fd"])
        os.write(release_fd, b"1")
        os.close(release_fd)
        child["release_fd"] = -1
    for child, record in zip(children, records, strict=True):
        process = child["process"]
        stdout, _ = process.communicate(timeout=60)
        if process.returncode != 0:
            raise RuntimeError("canonical pilot child failed after parent release")
        record["finished_time_ns"] = time.time_ns()
        record["return_code"] = process.returncode
        record["stdout_sha256"] = hashlib.sha256(stdout.encode("utf-8")).hexdigest()


def _validate_historical_orchestration(
    orchestration: dict,
    executions: list[dict],
    replay_records: list[dict],
) -> None:
    if (
        set(orchestration or {})
        != {
            "protocol",
            "parent_process_id",
            "hostname",
            "slurm_snapshot",
            "children",
        }
        or orchestration.get("protocol") != PILOT_ORCHESTRATION_PROTOCOL
    ):
        raise ValueError("pilot orchestration record has the wrong schema")
    parent_process_id = int(orchestration.get("parent_process_id", 0))
    hostname = orchestration.get("hostname")
    snapshot = orchestration.get("slurm_snapshot")
    try:
        _validate_slurm_snapshot_record(snapshot, role="producer")
    except (TypeError, ValueError) as error:
        raise ValueError("pilot orchestration identity is invalid") from error
    if (
        parent_process_id <= 0
        or hostname != CANONICAL_PILOT_ROLES["producer"]["hostname"]
    ):
        raise ValueError("pilot orchestration identity is invalid")
    children = orchestration.get("children")
    if not isinstance(children, list) or len(children) != 2:
        raise ValueError("pilot orchestration lacks two child records")
    expected_child_keys = {
        "replay_id",
        "command",
        "observed_process_id",
        "observed_parent_process_id",
        "started_time_ns",
        "ready_time_ns",
        "execution_sha256",
        "finished_time_ns",
        "return_code",
        "stdout_sha256",
    }
    for replay_id, child, execution, replay_record in zip(
        ("a", "b"),
        children,
        executions,
        replay_records,
        strict=True,
    ):
        command = child.get("command")
        if (
            set(child) != expected_child_keys
            or child.get("replay_id") != replay_id
            or not isinstance(command, list)
            or len(command) != 9
            or command[1:7]
            != [
                "-S",
                "-P",
                str(Path(__file__).resolve()),
                "pilot-replay-internal",
                "--replay-id",
                replay_id,
            ]
            or command[7] != "--hold-fd"
            or not str(command[8]).isdigit()
            or Path(command[0]).resolve()
            != Path(execution["python_executable"]).resolve()
            or child["observed_process_id"] != execution["process_id"]
            or child["observed_parent_process_id"] != parent_process_id
            or child["execution_sha256"] != replay_record.get("execution_sha256")
            or child["return_code"] != 0
            or not (
                int(child["started_time_ns"])
                <= int(execution["started_time_ns"])
                < int(execution["finished_time_ns"])
                <= int(child["ready_time_ns"])
                <= int(child["finished_time_ns"])
            )
            or execution["hostname"] != hostname
            or execution["slurm_snapshot"]["allocation"] != snapshot["allocation"]
            or not isinstance(child["stdout_sha256"], str)
            or len(child["stdout_sha256"]) != 64
        ):
            raise ValueError("pilot orchestration child binding is invalid")


def freeze_pilot_replays(
    first: Path,
    second: Path,
    out: Path,
    *,
    dataset_root: Path,
    canonical: bool = True,
    canonical_children: list[dict] | None = None,
) -> dict:
    if canonical:
        first = _lexical_absolute(first)
        second = _lexical_absolute(second)
        out = _lexical_absolute(out)
        dataset_root = _lexical_absolute(dataset_root)
    else:
        first = first.resolve()
        second = second.resolve()
        out = out.resolve()
        dataset_root = dataset_root.resolve()
    if len({first, second, out}) != 3:
        raise ValueError("pilot replay and frozen output paths must be distinct")
    if canonical and canonical_children is None:
        raise RuntimeError(
            "canonical pilot freeze requires live parent-owned child processes"
        )
    if canonical and (
        dataset_root != _canonical_path(CANONICAL_PILOT_DATASET)
        or first != _canonical_path(CANONICAL_PILOT_REPLAY_A)
        or second != _canonical_path(CANONICAL_PILOT_REPLAY_B)
        or out != _canonical_path(CANONICAL_PILOT_OUTPUT)
    ):
        raise ValueError("canonical pilot replay paths are frozen")
    first_report, first_execution = _validate_replay_output(
        first,
        canonical=canonical,
        replay_id="a",
        require_live_scheduler=canonical,
    )
    second_report, second_execution = _validate_replay_output(
        second,
        canonical=canonical,
        replay_id="b",
        require_live_scheduler=canonical,
    )
    if first_execution["execution_nonce"] == second_execution["execution_nonce"]:
        raise ValueError("pilot replays do not have independent execution receipts")
    if canonical and (
        first_execution["process_id"] == second_execution["process_id"]
        or first_execution["slurm_job_id"] != second_execution["slurm_job_id"]
        or first_execution["hostname"] != second_execution["hostname"]
    ):
        raise ValueError("canonical pilot replays lack two child-process executions")
    orchestration_records = None
    if canonical:
        orchestration_records = _validate_live_child_processes(
            canonical_children or [],
            (first_execution, second_execution),
            (first, second),
        )
    common_names = ("report.json", "cgb_schedule.jsonl", "uniform_schedule.jsonl")
    for name in common_names:
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise ValueError(f"pilot replays are not byte-identical: {name}")
    if first_report != second_report:
        raise AssertionError("byte-identical pilot reports parsed differently")
    executing_identity = None
    if canonical:
        executing_identity = scientific_identity(require_clean=True)
        if executing_identity != first_report["scientific_identity"]:
            raise ValueError("pilot replay identity changed before recomputation")

    expected_files, expected_report = _recompute_pilot_files(
        dataset_root,
        first_report,
        canonical=canonical,
    )
    for name in common_names:
        if (first / name).read_bytes() != expected_files[name]:
            raise ValueError(
                f"pilot replay differs from independent recomputation: {name}"
            )
    if first_report != expected_report:
        raise AssertionError(
            "independently reconstructed pilot report parsed differently"
        )
    if canonical and scientific_identity(require_clean=True) != executing_identity:
        raise RuntimeError("ACW scientific identity changed during recomputation")
    if canonical:
        second_observation = _validate_live_child_processes(
            canonical_children or [],
            (first_execution, second_execution),
            (first, second),
        )
        if second_observation != orchestration_records:
            raise RuntimeError("canonical child identity changed during recomputation")
        _release_child_processes(canonical_children or [], orchestration_records or [])
    if out.exists():
        raise FileExistsError(out)
    partial = out.with_name(out.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    try:
        common_files = {}
        for name in common_names:
            destination = partial / name
            shutil.copyfile(first / name, destination)
            common_files[name] = {
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        comparison = {
            "protocol": PILOT_COMPARISON_PROTOCOL,
            "reports_byte_identical": True,
            "schedules_byte_identical": True,
            "independently_recomputed": True,
            "independent_recomputation_sha256": {
                name: hashlib.sha256(expected_files[name]).hexdigest()
                for name in common_names
            },
            "dataset_manifest_payload_sha256": first_report[
                "dataset_manifest_payload_sha256"
            ],
            "scientific_identity": first_report["scientific_identity"],
            "orchestration": (
                {
                    "protocol": PILOT_ORCHESTRATION_PROTOCOL,
                    "parent_process_id": os.getpid(),
                    "hostname": socket.getfqdn(),
                    "slurm_snapshot": _slurm_snapshot(
                        required=True,
                        role="producer",
                    ),
                    "children": orchestration_records,
                }
                if canonical
                else None
            ),
            "common_files": common_files,
            "replays": [
                {
                    "replay_id": "a",
                    "path": CANONICAL_PILOT_REPLAY_A if canonical else str(first),
                    "execution_sha256": file_sha256(first / "execution.json"),
                    "execution_payload_sha256": first_execution["payload_sha256"],
                },
                {
                    "replay_id": "b",
                    "path": CANONICAL_PILOT_REPLAY_B if canonical else str(second),
                    "execution_sha256": file_sha256(second / "execution.json"),
                    "execution_payload_sha256": second_execution["payload_sha256"],
                },
            ],
        }
        comparison["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(comparison)
        ).hexdigest()
        comparison_path = partial / "replay_comparison.json"
        comparison_path.write_bytes(canonical_json_bytes(comparison) + b"\n")
        for path in partial.iterdir():
            path.chmod(0o444)
        if canonical and out != _canonical_path(CANONICAL_PILOT_OUTPUT):
            raise RuntimeError("canonical pilot path changed during publication")
        partial.replace(out)
        if canonical and out != _canonical_path(CANONICAL_PILOT_OUTPUT):
            raise RuntimeError("canonical pilot output became a symlink")
        out.chmod(0o555)
        return comparison
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _launch_held_replay(replay_id: str) -> dict:
    read_fd, write_fd = os.pipe()
    command = [
        sys.executable,
        "-S",
        "-P",
        str(Path(__file__).resolve()),
        "pilot-replay-internal",
        "--replay-id",
        replay_id,
        "--hold-fd",
        str(read_fd),
    ]
    environment = _canonical_pilot_environment()
    started_time_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            pass_fds=(read_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        os.close(read_fd)
    return {
        "replay_id": replay_id,
        "command": command,
        "process": process,
        "release_fd": write_fd,
        "started_time_ns": started_time_ns,
    }


def _wait_for_held_replay(child: dict, *, timeout_seconds: int = 82_800) -> None:
    replay_id = str(child["replay_id"])
    out = _canonical_path(
        CANONICAL_PILOT_REPLAY_A if replay_id == "a" else CANONICAL_PILOT_REPLAY_B
    )
    deadline = time.monotonic() + timeout_seconds
    process = child["process"]
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(
                f"canonical pilot child {replay_id} exited before custody check: "
                f"return_code={return_code} stdout_sha256="
                f"{hashlib.sha256(stdout.encode('utf-8')).hexdigest()}"
            )
        execution_path = out / "execution.json"
        if execution_path.is_file():
            child["ready_time_ns"] = time.time_ns()
            child["execution_sha256"] = file_sha256(execution_path)
            return
        time.sleep(0.25)
    raise TimeoutError(f"canonical pilot child {replay_id} did not become ready")


def _cleanup_held_replays(children: list[dict]) -> None:
    for child in children:
        release_fd = int(child.get("release_fd", -1))
        if release_fd >= 0:
            try:
                os.write(release_fd, b"1")
            except OSError:
                pass
            finally:
                os.close(release_fd)
                child["release_fd"] = -1
        process = child.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        if (
            isinstance(process, subprocess.Popen)
            and process.stdout is not None
            and not process.stdout.closed
        ):
            process.stdout.close()


def run_canonical_pilot() -> dict:
    """Own canonical data generation, two child fits, recomputation, and freeze."""
    start_runtime = require_canonical_pilot_runtime()
    if _canonical_pilot_role() != "producer":
        raise RuntimeError("canonical ACW pilot requires the producer role")
    _slurm_snapshot(required=True, role="producer")
    start_identity = scientific_identity(require_clean=True)
    dataset_root = _canonical_path(CANONICAL_PILOT_DATASET)
    first = _canonical_path(CANONICAL_PILOT_REPLAY_A)
    second = _canonical_path(CANONICAL_PILOT_REPLAY_B)
    out = _canonical_path(CANONICAL_PILOT_OUTPUT)
    verification = _canonical_path(CANONICAL_PILOT_VERIFICATION)
    for path in (dataset_root, first, second, out, verification):
        partial = path.with_name(path.name + ".partial")
        if (
            path.exists()
            or path.is_symlink()
            or partial.exists()
            or partial.is_symlink()
        ):
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    _regenerate_registered_dataset(
        dataset_root,
        development_seed_material(PILOT_SEED),
        {"kind": "pilot", "seed": PILOT_SEED},
    )
    verify_registered_dataset(dataset_root, allowed_kinds={"pilot"})
    children = []
    try:
        children = [_launch_held_replay(replay_id) for replay_id in ("a", "b")]
        for child in children:
            _wait_for_held_replay(child)
        comparison = freeze_pilot_replays(
            first,
            second,
            out,
            dataset_root=dataset_root,
            canonical=True,
            canonical_children=children,
        )
    finally:
        _cleanup_held_replays(children)
    if (
        scientific_identity(require_clean=True) != start_identity
        or require_canonical_pilot_runtime() != start_runtime
    ):
        raise RuntimeError("ACW scientific identity changed during pilot execution")
    load_pilot_report(out / "report.json")
    return comparison


def load_pilot_report(path: Path) -> dict:
    path = _lexical_absolute(path)
    if path != _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json":
        raise ValueError("canonical pilot report path is frozen")
    report = _validate_replay_report(path, canonical=True)
    comparison_path = path.parent / "replay_comparison.json"
    if comparison_path.is_symlink():
        raise ValueError("canonical pilot comparison may not be a symlink")
    comparison = _load_hash_bound_json(
        comparison_path,
        label="pilot replay comparison",
    )
    if set(comparison) != {
        "protocol",
        "reports_byte_identical",
        "schedules_byte_identical",
        "independently_recomputed",
        "independent_recomputation_sha256",
        "dataset_manifest_payload_sha256",
        "scientific_identity",
        "orchestration",
        "common_files",
        "replays",
        "payload_sha256",
    }:
        raise ValueError("pilot replay comparison has the wrong schema")
    if comparison.get("protocol") != PILOT_COMPARISON_PROTOCOL:
        raise ValueError("wrong pilot replay comparison protocol")
    if (
        comparison.get("reports_byte_identical") is not True
        or comparison.get("schedules_byte_identical") is not True
        or comparison.get("independently_recomputed") is not True
        or comparison.get("dataset_manifest_payload_sha256")
        != report["dataset_manifest_payload_sha256"]
        or comparison.get("scientific_identity") != report["scientific_identity"]
    ):
        raise ValueError("pilot replay comparison differs from the frozen report")
    common_files = comparison.get("common_files")
    if set(common_files or {}) != {
        "report.json",
        "cgb_schedule.jsonl",
        "uniform_schedule.jsonl",
    }:
        raise ValueError("pilot replay comparison common-file registry is incomplete")
    for name, record in common_files.items():
        target = path.parent / name
        if set(record) != {"bytes", "sha256"} or (
            target.stat().st_size != record["bytes"]
            or file_sha256(target) != record["sha256"]
        ):
            raise ValueError("frozen pilot file differs from replay comparison")
    recomputation_hashes = comparison.get("independent_recomputation_sha256")
    if set(recomputation_hashes or {}) != set(common_files):
        raise ValueError("pilot comparison lacks independent recomputation hashes")
    expected_files, expected_report = _recompute_pilot_files(
        _canonical_path(CANONICAL_PILOT_DATASET),
        report,
        canonical=True,
    )
    if scientific_identity(require_clean=True) != report["scientific_identity"]:
        raise RuntimeError("ACW scientific identity changed during report replay")
    for name in common_files:
        if (path.parent / name).read_bytes() != expected_files[
            name
        ] or recomputation_hashes[name] != hashlib.sha256(
            expected_files[name]
        ).hexdigest():
            raise ValueError(
                f"frozen pilot differs from fresh independent recomputation: {name}"
            )
    if report != expected_report:
        raise AssertionError("freshly recomputed pilot report parsed differently")
    replay_records = comparison.get("replays")
    if not isinstance(replay_records, list) or len(replay_records) != 2:
        raise ValueError("pilot replay comparison lacks two executions")
    historical_executions = []
    for expected_id, expected_path, record in zip(
        ("a", "b"),
        (CANONICAL_PILOT_REPLAY_A, CANONICAL_PILOT_REPLAY_B),
        replay_records,
        strict=True,
    ):
        if (
            set(record)
            != {
                "replay_id",
                "path",
                "execution_sha256",
                "execution_payload_sha256",
            }
            or record.get("replay_id") != expected_id
            or record.get("path") != expected_path
        ):
            raise ValueError("pilot replay comparison source registry differs")
        replay_root = _canonical_path(expected_path)
        replay_report, execution = _validate_replay_output(
            replay_root,
            canonical=True,
            replay_id=expected_id,
        )
        if replay_report != report or (
            file_sha256(replay_root / "execution.json")
            != record.get("execution_sha256")
            or execution["payload_sha256"] != record.get("execution_payload_sha256")
        ):
            raise ValueError("pilot replay source differs from frozen comparison")
        historical_executions.append(execution)
    _validate_historical_orchestration(
        comparison.get("orchestration"),
        historical_executions,
        replay_records,
    )
    if require_canonical_pilot_runtime() != report["canonical_runtime"]:
        raise RuntimeError("ACW runtime changed during report replay")
    return report


def _canonical_pilot_artifact_registry() -> dict[str, dict]:
    repository_root = Path(__file__).resolve().parents[1]
    registry = {}
    for relative in (
        CANONICAL_PILOT_DATASET,
        CANONICAL_PILOT_REPLAY_A,
        CANONICAL_PILOT_REPLAY_B,
        CANONICAL_PILOT_OUTPUT,
    ):
        root = _canonical_path(relative)
        _require_tree_without_symlinks(root)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                key = str(path.relative_to(repository_root))
                registry[key] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
    if len(registry) != CANONICAL_PILOT_ARTIFACT_FILES:
        raise ValueError("canonical pilot artifact registry has the wrong size")
    return registry


def _canonical_pilot_anchored_artifact_registry(
    validated_producer_artifacts: dict[str, dict] | None = None,
    validated_receipt_file: dict | None = None,
) -> dict[str, dict]:
    current_producer_artifacts = _canonical_pilot_artifact_registry()
    if (
        validated_producer_artifacts is not None
        and current_producer_artifacts != validated_producer_artifacts
    ):
        raise ValueError("producer artifacts changed after receipt validation")
    registry = dict(
        current_producer_artifacts
        if validated_producer_artifacts is None
        else validated_producer_artifacts
    )
    repository_root = Path(__file__).resolve().parents[1]
    verification_root = _canonical_path(CANONICAL_PILOT_VERIFICATION)
    _require_tree_without_symlinks(verification_root)
    files, directories = _dataset_tree_entries(verification_root)
    if files != {"verification.json"} or directories:
        raise ValueError("independent verification tree has the wrong registry")
    verification_path = verification_root / "verification.json"
    relative = str(verification_path.relative_to(repository_root))
    current_receipt_file = {
        "bytes": verification_path.stat().st_size,
        "sha256": file_sha256(verification_path),
    }
    if (
        validated_receipt_file is not None
        and current_receipt_file != validated_receipt_file
    ):
        raise ValueError("verification receipt changed after semantic validation")
    registry[relative] = dict(
        current_receipt_file
        if validated_receipt_file is None
        else validated_receipt_file
    )
    if len(registry) != CANONICAL_PILOT_ANCHORED_FILES:
        raise ValueError("anchored pilot artifact registry has the wrong size")
    return registry


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_independent_pilot_verification(
    *,
    expected_receipt: dict,
    expected_receipt_bytes: bytes,
) -> tuple[dict, dict, dict, dict, dict]:
    """Reopen the receipt while retaining its verifier-process provenance."""
    if (
        not isinstance(expected_receipt, dict)
        or not isinstance(expected_receipt_bytes, bytes)
        or expected_receipt_bytes != canonical_json_bytes(expected_receipt) + b"\n"
    ):
        raise ValueError("verifier-process receipt provenance is invalid")
    runtime = require_canonical_pilot_runtime()
    if _canonical_pilot_role() != "verifier":
        raise RuntimeError("pilot verification receipt requires the verifier role")
    identity = scientific_identity(require_clean=True)
    verification_root = _canonical_path(CANONICAL_PILOT_VERIFICATION)
    _require_tree_without_symlinks(verification_root)
    files, directories = _dataset_tree_entries(verification_root)
    if files != {"verification.json"} or directories:
        raise ValueError("independent verification tree has the wrong registry")
    receipt_path = verification_root / "verification.json"
    receipt_raw = receipt_path.read_bytes()
    if receipt_raw != expected_receipt_bytes:
        raise RuntimeError("verification receipt differs from verifier-process bytes")
    receipt = _load_hash_bound_json_bytes(
        receipt_raw,
        label="independent pilot verification",
        require_canonical_bytes=True,
    )
    if receipt != expected_receipt:
        raise RuntimeError("verification receipt differs from verifier-process record")
    receipt_file = {
        "bytes": len(expected_receipt_bytes),
        "sha256": hashlib.sha256(expected_receipt_bytes).hexdigest(),
    }
    expected_keys = {
        "protocol",
        "scientific_identity",
        "canonical_runtime",
        "process_id",
        "hostname",
        "python_executable",
        "started_time_ns",
        "finished_time_ns",
        "elapsed_wall_ns",
        "elapsed_monotonic_ns",
        "peak_rss_kib",
        "producer",
        "verifier_slurm_snapshot_start",
        "verifier_slurm_snapshot_finish",
        "dataset_manifest_payload_sha256",
        "pilot_report_payload_sha256",
        "pilot_report_sha256",
        "artifact_files",
        "artifact_files_payload_sha256",
        "artifact_file_count",
        "fresh_recomputation_complete",
        "claim_boundary",
        "payload_sha256",
    }
    producer = receipt.get("producer")
    if not isinstance(producer, dict) or set(producer) != {
        "hostname",
        "slurm_snapshot",
        "comparison_payload_sha256",
    }:
        raise ValueError("independent verification producer has the wrong schema")
    verifier_start = receipt.get("verifier_slurm_snapshot_start")
    verifier_finish = receipt.get("verifier_slurm_snapshot_finish")
    _validate_independent_verifier_allocation(
        producer["slurm_snapshot"],
        verifier_start,
        producer_hostname=producer["hostname"],
        verifier_hostname=receipt.get("hostname"),
    )
    _validate_independent_verifier_allocation(
        producer["slurm_snapshot"],
        verifier_finish,
        producer_hostname=producer["hostname"],
        verifier_hostname=receipt.get("hostname"),
    )
    if (
        set(receipt) != expected_keys
        or receipt.get("protocol") != PILOT_INDEPENDENT_VERIFICATION_PROTOCOL
        or receipt.get("scientific_identity") != identity
        or receipt.get("canonical_runtime") != runtime
        or int(receipt.get("process_id", 0)) <= 0
        or receipt.get("hostname") != CANONICAL_PILOT_ROLES["verifier"]["hostname"]
        or receipt.get("python_executable") != runtime["python_executable"]
        or int(receipt.get("started_time_ns", 0)) <= 0
        or int(receipt.get("finished_time_ns", 0))
        <= int(receipt.get("started_time_ns", 0))
        or int(receipt.get("elapsed_wall_ns", 0))
        != int(receipt.get("finished_time_ns", 0))
        - int(receipt.get("started_time_ns", 0))
        or int(receipt.get("elapsed_monotonic_ns", 0)) <= 0
        or int(receipt.get("peak_rss_kib", 0)) <= 0
        or verifier_start["allocation"] != verifier_finish["allocation"]
        or receipt.get("fresh_recomputation_complete") is not True
        or receipt.get("claim_boundary") != CANONICAL_PILOT_VERIFICATION_CLAIM
        or not _valid_sha256(producer.get("comparison_payload_sha256"))
    ):
        raise ValueError("independent verification receipt is invalid")
    report_path = _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json"
    comparison_path = report_path.parent / "replay_comparison.json"
    report = load_pilot_report(report_path)
    comparison = _load_hash_bound_json(
        comparison_path,
        label="pilot replay comparison",
    )
    artifact_files = _canonical_pilot_artifact_registry()
    if (
        receipt.get("dataset_manifest_payload_sha256")
        != report["dataset_manifest_payload_sha256"]
        or receipt.get("pilot_report_payload_sha256") != report["payload_sha256"]
        or receipt.get("pilot_report_sha256") != file_sha256(report_path)
        or producer.get("comparison_payload_sha256") != comparison["payload_sha256"]
        or receipt.get("artifact_files") != artifact_files
        or receipt.get("artifact_file_count") != len(artifact_files)
        or receipt.get("artifact_files_payload_sha256")
        != hashlib.sha256(canonical_json_bytes(artifact_files)).hexdigest()
    ):
        raise ValueError("independent verification receipt differs from pilot bytes")
    return receipt, receipt_file, report, comparison, artifact_files


def _validate_pilot_artifact_registry_record(
    registry: dict,
    *,
    identity: dict,
    receipt: dict,
    report: dict,
    comparison: dict,
    artifact_files: dict,
) -> None:
    expected_keys = {
        "protocol",
        "scientific_identity",
        "canonical_paths",
        "dataset_manifest_payload_sha256",
        "pilot_report_payload_sha256",
        "pilot_report_sha256",
        "pilot_replay_comparison_payload_sha256",
        "pilot_replay_comparison_sha256",
        "independent_verification_payload_sha256",
        "independent_verification_sha256",
        "artifact_files",
        "artifact_file_count",
        "artifact_files_payload_sha256",
        "activation_allowlist",
        "claim_boundary",
        "payload_sha256",
    }
    repository_root = Path(__file__).resolve().parents[1]
    report_path = _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json"
    comparison_path = report_path.parent / "replay_comparison.json"
    verification_path = (
        _canonical_path(CANONICAL_PILOT_VERIFICATION) / "verification.json"
    )
    expected_paths = {
        "dataset": CANONICAL_PILOT_DATASET,
        "pilot": CANONICAL_PILOT_OUTPUT,
        "replay_a": CANONICAL_PILOT_REPLAY_A,
        "replay_b": CANONICAL_PILOT_REPLAY_B,
        "verification": CANONICAL_PILOT_VERIFICATION,
    }
    if (
        set(registry) != expected_keys
        or registry.get("protocol") != PILOT_ARTIFACT_REGISTRY_PROTOCOL
        or registry.get("scientific_identity") != identity
        or registry.get("canonical_paths") != expected_paths
        or registry.get("dataset_manifest_payload_sha256")
        != report["dataset_manifest_payload_sha256"]
        or registry.get("pilot_report_payload_sha256") != report["payload_sha256"]
        or registry.get("pilot_report_sha256") != file_sha256(report_path)
        or registry.get("pilot_replay_comparison_payload_sha256")
        != comparison["payload_sha256"]
        or registry.get("pilot_replay_comparison_sha256")
        != file_sha256(comparison_path)
        or registry.get("independent_verification_payload_sha256")
        != receipt["payload_sha256"]
        or registry.get("independent_verification_sha256")
        != file_sha256(verification_path)
        or registry.get("artifact_files") != artifact_files
        or registry.get("artifact_file_count") != len(artifact_files)
        or registry.get("artifact_files_payload_sha256")
        != hashlib.sha256(canonical_json_bytes(artifact_files)).hexdigest()
        or registry.get("activation_allowlist")
        != list(CANONICAL_PILOT_ACTIVATION_ALLOWLIST)
        or registry.get("claim_boundary") != CANONICAL_PILOT_REGISTRY_CLAIM
        or not all(
            (repository_root / relative).is_file() for relative in artifact_files
        )
    ):
        raise ValueError("pilot artifact registry differs from verified artifacts")


def build_canonical_pilot_artifact_registry(
    *,
    expected_receipt: dict,
    expected_receipt_bytes: bytes,
) -> dict:
    """Build A from the receipt still held by the verifier process."""
    receipt, receipt_file, report, comparison, producer_artifacts = (
        load_independent_pilot_verification(
            expected_receipt=expected_receipt,
            expected_receipt_bytes=expected_receipt_bytes,
        )
    )
    identity = scientific_identity(require_clean=True)
    artifact_files = _canonical_pilot_anchored_artifact_registry(
        producer_artifacts,
        receipt_file,
    )
    out = _canonical_path(CANONICAL_PILOT_REGISTRY)
    partial = out.with_name(out.name + ".partial")
    if out.exists() or out.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(out)
    registry = {
        "protocol": PILOT_ARTIFACT_REGISTRY_PROTOCOL,
        "scientific_identity": identity,
        "canonical_paths": {
            "dataset": CANONICAL_PILOT_DATASET,
            "pilot": CANONICAL_PILOT_OUTPUT,
            "replay_a": CANONICAL_PILOT_REPLAY_A,
            "replay_b": CANONICAL_PILOT_REPLAY_B,
            "verification": CANONICAL_PILOT_VERIFICATION,
        },
        "dataset_manifest_payload_sha256": report["dataset_manifest_payload_sha256"],
        "pilot_report_payload_sha256": report["payload_sha256"],
        "pilot_report_sha256": file_sha256(
            _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json"
        ),
        "pilot_replay_comparison_payload_sha256": comparison["payload_sha256"],
        "pilot_replay_comparison_sha256": file_sha256(
            _canonical_path(CANONICAL_PILOT_OUTPUT) / "replay_comparison.json"
        ),
        "independent_verification_payload_sha256": receipt["payload_sha256"],
        "independent_verification_sha256": file_sha256(
            _canonical_path(CANONICAL_PILOT_VERIFICATION) / "verification.json"
        ),
        "artifact_files": artifact_files,
        "artifact_file_count": len(artifact_files),
        "artifact_files_payload_sha256": hashlib.sha256(
            canonical_json_bytes(artifact_files)
        ).hexdigest(),
        "activation_allowlist": list(CANONICAL_PILOT_ACTIVATION_ALLOWLIST),
        "claim_boundary": CANONICAL_PILOT_REGISTRY_CLAIM,
    }
    registry["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(registry)
    ).hexdigest()
    registry_bytes = canonical_json_bytes(registry) + b"\n"
    try:
        partial.write_bytes(registry_bytes)
        partial.chmod(0o444)
        partial.replace(out)
        if out.is_symlink():
            raise RuntimeError("pilot artifact registry became a symlink")
        final_receipt_path = (
            _canonical_path(CANONICAL_PILOT_VERIFICATION) / "verification.json"
        )
        final_receipt_bytes = final_receipt_path.read_bytes()
        if final_receipt_bytes != expected_receipt_bytes:
            raise RuntimeError(
                "verification receipt changed while building its registry"
            )
        final_receipt = _load_hash_bound_json_bytes(
            final_receipt_bytes,
            label="independent pilot verification",
            require_canonical_bytes=True,
        )
        final_registry_bytes = out.read_bytes()
        if final_registry_bytes != registry_bytes:
            raise RuntimeError("pilot artifact registry changed after publication")
        final_artifact_files = _canonical_pilot_anchored_artifact_registry(
            producer_artifacts,
            receipt_file,
        )
        if (
            final_receipt != receipt
            or scientific_identity(require_clean=True) != identity
            or final_artifact_files != artifact_files
        ):
            raise RuntimeError("pilot artifacts changed while building their registry")
        _validate_pilot_artifact_registry_record(
            _load_hash_bound_json_bytes(
                final_registry_bytes,
                label="pilot artifact registry",
                require_canonical_bytes=True,
            ),
            identity=identity,
            receipt=receipt,
            report=report,
            comparison=comparison,
            artifact_files=final_artifact_files,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        raise
    return registry


def _validate_independent_verifier_allocation(
    producer_snapshot: dict,
    verifier_snapshot: dict,
    *,
    producer_hostname: str,
    verifier_hostname: str,
) -> None:
    try:
        producer = _validate_slurm_snapshot_record(
            producer_snapshot,
            role="producer",
        )
        verifier = _validate_slurm_snapshot_record(
            verifier_snapshot,
            role="verifier",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "independent pilot verification requires canonical producer and verifier jobs"
        ) from error
    if (
        verifier["job_id"] != os.environ.get("SLURM_JOB_ID")
        or verifier["num_cpus"] != int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        or producer_hostname != CANONICAL_PILOT_ROLES["producer"]["hostname"]
        or verifier_hostname != CANONICAL_PILOT_ROLES["verifier"]["hostname"]
        or producer.get("job_id") == verifier.get("job_id")
        or producer["node_list"] != CANONICAL_PILOT_ROLES["producer"]["node_list"]
        or verifier["node_list"] != CANONICAL_PILOT_ROLES["verifier"]["node_list"]
    ):
        raise ValueError(
            "independent pilot verification requires canonical producer and verifier jobs"
        )


def verify_canonical_pilot_independently() -> dict:
    """Recompute v5 in a separate Slurm allocation and freeze its receipt."""
    runtime = require_canonical_pilot_runtime()
    if _canonical_pilot_role() != "verifier":
        raise RuntimeError(
            "canonical ACW independent replay requires the verifier role"
        )
    identity = scientific_identity(require_clean=True)
    out = _canonical_path(CANONICAL_PILOT_VERIFICATION)
    partial = out.with_name(out.name + ".partial")
    if out.exists() or out.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(out)
    started_time_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    report_path = _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json"
    comparison_path = report_path.parent / "replay_comparison.json"
    comparison = _load_hash_bound_json(
        comparison_path,
        label="pilot replay comparison",
    )
    producer_orchestration = comparison.get("orchestration")
    if not isinstance(producer_orchestration, dict):
        raise ValueError("pilot comparison lacks producer orchestration")
    producer_snapshot = producer_orchestration.get("slurm_snapshot")
    verifier_snapshot_start = _slurm_snapshot(required=True, role="verifier")
    _validate_independent_verifier_allocation(
        producer_snapshot,
        verifier_snapshot_start,
        producer_hostname=producer_orchestration.get("hostname"),
        verifier_hostname=socket.getfqdn(),
    )
    report = load_pilot_report(report_path)
    artifact_files = _canonical_pilot_artifact_registry()
    final_runtime = require_canonical_pilot_runtime()
    verifier_snapshot_finish = _slurm_snapshot(required=True, role="verifier")
    _validate_independent_verifier_allocation(
        producer_snapshot,
        verifier_snapshot_finish,
        producer_hostname=producer_orchestration.get("hostname"),
        verifier_hostname=socket.getfqdn(),
    )
    final_comparison = _load_hash_bound_json(
        comparison_path,
        label="pilot replay comparison",
    )
    final_report = _load_hash_bound_json(report_path, label="pilot report")
    final_artifact_files = _canonical_pilot_artifact_registry()
    if (
        scientific_identity(require_clean=True) != identity
        or final_runtime != runtime
        or verifier_snapshot_finish["allocation"]
        != verifier_snapshot_start["allocation"]
        or final_comparison != comparison
        or final_report != report
        or final_artifact_files != artifact_files
    ):
        raise RuntimeError("ACW pilot or runtime changed during verification")
    finished_monotonic_ns = time.monotonic_ns()
    finished_time_ns = time.time_ns()
    receipt = {
        "protocol": PILOT_INDEPENDENT_VERIFICATION_PROTOCOL,
        "scientific_identity": identity,
        "canonical_runtime": runtime,
        "process_id": os.getpid(),
        "hostname": socket.getfqdn(),
        "python_executable": str(Path(sys.executable).resolve()),
        "started_time_ns": started_time_ns,
        "finished_time_ns": finished_time_ns,
        "elapsed_wall_ns": finished_time_ns - started_time_ns,
        "elapsed_monotonic_ns": finished_monotonic_ns - started_monotonic_ns,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "producer": {
            "hostname": producer_orchestration["hostname"],
            "slurm_snapshot": producer_snapshot,
            "comparison_payload_sha256": comparison["payload_sha256"],
        },
        "verifier_slurm_snapshot_start": verifier_snapshot_start,
        "verifier_slurm_snapshot_finish": verifier_snapshot_finish,
        "dataset_manifest_payload_sha256": report["dataset_manifest_payload_sha256"],
        "pilot_report_payload_sha256": report["payload_sha256"],
        "pilot_report_sha256": file_sha256(report_path),
        "artifact_files": artifact_files,
        "artifact_files_payload_sha256": hashlib.sha256(
            canonical_json_bytes(artifact_files)
        ).hexdigest(),
        "artifact_file_count": len(artifact_files),
        "fresh_recomputation_complete": True,
        "claim_boundary": CANONICAL_PILOT_VERIFICATION_CLAIM,
    }
    receipt["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    partial.mkdir(parents=True)
    try:
        receipt_path = partial / "verification.json"
        receipt_path.write_bytes(receipt_bytes)
        receipt_path.chmod(0o444)
        if out != _canonical_path(CANONICAL_PILOT_VERIFICATION):
            raise RuntimeError("independent verification path changed")
        partial.replace(out)
        if out != _canonical_path(CANONICAL_PILOT_VERIFICATION):
            raise RuntimeError("independent verification output became a symlink")
        out.chmod(0o555)
        if (out / "verification.json").read_bytes() != receipt_bytes:
            raise RuntimeError("independent verification receipt changed after freeze")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return receipt


def verify_and_build_canonical_pilot_artifact_registry() -> tuple[dict, dict]:
    """Verify the pilot and publish A without dropping receipt provenance."""
    receipt = verify_canonical_pilot_independently()
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    registry = build_canonical_pilot_artifact_registry(
        expected_receipt=receipt,
        expected_receipt_bytes=receipt_bytes,
    )
    return receipt, registry


def _validate_scored_seed_identity(seed_identity: dict) -> None:
    kind = seed_identity.get("kind")
    if kind == "development":
        if seed_identity != {"kind": "development", "seed": seed_identity.get("seed")}:
            raise ValueError("scored development identity has the wrong schema")
        if int(seed_identity["seed"]) not in DEVELOPMENT_SEEDS:
            raise ValueError(
                "scored development identity is outside the frozen registry"
            )
        return
    if kind == "confirmation":
        if set(seed_identity) != {"kind", "index", "commitment"}:
            raise ValueError("scored confirmation identity has the wrong schema")
        index = int(seed_identity["index"])
        if not 0 <= index < len(CONFIRMATION_COMMITMENTS):
            raise ValueError(
                "scored confirmation identity is outside the frozen registry"
            )
        if seed_identity["commitment"] != CONFIRMATION_COMMITMENTS[index]:
            raise ValueError(
                "scored confirmation commitment is outside the frozen registry"
            )
        return
    raise ValueError(
        "canonical scored bundle may not use a pilot or unregistered domain"
    )


def _write_array(path: Path, array: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": file_sha256(path),
    }


def _require_committed_pilot_anchor() -> None:
    raise RuntimeError(CANONICAL_BUNDLE_BLOCK)


def build_trainer_bundle(
    dataset_root: Path,
    schedule_path: Path,
    out: Path,
    *,
    canonical: bool = True,
    pilot_report_path: Path | None = None,
) -> dict:
    dataset_root = dataset_root.resolve()
    schedule_path = schedule_path.resolve()
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if canonical:
        _require_committed_pilot_anchor()
    source_manifest = _load_manifest(dataset_root)
    data_replay_verification = None
    if canonical:
        _validate_scored_seed_identity(source_manifest.get("seed_identity", {}))
        if source_manifest["seed_identity"].get("kind") == "development":
            data_replay_verification = verify_registered_dataset(
                dataset_root,
                allowed_kinds={"development"},
            )
    data = load_public_training_data(dataset_root, reject_oracle=False)
    _, oracle_answers, loaded_manifest = load_oracle_truth(dataset_root)
    if loaded_manifest != source_manifest:
        raise RuntimeError("dataset manifest changed during bundle construction")
    pilot_report = None
    pilot_comparison = None
    schedule_kind = "noncanonical"
    if canonical:
        if pilot_report_path is None:
            raise ValueError(
                "canonical trainer bundle requires the frozen pilot report"
            )
        pilot_report_path = pilot_report_path.resolve()
        pilot_report = load_pilot_report(pilot_report_path)
        pilot_comparison = _load_hash_bound_json(
            pilot_report_path.parent / "replay_comparison.json",
            label="pilot replay comparison",
        )
        schedule_kind = schedule_path.name
        if schedule_kind not in {"cgb_schedule.jsonl", "uniform_schedule.jsonl"}:
            raise ValueError("canonical schedule has an unregistered filename")
        if schedule_path != pilot_report_path.parent / schedule_kind:
            raise ValueError(
                "canonical schedule is not the pilot report's bound sibling"
            )
        schedule_record = pilot_report["schedules"][schedule_kind]
        if set(schedule_record) != {"bytes", "rows", "sha256"}:
            raise ValueError("pilot schedule binding has the wrong schema")
        if (
            schedule_path.stat().st_size != schedule_record["bytes"]
            or file_sha256(schedule_path) != schedule_record["sha256"]
            or schedule_record["rows"] != CANONICAL_LABELS
        ):
            raise ValueError("canonical schedule differs from the frozen pilot report")
    schedule = load_query_schedule(schedule_path)
    validate_query_schedule(
        schedule,
        data.histories,
        refinement_rounds=REFINEMENT_ROUNDS
        if canonical
        else max(row["round"] for row in schedule),
        canonical=canonical,
    )
    partial = out.with_name(out.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.mkdir(parents=True)
    arrays = {}
    try:
        round_zero = [row for row in schedule if row["round"] == 0]
        initial_queries = np.empty((data.histories, 2), dtype=np.int8)
        initial_answers = np.empty_like(initial_queries)
        grouped_initial = [[] for _ in range(data.histories)]
        for row in round_zero:
            grouped_initial[row["history_id"]].append(row["query_id"])
        for history_id, queries in enumerate(grouped_initial):
            queries = sorted(queries)
            if len(queries) != 2:
                raise ValueError("bundle schedule lacks two round-zero queries")
            initial_queries[history_id] = queries
            initial_answers[history_id] = oracle_answers[history_id, queries]

        for relative in PUBLIC_ARRAYS:
            destination = partial / relative
            if relative.endswith("initial_queries.npy"):
                arrays[relative] = _write_array(destination, initial_queries)
            elif relative.endswith("initial_answers.npy"):
                arrays[relative] = _write_array(destination, initial_answers)
            else:
                source = dataset_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                source_record = source_manifest["arrays"][relative]
                if file_sha256(destination) != source_record["sha256"]:
                    raise RuntimeError(f"bundle copy hash mismatch: {relative}")
                arrays[relative] = dict(source_record)

        curriculum_rows = [
            {
                "history_id": row["history_id"],
                "query_id": row["query_id"],
                "answer": int(oracle_answers[row["history_id"], row["query_id"]]),
                "round": row["round"],
            }
            for row in schedule
        ]
        curriculum_path = partial / "curriculum.jsonl"
        curriculum_path.write_bytes(
            b"".join(canonical_json_bytes(row) + b"\n" for row in curriculum_rows)
        )
        if curriculum_query_schedule_sha256(curriculum_path) != file_sha256(
            schedule_path
        ):
            raise RuntimeError(
                "curriculum-derived schedule differs from the selected pilot schedule"
            )
        files = {
            "curriculum.jsonl": {
                "bytes": curriculum_path.stat().st_size,
                "rows": len(curriculum_rows),
                "sha256": file_sha256(curriculum_path),
            }
        }
        pilot_artifacts = None
        if canonical:
            pilot_artifacts = {}
            pilot_sources = {
                "pilot/report.json": pilot_report_path,
                "pilot/replay_comparison.json": (
                    pilot_report_path.parent / "replay_comparison.json"
                ),
                "pilot/cgb_schedule.jsonl": (
                    pilot_report_path.parent / "cgb_schedule.jsonl"
                ),
                "pilot/uniform_schedule.jsonl": (
                    pilot_report_path.parent / "uniform_schedule.jsonl"
                ),
            }
            for relative, source in pilot_sources.items():
                destination = partial / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                pilot_artifacts[relative] = {
                    "bytes": destination.stat().st_size,
                    "sha256": file_sha256(destination),
                }
        manifest = {
            "protocol": BUNDLE_PROTOCOL,
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "seed_identity": source_manifest["seed_identity"],
            "data_replay_verification": data_replay_verification,
            "query_schedule_sha256": file_sha256(schedule_path),
            "query_schedule_kind": schedule_kind,
            "pilot_report_payload_sha256": (
                pilot_report["payload_sha256"] if pilot_report is not None else None
            ),
            "pilot_report_sha256": (
                file_sha256(pilot_report_path)
                if pilot_report_path is not None
                else None
            ),
            "pilot_replay_comparison_payload_sha256": (
                pilot_comparison["payload_sha256"]
                if pilot_comparison is not None
                else None
            ),
            "pilot_replay_comparison_sha256": (
                file_sha256(pilot_report_path.parent / "replay_comparison.json")
                if pilot_report_path is not None
                else None
            ),
            "pilot_artifacts": pilot_artifacts,
            "arrays": arrays,
            "files": files,
            "oracle_paths_exported": 0,
        }
        manifest["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        (partial / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        if any(
            "oracle" in str(path.relative_to(partial)).lower()
            for path in partial.rglob("*")
        ):
            raise RuntimeError("trainer bundle contains an oracle-named path")
        for path in partial.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        partial.replace(out)
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        out.chmod(0o555)
        return manifest
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pilot-run")
    replay = subparsers.add_parser("pilot-replay-internal", help=argparse.SUPPRESS)
    replay.add_argument("--replay-id", choices=("a", "b"), required=True)
    replay.add_argument("--hold-fd", type=int, required=True, help=argparse.SUPPRESS)
    subparsers.add_parser("runtime-identity-internal", help=argparse.SUPPRESS)
    subparsers.add_parser("verify-pilot")
    subparsers.add_parser("verify-pilot-independent")
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--dataset", type=Path, required=True)
    bundle.add_argument("--schedule", type=Path, required=True)
    bundle.add_argument("--pilot-report", type=Path, required=True)
    bundle.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pilot-run":
        comparison = run_canonical_pilot()
        print(
            "[acw-pilot-run] byte_identical=1 independently_recomputed=1 "
            f"payload_sha256={comparison['payload_sha256']}"
        )
    elif args.command == "pilot-replay-internal":
        expected_out = (
            CANONICAL_PILOT_REPLAY_A
            if args.replay_id == "a"
            else CANONICAL_PILOT_REPLAY_B
        )
        report = execute_pilot_replay(
            _canonical_path(CANONICAL_PILOT_DATASET),
            _canonical_path(expected_out),
            replay_id=args.replay_id,
            canonical=True,
            hold_fd=args.hold_fd,
        )
        print(
            f"[acw-pilot-replay-{args.replay_id}] labels={report['labels']} "
            f"payload_sha256={report['payload_sha256']}"
        )
    elif args.command == "runtime-identity-internal":
        _canonical_pilot_environment(require_committed_batch_script=False)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        first = pilot_runtime_identity()
        second = pilot_runtime_identity()
        if first != second:
            raise RuntimeError("uncached ACW runtime identity changed within one job")
        if first != CANONICAL_PILOT_RUNTIME:
            raise RuntimeError("ACW runtime probe differs from its canonical pin")
        print(
            hashlib.sha256(canonical_json_bytes(first)).hexdigest(),
            json.dumps(first, sort_keys=True, separators=(",", ":")),
        )
    elif args.command == "verify-pilot":
        report = load_pilot_report(
            _canonical_path(CANONICAL_PILOT_OUTPUT) / "report.json"
        )
        print(
            f"[acw-pilot-verify] complete=1 payload_sha256={report['payload_sha256']}"
        )
    elif args.command == "verify-pilot-independent":
        receipt, registry = verify_and_build_canonical_pilot_artifact_registry()
        print(
            "[acw-pilot-independent-verify] complete=1 different_job=1 "
            f"different_node=1 receipt_payload_sha256={receipt['payload_sha256']} "
            f"files={registry['artifact_file_count']} "
            f"registry_payload_sha256={registry['payload_sha256']}"
        )
    else:
        manifest = build_trainer_bundle(
            args.dataset,
            args.schedule,
            args.out,
            canonical=True,
            pilot_report_path=args.pilot_report,
        )
        print(f"[acw-bundle] payload_sha256={manifest['payload_sha256']}")


if __name__ == "__main__":
    main()
