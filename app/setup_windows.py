# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""Windows installation and read-only environment checks. Run via install.bat."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent.parent
# Match the tested network, media and weight-reading library versions in requirements.txt. triton-windows (the
# compiler of the engine's GPU kernels) is the build for the PyTorch that install() pins below: change them together.
ENGINE_PACKAGES = ('av', 'diffusers', 'transformers', 'tokenizers', 'safetensors', 'gguf', 'sentencepiece',
                   'einops', 'scipy', 'triton-windows')
# VoxCPM declares funasr but never imports it. funasr pulls in kaldiio, whose license allows
# evaluation only (docs/LICENSING.md), so VoxCPM is installed without its own dependency list
# and both packages are removed from earlier installations.
VOXCPM = 'voxcpm==2.0.3'
LLAMA_WHEEL = ('https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-vulkan/'
               'llama_cpp_python-0.3.36-py3-none-win_amd64.whl'
               '#sha256=b3d58d70e9fa6216f3d1919bb6141a98eb22eb889e14a44d080dd250e969fab4')
RUNTIME_PINS = ('mediapipe==1.1.1', 'protobuf==7.36.2', 'accelerate==1.15.0')
UNUSED_PACKAGES = ('funasr', 'kaldiio')
_UNUSED_REQUIREMENT = re.compile(r'voxcpm \S+ requires funasr, which is not installed\.')


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, cwd=ROOT, **kwargs)


def prepare_venv():
    """Repair an incompatible environment by preserving it before recreation."""
    environment = ROOT / "venv"
    interpreter = environment / "Scripts/python.exe"
    if interpreter.is_file():
        try:
            result = subprocess.run([str(interpreter), "-c",
                "import sys; assert sys.version_info[:2] in ((3,11),(3,12)) and sys.maxsize>2**32"],
                cwd=ROOT, capture_output=True, timeout=30, check=False)
            if result.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    if environment.exists() or environment.is_symlink():
        expected = ROOT.resolve() / "venv"
        if environment.resolve() != expected or environment.is_symlink():
            raise RuntimeError("Refusing to replace a redirected venv folder. Preserve it manually and retry.")
        preserved = ROOT / ("venv.preserved_" + uuid.uuid4().hex)
        environment.rename(preserved)
        print(f"[OK] Previous environment preserved at {preserved}")
    run([sys.executable, "-m", "venv", environment])


def engine_pins():
    """The `name==version` lines of requirements.txt for ENGINE_PACKAGES (without an environment marker)."""
    pins = []
    requirements = ROOT / 'requirements.txt'
    for line in (requirements.read_text(encoding='utf-8').splitlines() if requirements.is_file() else []):
        name, _, version = line.split('#', 1)[0].split(';', 1)[0].strip().partition('==')
        if version and name.strip().lower() in ENGINE_PACKAGES:
            pins.append(f'{name.strip()}=={version.strip()}')
    return pins


def constraints_text():
    """The pip constraints: the CUDA torch triple as installed, and the pinned engine libraries."""
    torch_lines = ''.join(n + '==' + importlib.metadata.version(n) + '\n' for n in ('torch', 'torchvision', 'torchaudio'))
    return torch_lines + ''.join(pin + '\n' for pin in engine_pins())


def voxcpm_requirements():
    """VoxCPM's declared dependencies, without funasr and the development extras."""
    listed = run([sys.executable, '-c',
                  'import importlib.metadata as m; print(*(m.requires("voxcpm") or []), sep="\\n")'],
                 capture_output=True, text=True).stdout.splitlines()
    return [line.strip() for line in listed if line.strip() and 'extra ==' not in line
            and re.split(r'[\s;<>=!~\[(]', line.strip(), maxsplit=1)[0].lower() not in UNUSED_PACKAGES]


def broken_requirements():
    """Problems reported by pip check, except the funasr requirement that VoxCPM never uses."""
    output = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout
    return [line.strip() for line in output.splitlines() if line.strip()
            and not line.startswith('No broken requirements')
            and not _UNUSED_REQUIREMENT.fullmatch(line.strip())]


def check_environment(cuda=True):
    errors = []
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or sys.maxsize <= 2**32:
        errors.append('Use Python 3.11 or 3.12 (64-bit).')
    for package in ('gradio', 'requests', 'Pillow', 'huggingface_hub', 'voxcpm', 'opencv-python'):
        try:
            print(f'[OK] {package} {importlib.metadata.version(package)}')
        except importlib.metadata.PackageNotFoundError:
            errors.append(f'{package} is missing. Run install.bat.')
    pins = [*engine_pins(), *RUNTIME_PINS, 'llama-cpp-python==0.3.36']
    for pin in pins:
        name, _, wanted = pin.partition('==')
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found = 'missing'
        if found != wanted:
            errors.append(f'{name} is {found}; the engine requires {wanted}. Run install.bat.')
    broken = broken_requirements()
    errors += [f'{line} Run install.bat.' for line in broken]
    if not broken:
        print('[OK] Python dependencies are consistent.')
    if cuda:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append('CUDA is unavailable. Install an NVIDIA driver and run install.bat.')
            else:
                torch.ones(1, device='cuda')
                torch.cuda.synchronize()
                print(f'[OK] CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}')
        except Exception as exc:
            errors.append(f'CUDA check failed: {exc}')
    if shutil.disk_usage(ROOT).free < 40 * 1024**3:
        print('[WARNING] Less than 40 GB free. Video model bundles may need more space.')
    for error in errors:
        print('[ERROR]', error)
    return not errors


def install():
    from . import config

    if sys.version_info[:2] not in ((3, 11), (3, 12)) or sys.maxsize <= 2**32:
        raise RuntimeError('Python 3.11 or 3.12 (64-bit) is required.')
    pip = [sys.executable, '-m', 'pip', 'install']
    run(pip + ['--upgrade', 'pip', 'wheel', 'setuptools'])
    run(pip + ['torch==2.14.1+cu130', 'torchvision==0.29.1+cu130', 'torchaudio==2.11.0+cu130',
               '--index-url', 'https://download.pytorch.org/whl/cu130'])
    constraints = ROOT / 'venv/torch-constraints.txt'
    constraints.parent.mkdir(exist_ok=True)
    constraints.write_text(constraints_text(), encoding='utf-8')
    os.environ['PIP_CONSTRAINT'] = str(constraints)
    run(pip + ['--no-deps', VOXCPM])
    run(pip + ['-r', 'requirements.txt', *voxcpm_requirements(), 'opencv-python==5.0.0.93',
               *RUNTIME_PINS, LLAMA_WHEEL])
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', *UNUSED_PACKAGES])
    if not check_environment():
        raise RuntimeError('Environment verification failed. See the errors above.')
    config.ensure_dirs()
    print('[OK] The app and its engine are installed.')


TIER_LABELS = {'lite': 'Lite', 'full': 'Full quality'}


def _missing_gb(model_ids) -> float:
    """Free disk space the missing catalog files need at the installation's peak, judged by name and size (hashes
    are checked while installing): their downloads, plus the files install steps make from official sources."""
    from . import model_manager
    return model_manager.missing_bytes(list(model_ids))[1] / 1e9


def choose_tier(answer=None) -> str:
    """'lite' or 'full': the installer argument, else the user's pick, else the default."""
    from . import download_models, model_manager, prefs, vram
    if answer in model_manager.TIERS:
        return answer
    recommended = model_manager.default_tier()
    current = prefs.load().get('tier')
    default = current if current in model_manager.TIERS else recommended
    gpu = vram.get_gpu_info()
    card = f'{gpu.name}, {gpu.total_gb:.0f} GB' if gpu.total_gb else 'not detected'
    size = {t: download_models.size_gb(download_models.model_ids('all', t)) for t in model_manager.TIERS}
    print()
    print('Choose the model set. Both have the same features; models keep their own licenses (MODEL_LICENSES.md).')
    print(f'  1) Lite          for graphics cards with 8-12 GB of VRAM, about {size["lite"]:.0f} GB to download')
    print(f'  2) Full quality  for graphics cards with 16 GB or more, about {size["full"]:.0f} GB to download')
    print(f'Your graphics card: {card}. Recommended: {TIER_LABELS[recommended]}.')
    try:
        reply = input(f'Type 1 or 2 and press Enter, or just press Enter for {TIER_LABELS[default]}: ').strip()
    except EOFError:
        reply = ''
    return {'1': 'lite', '2': 'full'}.get(reply, default)


def install_models(answer=None) -> bool:
    """Download and verify every model of the chosen set. A rerun resumes and skips finished files."""
    if answer == 'skip':
        print('[OK] Model downloads skipped: each feature downloads its models the first time you use it.')
        return True
    from . import download_models, model_manager, prefs
    tier = choose_tier(answer)
    ids = download_models.model_ids('all', tier)
    need, free = _missing_gb(ids), shutil.disk_usage(ROOT).free / 1e9
    if need + 10 > free:  # room for temporary video files
        print(f'[ERROR] The {TIER_LABELS[tier]} models need {need:.0f} GB more, but only {free:.0f} GB is free here.')
        print('Free up space or choose Lite, then run install.bat again.')
        return False
    prefs.save(tier=tier)
    if need:
        print(f'Downloading the {TIER_LABELS[tier]} models (up to {need:.0f} GB of disk space while installing). '
              'Leave this window open.')
        print('If the download stops, run install.bat again: finished files are kept.')
    else:
        print(f'Checking the {TIER_LABELS[tier]} models...')

    def progress(frac: float, desc: str) -> None:
        print(f'\r  {frac * 100:5.1f}%  {desc[:90]:90s}', end='', flush=True)

    error = model_manager.ensure(ids, progress, exact=True)
    print()
    if error:
        print('[ERROR]', error)
        return False
    print(f'[OK] All {TIER_LABELS[tier]} models are installed and verified.')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prepare-venv', action='store_true')
    parser.add_argument('--models', nargs='?', const='', default=None, metavar='lite|full|skip',
                        help='download every model of a set (asks when no set is given)')
    args = parser.parse_args()
    try:
        if args.prepare_venv:
            prepare_venv()
            return 0
        if args.models is not None:
            return 0 if install_models(args.models.lower() or None) else 1
        if args.install:
            install()
            return 0
        return 0 if check_environment() else 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'[ERROR] Installation stopped: {exc}')
        print('No user outputs, models or character data have been removed. Fix the error and run install.bat again.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
