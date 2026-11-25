# SPDX-FileCopyrightText: 2026 Karol Kosciolek
# SPDX-License-Identifier: GPL-3.0-only
"""CPU parity of the engine's EzAudio port against the official code, with the real weights, on small inputs.

    venv\\Scripts\\python.exe -I tools\\parity_ezaudio.py --bundle models\\ezaudio --reference <EzAudio checkout>
        [--seconds 1.5] [--steps 3] [--seed 2024] [--json out.json]

The reference is a code-only checkout of github.com/haidog-yaqub/EzAudio. Its pure-PyTorch modules are imported
(MaskDiT, the Oobleck decoder, src/inference.py); module-level imports it never uses for inference (training losses,
quantizers, data loading) are replaced by empty modules, so nothing is installed. Run it with `python -I`: the
checkout is appended to sys.path, the repository put in front explicitly.

Compared (float32 on both sides unless noted; max abs error, PSNR against the reference's peak, SNR):
1. FLAN-T5-XL: token ids against transformers' T5Tokenizer, hidden states against transformers' T5EncoderModel
   loaded from the same shard (ours in float32, and with float16 storage as the GPU runs it).
2. The diffusion transformer, one forward at t = 999, 500 and 20, plain and inpainting (known frames + mask), with
   guidance's batch of two (ours in float32 and in bfloat16).
3. The sampler and the whole chain: the reference `inference()` (diffusers DDIMScheduler, CPU generator, guidance
   5, rescale 0.75, eta 1) against our window sampler with the same seed: every DDIM step, the final latent and
   the 24 kHz waveform (ours in float32, and bfloat16 transformer + float16 text encoder).
4. The waveform decoder on a random latent and on the latent of step 3, decoded at once and in chunks.

    venv\\Scripts\\python.exe -I tools\\parity_ezaudio.py --make-fixture --reference <EzAudio checkout>

regenerates tests/fixtures/engine_ezaudio.json without any model file: the outputs of the official MaskDiT, the
official Oobleck decoder and the official `inference()` loop on tiny networks with deterministic weights
(`seeded_tensors`), so the default test suite checks the port's math against the reference on every machine.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
import types
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "engine_ezaudio.json"
REFERENCE_COMMIT = "86f18268117dde73bb274401c610e639e4a8fae1"  # github.com/haidog-yaqub/EzAudio
PROMPTS = ["a dog barking in the distance", "light rain on a window with distant thunder",
           "footsteps on a wooden floor, high quality realistic sound effects recording",
           "a busy cafe with clinking cups and quiet chatter", "wind blowing through tall grass", ""]


def _stub_unused_imports() -> None:
    """Empty modules for what the reference imports at module level but does not use to generate audio."""
    class Unused:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("this reference class is not part of the parity run")

    def module(name: str, **attributes) -> types.ModuleType:
        from importlib.machinery import ModuleSpec

        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        stub.__spec__ = ModuleSpec(name, loader=None)  # find_spec() of other libraries must not fail on it
        sys.modules[name] = stub
        return stub

    module("torchaudio", transforms=module("torchaudio.transforms"))
    module("alias_free_torch", Activation1d=Unused)
    module("vector_quantize_pytorch", ResidualVQ=Unused, FSQ=Unused)
    module("audiotools", AudioSignal=Unused, STFTParams=Unused)
    for name in ("pandas", "librosa", "soundfile"):
        module(name)


def _import_reference(reference: Path):
    _stub_unused_imports()
    sys.path.append(str(reference))
    with contextlib.redirect_stdout(io.StringIO()):
        import src
        from src.inference import inference
        from src.models.conditioners import MaskDiT
        from src.modules.stable_vae.models.autoencoders import OobleckDecoder
    if not any(Path(path).resolve().is_relative_to(reference.resolve()) for path in src.__path__):
        raise RuntimeError(f"'src' was imported from {list(src.__path__)}, not from {reference}")
    return MaskDiT, OobleckDecoder, inference


def metrics(got, reference, mask=None) -> dict:
    """Errors of `got` against `reference` (float64): max abs, relative L2, PSNR (reference peak), SNR, cosine."""
    import torch

    got, reference = got.detach().double().cpu(), reference.detach().double().cpu()
    if mask is not None:
        got, reference = got[mask], reference[mask]
    difference = got - reference
    mse = float(difference.pow(2).mean())
    power = float(reference.pow(2).mean())
    peak = float(reference.abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(got.flatten(), reference.flatten(), dim=0))
    return {"max_abs": float(difference.abs().max()), "relative_l2": float(difference.norm() / reference.norm()),
            "psnr_db": math.inf if mse == 0 else 10 * math.log10(peak * peak / mse),
            "snr_db": math.inf if mse == 0 else 10 * math.log10(power / mse), "cosine": cosine, "peak": peak}


def show(results: dict, name: str, values: dict) -> None:
    results[name] = values
    print(f"{name:58s} max|d|={values['max_abs']:.3e}  rel={values['relative_l2']:.2e}  "
          f"PSNR={values['psnr_db']:6.1f} dB  SNR={values['snr_db']:6.1f} dB  cos={values['cosine']:.7f}")


def reference_t5(folder: Path):
    """transformers' T5EncoderModel with the shard's encoder tensors, via load_state_dict (not our loader)."""
    import torch
    from safetensors import safe_open
    from transformers import T5Config, T5EncoderModel

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = T5EncoderModel(T5Config(**config))
    with safe_open(str(folder / "model-00001-of-00002.safetensors"), framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys() if name.startswith(("encoder.", "shared."))}
    report = model.load_state_dict(state, strict=False, assign=True)
    if report.unexpected_keys or set(report.missing_keys) - {"encoder.embed_tokens.weight"}:
        raise RuntimeError(f"reference T5: {report}")
    model.encoder.embed_tokens.weight = model.shared.weight
    return model.eval()


class CachedEncoder:
    """The reference's text encoder, answered from hidden states the real reference T5 computed earlier."""

    def __init__(self, ids, hidden):
        self.rows = [(row, state) for row, state in zip(ids, hidden)]

    def __call__(self, input_ids, attention_mask):
        found = []
        for row in input_ids:
            match = [state for ids, state in self.rows if ids.tolist() == row.tolist()]
            if not match:
                raise RuntimeError("the reference asked for a prompt that was not encoded")
            found.append(match[0])
        import torch

        return types.SimpleNamespace(last_hidden_state=torch.stack(found))


class CapturingDecoder:
    """The reference's autoencoder argument: keeps the latent it is asked to decode."""

    def __init__(self, decoder):
        self.decoder, self.latent = decoder, None

    def __call__(self, embedding):
        self.latent = embedding.clone()
        return self.decoder(embedding)


# ----------------------------- tiny-network fixture (tests/fixtures/engine_ezaudio.json) -----------------------------

# The official MaskDiT with the published options and tiny sizes (tests/test_engine_ezaudio.TINY): 4 latent channels,
# width 32, 2 heads, depth 4 (two blocks per half and the middle block), GEGLU hidden 48, SOLA rank 3 = alpha.
TINY_DIT = dict(mae=True, mae_prob=0.25, mask_ratio=[0.25, 1.0], mask_span=10, img_size=500, patch_size=1, in_chans=9,
                out_chans=4, input_type="1d", embed_dim=32, depth=4, num_heads=2, mlp_ratio=1.5, qkv_bias=False,
                qk_scale=None, qk_norm="layernorm", norm_layer="layernorm", act_layer="geglu", context_norm=True,
                use_checkpoint=False, time_fusion="ada_sola_bias", ada_sola_rank=3, ada_sola_alpha=3, cls_dim=None,
                context_dim=12, context_fusion="cross", context_max_length=None, context_pe_method="none",
                pe_method="none", rope_mode="shared", use_conv=True, skip=True, skip_norm=True)
# The bundle's vae/config.json with 4 instead of 128 base channels (24 kHz, hop 480, 128 latent channels, snake).
TINY_DECODER_CONFIG = {"sample_rate": 24000, "model": {
    "decoder": {"type": "oobleck", "config": {"out_channels": 1, "channels": 4, "c_mults": [1, 2, 4, 8],
                                              "strides": [2, 4, 6, 10], "latent_dim": 128, "use_snake": True,
                                              "final_tanh": False}},
    "latent_dim": 128, "downsampling_ratio": 480, "io_channels": 1}}
TINY_CHAIN = dict(max_tokens=8, latent_rate=50, latent_scale=0.5, latent_shift=0.1, vocab=40, table_seed=21,
                  negative="speech", steps=3)


def seeded_tensors(shapes) -> dict:
    """Deterministic float32 tensors for a {name: shape} map; the seed of each is the CRC-32 of its name.

    The scales make a wrong formula visible: norm scales about 1, SnakeBeta's log alpha and log beta spread by a
    factor of about 1.6 each way (so exchanging them changes the output), weight-norm gains about 1, the AdaLN table
    and the mask token about 0.3, other vectors 0.1, matrices 1 / sqrt(fan-in)."""
    import torch

    tensors = {}
    for name, shape in shapes.items():
        values = torch.randn(tuple(shape), generator=torch.Generator().manual_seed(zlib.crc32(name.encode())))
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("alpha", "beta"):
            values = 0.5 * values
        elif leaf == "weight_g":
            values = 1.0 + 0.25 * values
        elif len(shape) == 1 and leaf == "weight" and "norm" in name:
            values = 1.0 + 0.1 * values
        elif leaf in ("scale_shift_table", "mask_embed"):
            values = 0.3 * values
        elif len(shape) == 1:
            values = 0.1 * values
        else:
            values = values / math.sqrt(math.prod(shape[1:]))
        tensors[name] = values
    return tensors


def seeded_inputs(shape, seed: int):
    import torch

    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed))


def chain_frames(seconds: float, latent_rate: int) -> int:
    """Latent frames of a sound with the timeline's two 0.25 s handles (the pipeline's contract)."""
    return math.ceil(round((seconds + 0.5) * latent_rate, 6))


class CharTokens:
    """Character tokens for the chain fixture: the pipeline calls `encode` (as on SentencePiece), the reference calls
    the object as a Hugging Face T5 tokenizer (cut to max_length - 1, end token 1, padding 0 up to max_length)."""

    @staticmethod
    def encode(prompt: str):
        return [2 + ord(char) % 37 for char in prompt]

    def __call__(self, prompts, max_length, padding, truncation, return_tensors):
        import torch

        ids = torch.zeros((len(prompts), max_length), dtype=torch.long)
        mask = torch.zeros((len(prompts), max_length), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            tokens = self.encode(prompt)[:max_length - 1] + [1]
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
        return types.SimpleNamespace(input_ids=ids, attention_mask=mask)


def table_encoder(vocab: int, width: int, seed: int):
    """A stand-in text encoder for the chain fixture with the interface both sides use (`shared`, and a call with
    input_ids or inputs_embeds plus attention_mask): each token's embedding, scaled by its position and shifted by
    its mask, so that order, padding and the mask all reach the transformer."""
    import torch

    class TableEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Embedding(vocab, width)
            with torch.no_grad():
                self.shared.weight.copy_(seeded_inputs((vocab, width), seed))

        def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
            embeds = self.shared(input_ids) if inputs_embeds is None else inputs_embeds
            position = torch.arange(embeds.shape[1], dtype=embeds.dtype)[None, :, None]
            hidden = embeds * (1 + 0.1 * position) + 0.05 * attention_mask[..., None].to(embeds.dtype)
            return types.SimpleNamespace(last_hidden_state=hidden)

    return TableEncoder().eval()


def _numbers(tensor):
    return [float(f"{value:.9g}") for value in tensor.detach().float().reshape(-1).tolist()]


def _tiny_reference_dit(MaskDiT):
    """The tiny official MaskDiT with seeded weights; its rotary frequencies stay the module's own buffers."""
    with contextlib.redirect_stdout(io.StringIO()):
        model = MaskDiT(**TINY_DIT)
    buffers = {name: tensor.clone() for name, tensor in model.named_buffers()}
    shapes = {name: list(tensor.shape) for name, tensor in model.state_dict().items() if name not in buffers}
    model.load_state_dict({**seeded_tensors(shapes), **buffers}, strict=True)
    return model.eval(), shapes, buffers


def _dit_fixture(model, shapes, buffers) -> dict:
    """Outputs of the official forward at t = 999, 500 and 20, plain and inpainting (known frames + mask)."""
    import torch

    frames, tokens = 23, 7
    data = {"config": TINY_DIT, "shapes": shapes, "buffers": {name: _numbers(t) for name, t in buffers.items()},
            "frames": frames, "tokens": tokens, "latent_seed": 61, "context_seed": 62, "known_seed": 63,
            "context_mask": [[True] * tokens, [True] * 3 + [False] * (tokens - 3)], "generate": [8, 17],
            "outputs": []}
    channels, width = TINY_DIT["out_chans"], TINY_DIT["context_dim"]
    latent = seeded_inputs((2, channels, frames), data["latent_seed"])
    context = seeded_inputs((2, tokens, width), data["context_seed"])
    known = seeded_inputs((2, channels, frames), data["known_seed"])
    keys = torch.tensor(data["context_mask"])
    generate = torch.zeros((2, channels, frames), dtype=torch.bool)
    generate[..., data["generate"][0]:data["generate"][1]] = True
    for timestep in (999, 500, 20):
        plain, _ = model(latent.clone(), torch.tensor(timestep), context, context_mask=keys)
        filled, _ = model(latent.clone(), torch.tensor(timestep), context, context_mask=keys, gt=known.clone(),
                          mae_mask_infer=generate)
        data["outputs"].append({"timestep": timestep, "plain": _numbers(plain), "inpainting": _numbers(filled)})
    return data


def _decoder_fixture(OobleckDecoder) -> dict:
    """The official decoder's waveform for a seeded latent, with seeded weight-normalized convolutions."""
    decoder = OobleckDecoder(**TINY_DECODER_CONFIG["model"]["decoder"]["config"])
    shapes = {name: list(tensor.shape) for name, tensor in decoder.state_dict().items()}
    decoder.load_state_dict(seeded_tensors(shapes), strict=True)
    decoder.eval()
    data = {"config_json": TINY_DECODER_CONFIG, "shapes": shapes, "frames": 6, "latent_seed": 64}
    waveform = decoder(seeded_inputs((1, 128, data["frames"]), data["latent_seed"]))[0]
    data["waveform"], data["peak"] = _numbers(waveform), float(waveform.abs().max())
    return data


def _chain_fixture(model, inference) -> dict:
    """The latent the official `inference()` hands to its autoencoder (after scale/shift), for two sounds and two
    guidance settings, with the tiny MaskDiT, `CharTokens`, `table_encoder` and diffusers' DDIMScheduler."""
    from diffusers import DDIMScheduler

    chain = dict(TINY_CHAIN)
    params = {"text_encoder": {"max_length": chain["max_tokens"]}, "model": {"out_chans": TINY_DIT["out_chans"]},
              "autoencoder": {"scale": chain["latent_scale"], "shift": chain["latent_shift"]}}
    encoder = table_encoder(chain["vocab"], TINY_DIT["context_dim"], chain["table_seed"])
    sounds = [["rain on a roof", 0.1, 5], ["a door", 0.2, 77]]
    # cfg 1 is the engine's "conditional pass only"; the reference spells it guidance 0.
    runs = [{"cfg": 5.0, "reference_guidance": 5.0, "rescale": 0.75, "eta": 1.0},
            {"cfg": 1.0, "reference_guidance": 0.0, "rescale": 0.0, "eta": 0.5}]
    for run in runs:
        run["latents"], run["frames"] = [], []
        for prompt, seconds, seed in sounds:
            frames = chain_frames(seconds, chain["latent_rate"])
            scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085,
                                      beta_end=0.012, prediction_type="v_prediction", rescale_betas_zero_snr=True,
                                      timestep_spacing="trailing", clip_sample=False)
            latent = inference(lambda embedding: embedding.clone(), model, None, None, CharTokens(), encoder, params,
                               scheduler, [prompt], [chain["negative"]], frames,
                               guidance_scale=run["reference_guidance"], guidance_rescale=run["rescale"],
                               ddim_steps=chain["steps"], eta=run["eta"], random_seed=seed, device="cpu")
            run["frames"].append(frames)
            run["latents"].append(_numbers(latent))
    return {**chain, "sounds": sounds, "runs": runs}


def fixture_data(reference: Path) -> dict:
    """Everything tests/fixtures/engine_ezaudio.json holds, computed by the official code in `reference`."""
    import torch

    MaskDiT, OobleckDecoder, inference = _import_reference(reference)
    with torch.no_grad():
        model, shapes, buffers = _tiny_reference_dit(MaskDiT)
        return {"description": "Outputs of the official EzAudio code on tiny networks with the deterministic weights "
                               "of tools/parity_ezaudio.py seeded_tensors (tools/parity_ezaudio.py --make-fixture).",
                "reference": f"github.com/haidog-yaqub/EzAudio@{REFERENCE_COMMIT}", "torch": torch.__version__,
                "transformer": _dit_fixture(model, shapes, buffers), "decoder": _decoder_fixture(OobleckDecoder),
                "chain": _chain_fixture(model, inference)}


def make_fixture(reference: Path, target: Path = FIXTURE) -> int:
    data = fixture_data(reference)
    target.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size // 1024} KiB)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, help="models/ezaudio (official file layout)")
    parser.add_argument("--reference", type=Path, required=True, help="checkout of haidog-yaqub/EzAudio")
    parser.add_argument("--make-fixture", action="store_true",
                        help="regenerate tests/fixtures/engine_ezaudio.json (tiny networks, no model files)")
    parser.add_argument("--fixture-path", type=Path, default=FIXTURE, help="where --make-fixture writes")
    parser.add_argument("--seconds", type=float, default=1.5, help="length of the sampled audio")
    parser.add_argument("--forward-frames", type=int, default=100, help="latent frames of the forward check")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--json", type=Path, help="write the numbers here")
    args = parser.parse_args(argv)
    if args.make_fixture:
        return make_fixture(args.reference, args.fixture_path)
    if args.bundle is None:
        parser.error("--bundle is required for the parity run")

    sys.path.insert(0, str(ROOT))
    import torch
    from diffusers import DDIMScheduler
    from transformers import T5Config, T5EncoderModel, T5Tokenizer  # noqa: F401 - imported before the stubs exist

    from engine.models.ezaudio import config, text, transformer, vae
    from engine.pipelines import ezaudio
    from engine.sampling import ddim

    MaskDiT, OobleckDecoder, inference = _import_reference(args.reference)
    torch.set_grad_enabled(False)
    bundle, results, timings = args.bundle, {}, {}
    params = config.read_yaml(bundle / "ezaudio-xl.yml")
    settings = config.load(bundle / "ezaudio-xl.yml")
    folder = bundle / "text_encoder"

    # 1. Text encoder.
    started = time.perf_counter()
    tokenizer = T5Tokenizer.from_pretrained(str(folder), local_files_only=True)
    batch = tokenizer(PROMPTS, max_length=settings.max_tokens, padding="max_length", truncation=True,
                      return_tensors="pt")
    ids, mask = text.tokenize(text.load_tokenizer(folder), [text.clean_prompt(p) for p in PROMPTS], settings.max_tokens)
    if ids.tolist() != batch.input_ids.tolist() or mask.tolist() != batch.attention_mask.bool().tolist():
        raise RuntimeError("token ids differ from transformers' T5Tokenizer")
    print(f"token ids: identical to transformers' T5Tokenizer for {len(PROMPTS)} prompts")
    results["token_ids_identical"] = True
    reference_encoder = reference_t5(folder)
    reference_hidden = reference_encoder(input_ids=batch.input_ids,
                                         attention_mask=batch.attention_mask.bool()).last_hidden_state.float()
    del reference_encoder
    hidden = {}
    for dtype in (torch.float32, torch.bfloat16):
        encoder = text.load_encoder(folder, "cpu", dtype)
        hidden[dtype] = text.encode(encoder, ids, mask)
        del encoder
        storage = "float32" if dtype is torch.float32 else "float16 storage"
        show(results, f"T5 hidden states, real tokens ({storage})", metrics(hidden[dtype], reference_hidden, mask))
    timings["text"] = time.perf_counter() - started

    # 2. Diffusion transformer, one forward.
    started = time.perf_counter()
    state = torch.load(bundle / "ezaudio_s3_xl.pt", map_location="cpu", weights_only=True)["model"]
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
        reference_dit = MaskDiT(**params["model"])
    reference_dit.load_state_dict(state, strict=True, assign=True)
    reference_dit.eval()
    del state
    ours = {torch.float32: transformer.load(bundle / "ezaudio_s3_xl.pt", "cpu", torch.float32)}
    frames = args.forward_frames
    generator = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, 128, frames, generator=generator).expand(2, -1, -1)
    known = torch.randn(1, 128, frames, generator=generator)
    generate = torch.zeros(1, 128, frames, dtype=torch.bool)
    generate[..., frames // 3: 2 * frames // 3] = True
    pair = torch.stack((hidden[torch.float32][0], hidden[torch.float32][-1]))
    pair_mask = torch.stack((mask[0], mask[-1]))
    expected = {}
    for timestep in (999, 500, 20):
        plain, _ = reference_dit(latent.clone(), torch.tensor(timestep), pair, context_mask=pair_mask)
        filled, _ = reference_dit(latent.clone(), torch.tensor(timestep), pair, context_mask=pair_mask,
                                  gt=known.expand(2, -1, -1).clone(), mae_mask_infer=generate.expand(2, -1, -1))
        expected[timestep] = plain, filled
    ours[torch.bfloat16] = transformer.load(bundle / "ezaudio_s3_xl.pt", "cpu", torch.bfloat16)
    for dtype, network in ours.items():
        context, rotary = network.prepare_context(pair), network.prepare_rotary(frames)
        for timestep, (plain, filled) in expected.items():
            got_plain = network(latent, torch.tensor(timestep), context, pair_mask, rotary)
            got_filled = network(latent, torch.tensor(timestep), context, pair_mask, rotary, known, generate[:, :1])
            name = "float32" if dtype is torch.float32 else "bfloat16"
            show(results, f"DiT forward t={timestep} ({name})", metrics(got_plain, plain))
            show(results, f"DiT forward t={timestep}, inpainting ({name})", metrics(got_filled, filled))
    timings["transformer"] = time.perf_counter() - started

    # 3. Sampler and the whole chain against src/inference.py.
    started = time.perf_counter()
    decoder_config = json.loads((bundle / "vae" / "config.json").read_text(encoding="utf-8"))
    reference_decoder = OobleckDecoder(**decoder_config["model"]["decoder"]["config"])
    vae_state = torch.load(bundle / "vae" / "1m.pt", map_location="cpu", weights_only=True)["state_dict"]
    reference_decoder.load_state_dict({name.removeprefix("autoencoder.decoder."): tensor
                                       for name, tensor in vae_state.items()
                                       if name.startswith("autoencoder.decoder.")}, strict=True)
    reference_decoder.eval()
    del vae_state
    ours_decoder = vae.load(bundle / "vae" / "1m.pt", "cpu")
    sample_frames = math.ceil(round(args.seconds * settings.latent_rate, 6))
    scheduler = DDIMScheduler(**params["diff"])
    reference_steps = []
    original_step = scheduler.step

    def recorded_step(*step_args, **step_kwargs):
        output = original_step(*step_args, **step_kwargs)
        reference_steps.append(output.prev_sample.clone())
        return output

    scheduler.step = recorded_step
    capture = CapturingDecoder(reference_decoder)
    selected = [0, len(PROMPTS) - 1]
    reference_wave = inference(capture, reference_dit, None, None, tokenizer,
                               CachedEncoder(batch.input_ids[selected], reference_hidden[selected]), params, scheduler,
                               [PROMPTS[0]], None, sample_frames, guidance_scale=5.0, guidance_rescale=0.75,
                               ddim_steps=args.steps, eta=1.0, random_seed=args.seed, device="cpu")
    del reference_dit
    schedule = ddim.scaled_linear(settings.train_timesteps, settings.beta_start, settings.beta_end, True)
    timesteps = ddim.trailing_timesteps(settings.train_timesteps, args.steps)
    for dtype, network in ours.items():
        name = "float32" if dtype is torch.float32 else "bfloat16 DiT + float16 T5"
        our_steps = []
        original = ddim.step

        def recorded(*step_args, **step_kwargs):
            our_steps.append(original(*step_args, **step_kwargs))
            return our_steps[-1]

        context = network.prepare_context(torch.stack((hidden[dtype][0], hidden[dtype][-1])))
        ddim.step = recorded
        try:
            latent_out = ezaudio._sample_window(network, schedule, timesteps, args.steps, context, pair_mask, None,
                                                sample_frames, 5.0, 0.75, 1.0,
                                                torch.Generator(device="cpu").manual_seed(args.seed),
                                                torch.device("cpu"), lambda: None)
        finally:
            ddim.step = original
        for number, (got, want) in enumerate(zip(our_steps, reference_steps), 1):
            show(results, f"DDIM step {number}/{args.steps} latent ({name})", metrics(got, want))
        show(results, f"final latent ({name})", metrics(latent_out, capture.latent))
        wave = vae.decode(ours_decoder, latent_out, lambda: None)
        show(results, f"24 kHz waveform, whole chain ({name})", metrics(wave, reference_wave[0]))
    timings["sampler"] = time.perf_counter() - started

    # 4. Waveform decoder.
    started = time.perf_counter()
    random_latent = torch.randn(1, 128, 300, generator=torch.Generator().manual_seed(args.seed + 1))
    for name, latent_in in (("random latent, 6 s", random_latent), ("sampled latent", capture.latent)):
        want = reference_decoder(latent_in)[0]
        show(results, f"decoder, {name}, at once", metrics(ours_decoder(latent_in)[0], want))
        show(results, f"decoder, {name}, in 200-frame chunks", metrics(vae.decode(ours_decoder, latent_in,
                                                                                    lambda: None), want))
    timings["decoder"] = time.perf_counter() - started
    print("CPU seconds per part: " + ", ".join(f"{key} {value:.1f}" for key, value in timings.items()))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"results": results, "seconds": timings, "settings": vars(args)},
                                        indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
