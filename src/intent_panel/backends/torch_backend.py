"""Offline generation and activation readouts for Qwen, Llama and Gemma."""

from dataclasses import asdict
import gc
import os
from pathlib import Path
import time

import numpy as np

from .base import ModelBackend, PrefillState
from ..config import SamplingConfig, resolve_path
from ..contracts import inspect_model
from ..errors import ConfigurationError
from ..frames import reconstruction_prompt, token_frame, validate_messages
from ..serialization import read_json, require


class TransformersBackend(ModelBackend):
    def __init__(self, spec, runtime, config_path):
        self.spec, self.runtime, self.config_path = spec, runtime, Path(config_path).resolve()
        self.path, self.raw_config, self.shape = inspect_model(spec, config_path)
        if self.shape.model_type not in ("qwen2", "llama", "gemma3", "gemma3_text"):
            raise ConfigurationError(
                "Transformers hook layout is not implemented for " + self.shape.model_type
            )
        if self.shape.quantization is not None:
            raise ConfigurationError(
                "This Transformers adapter requires an unquantized safetensors checkpoint"
            )
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
            if os.environ.get(name) != "1":
                raise ConfigurationError("Offline inference requires " + name + "=1")
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration,
        )

        self.torch = torch
        if spec.device != "cpu" and not spec.device.startswith("cuda"):
            raise ConfigurationError("Transformers device must be cpu or one cuda device")
        self.device = torch.device(spec.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ConfigurationError("CUDA is not available locally")
            total = torch.cuda.get_device_properties(self.device).total_memory
            fraction = min(1.0, runtime.memory_limit_gib * 1024**3 / total)
            torch.cuda.set_per_process_memory_fraction(fraction, self.device)
        raw = dict(self.raw_config)
        if "rope_theta" not in raw and "rope_parameters" in raw:
            if raw["rope_parameters"].get("rope_type") != "default":
                raise ConfigurationError("Unsupported RoPE export")
            raw["rope_theta"] = raw["rope_parameters"]["rope_theta"]
        family = raw.pop("model_type")
        config = AutoConfig.for_model(family, **raw)
        parent = {
            "qwen2": Qwen2ForCausalLM,
            "llama": LlamaForCausalLM,
            "gemma3_text": Gemma3ForCausalLM,
            "gemma3": Gemma3ForConditionalGeneration,
        }[family]

        class ConfiguredModel(parent):
            def __init__(inner_self, configuration):
                super().__init__(configuration)
                decoder = (
                    inner_self.model.language_model if family == "gemma3" else inner_self.model
                )
                if spec.final_norm == "identity":
                    decoder.norm = torch.nn.Identity()
                if spec.language_model_head == "absent":
                    inner_self.lm_head = None

        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            spec.dtype
        ]
        self.model, loading = ConfiguredModel.from_pretrained(
            str(self.path),
            config=config,
            dtype=dtype,
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation=spec.attention,
            output_loading_info=True,
        )
        require(
            not loading.get("missing_keys")
            and not loading.get("unexpected_keys")
            and not loading.get("mismatched_keys"),
            "Checkpoint weights do not match the declared decoder/normalization layout",
        )
        self.model.to(self.device)
        self.model.eval()
        self.decoder = self.model.model.language_model if family == "gemma3" else self.model.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path),
            local_files_only=True,
            token=False,
            trust_remote_code=False,
            **spec.tokenizer_options,
        )
        generation = self.path / "generation_config.json"
        generated = read_json(generation) if generation.exists() else {}
        eos = spec.stop_token_ids
        if eos is None:
            eos = generated.get(
                "eos_token_id", self.raw_config.get("eos_token_id", self.tokenizer.eos_token_id)
            )
        self.eos = set(eos if isinstance(eos, list) else [eos])
        require(
            bool(self.eos)
            and all(type(i) is int and 0 <= i < self.shape.vocabulary for i in self.eos),
            "Invalid checkpoint EOS IDs",
        )
        self.forward_calls = 0

    def encode_messages(self, messages, add_generation_prompt=True):
        validate_messages(messages)
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **self.spec.chat_template_options,
            )
        )

    def _ids(self, ids):
        return self.torch.tensor([ids], dtype=self.torch.long, device=self.device)

    def _forward(self, ids=None, cache=None, embeddings=None):
        if self.spec.language_model_head == "absent":
            raise ConfigurationError(
                "A headless reconstructor cannot generate language-model logits"
            )
        kwargs = dict(use_cache=True, return_dict=True, logits_to_keep=1)
        if ids is not None:
            kwargs["input_ids"] = self._ids(ids)
        if embeddings is not None:
            kwargs["inputs_embeds"] = embeddings
        if cache is not None:
            kwargs["past_key_values"] = cache
        with self.torch.inference_mode():
            result = self.model(**kwargs)
        self.forward_calls += 1
        return result.logits[:, -1, :], result.past_key_values

    def prefill(self, frame, state_spec, delta=None):
        torch = self.torch
        records = {}
        positions = list(frame.positions)
        require(1 <= state_spec.layer <= self.shape.layers, "Capture layer is outside the decoder")
        if delta is not None:
            delta = np.asarray(delta, dtype=np.float32)
            require(
                delta.shape == (self.shape.hidden_size,) and np.isfinite(delta).all(),
                "Invalid private-state delta",
            )

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            records["before"] = hidden[0, positions, :].detach().float().cpu().numpy().copy()
            if delta is not None:
                active = torch.zeros((hidden.shape[1],), dtype=hidden.dtype, device=hidden.device)
                active[positions] = 1
                active = active[None, :, None]
                changed = (
                    hidden
                    + active
                    * torch.as_tensor(delta, device=hidden.device, dtype=hidden.dtype)[
                        None, None, :
                    ]
                )
                records["outside"] = float(((changed - hidden) * (1 - active)).abs().max().item())
            else:
                changed = hidden
                records["outside"] = 0.0
            records["after"] = changed[0, positions, :].detach().float().cpu().numpy().copy()
            return (changed,) + output[1:] if isinstance(output, tuple) else changed

        handle = self.decoder.layers[state_spec.layer - 1].register_forward_hook(hook)
        try:
            logits, cache = self._forward(list(frame.base_ids))
        finally:
            handle.remove()
        require(
            cache.get_seq_length() == len(frame.base_ids) and records["outside"] == 0,
            "Private prefill scope/cache differs",
        )

        def pool(x):
            return (
                x.mean(axis=0, dtype=np.float32) if state_spec.pooling == "mean" else x[-1].copy()
            )

        before, after = records["before"], records["after"]
        metadata = dict(
            frame.metadata(),
            layer=state_spec.layer,
            hidden_size=self.shape.hidden_size,
            hook_point=state_spec.hook_point,
            pooling=state_spec.pooling,
            outside_token_error=records["outside"],
            intervention_during_question=False,
            intervention_during_generation=False,
            dtype=self.spec.dtype,
        )
        del logits
        return PrefillState(cache, pool(before), pool(after), metadata, before, after)

    def _sample(self, logits, sampling, generator):
        torch = self.torch
        if sampling.temperature == 0:
            return int(torch.argmax(logits).item())
        probabilities = torch.softmax(logits.float() / sampling.temperature, dim=-1)
        if sampling.top_p < 1:
            sorted_probabilities, indices = torch.sort(probabilities, descending=True)
            excluded = (
                torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities >= sampling.top_p
            )
            sorted_probabilities = sorted_probabilities.masked_fill(excluded, 0)
            sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()
            chosen = torch.multinomial(sorted_probabilities, 1, generator=generator)
            return int(indices[chosen].item())
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    def _decode(self, logits, cache, sampling, seed, prompt_tokens):
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        tokens = []
        finish = "length"
        for index in range(sampling.max_tokens):
            token = self._sample(logits[0], sampling, generator)
            if token in self.eos:
                finish = "stop"
                break
            tokens.append(token)
            if index + 1 < sampling.max_tokens:
                logits, cache = self._forward([token], cache)
        return dict(
            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
            token_ids=tokens,
            generation_tokens=len(tokens),
            prompt_tokens=prompt_tokens,
            finish_reason=finish,
            sampling=asdict(sampling),
            seed=seed,
            stop_token_ids=sorted(self.eos),
        )

    def _generate_ids(self, ids, sampling, seed, embeddings=None):
        if len(ids) + sampling.max_tokens > self.runtime.maximum_context_tokens:
            raise ConfigurationError("Generation exceeds context budget")
        started = time.monotonic()
        logits, cache = self._forward(ids if embeddings is None else None, embeddings=embeddings)
        response = self._decode(logits, cache, sampling, seed, len(ids))
        response["elapsed_seconds"] = time.monotonic() - started
        return response

    def generate(self, messages, sampling, seed):
        return self._generate_ids(self.encode_messages(messages), sampling, seed)

    def branch(self, messages, question, state_spec, sampling, seed, delta=None):
        started = time.monotonic()
        frame = token_frame(
            self.tokenizer,
            messages,
            state_spec,
            question,
            self.runtime.maximum_context_tokens - sampling.max_tokens,
            self.spec.chat_template_options,
        )
        state = self.prefill(frame, state_spec, delta)
        logits, cache = self._forward(list(frame.suffix_ids), state.cache)
        response = self._decode(logits, cache, sampling, seed, len(frame.full_ids))
        response["elapsed_seconds"] = time.monotonic() - started
        state.cache = None
        return response, state

    def verbalize(self, activation, injection, sampling, seed):
        torch = self.torch
        vector = np.asarray(activation, dtype=np.float32)
        require(
            vector.shape == (self.shape.hidden_size,) and np.isfinite(vector).all(),
            "AV activation has wrong shape/finiteness",
        )
        norm = float(np.linalg.norm(vector))
        require(norm > 0, "Zero AV input")
        ids = list(
            self.tokenizer.encode(
                injection.prompt.format(injection_char=injection.token), add_special_tokens=False
            )
        )
        require(
            self.tokenizer.encode(injection.token, add_special_tokens=False)
            == [injection.token_id],
            "Injection token differs",
        )
        slots = [i for i, t in enumerate(ids) if t == injection.token_id]
        require(len(slots) == 1, "AV injection slot is not unique")
        slot = slots[0]
        for offset, expected in [
            (-1, injection.left_neighbor_id),
            (1, injection.right_neighbor_id),
        ]:
            if expected is not None:
                require(
                    0 <= slot + offset < len(ids) and ids[slot + offset] == expected,
                    "AV neighbor differs",
                )
        with torch.inference_mode():
            # Call the native embedding MODULE. Gemma3's module already applies
            # sqrt(hidden_size); multiplying again would corrupt every context
            # embedding. The injected vector is then set at its trained norm.
            embedding = self.model.get_input_embeddings()(self._ids(ids))
            value = (vector / norm if injection.normalize else vector) * injection.scale
            embedding[:, slot, :] = torch.as_tensor(
                value, dtype=embedding.dtype, device=self.device
            )[None, :]
            response = self._generate_ids(ids, sampling, seed, embedding)
        response.update(
            injection_slot=slot,
            input_norm=norm,
            input_normalized=injection.normalize,
            injection_scale=injection.scale,
            source_text_supplied=False,
        )
        return response

    def reconstruct(self, text, reconstruction):
        torch = self.torch
        ids = self.tokenizer.encode(
            reconstruction_prompt(text, reconstruction),
            add_special_tokens=reconstruction.add_special_tokens,
        )
        if not ids or len(ids) > self.runtime.maximum_context_tokens:
            raise ConfigurationError("Invalid AR input length")
        captured = {}
        handle = None
        if reconstruction.readout == "block_output":

            def capture(module, args, output):
                captured["hidden"] = (
                    (output[0] if isinstance(output, tuple) else output)[0, -1, :].detach().float()
                )

            handle = self.decoder.layers[reconstruction.layer - 1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                output = self.decoder(self._ids(ids), use_cache=False, return_dict=True)
                hidden = (
                    output.last_hidden_state[0, -1, :].float()
                    if reconstruction.readout == "final_hidden"
                    else captured["hidden"]
                )
                from safetensors.torch import load_file

                head = load_file(
                    str(resolve_path(reconstruction.head_path, self.config_path)), device="cpu"
                )
                weight = head[reconstruction.head_weight_key].to(
                    device=self.device, dtype=torch.float32
                )
                prediction = hidden @ (
                    weight.T if reconstruction.head_orientation == "output_input" else weight
                )
                if reconstruction.head_bias_key is not None:
                    prediction = prediction + head[reconstruction.head_bias_key].to(
                        self.device, dtype=torch.float32
                    )
                vector = prediction.detach().float().cpu().numpy().copy()
        finally:
            if handle is not None:
                handle.remove()
        norm = float(np.linalg.norm(vector))
        require(np.isfinite(vector).all() and norm > 0, "AR produced an invalid vector")
        return (vector / norm if reconstruction.normalize_output else vector), dict(
            raw_norm=norm,
            output_normalized=reconstruction.normalize_output,
            readout=reconstruction.readout,
            final_norm=self.spec.final_norm,
        )

    def _cache_error(self, left, right, indices=None):
        values = []
        for i in range(self.shape.layers) if indices is None else indices:
            for a, b in zip(left[i], right[i]):
                require(a.shape == b.shape, "Compared cache shapes differ")
                values.append(float((a - b).abs().max().item()))
        return max(values, default=0.0)

    def technical_check(self, state_spec):
        rows = []
        sampling = SamplingConfig(max_tokens=4)
        for text, question in [
            ("Remember the color blue.", "Which color was named?"),
            ("A box contains three wooden blocks.", "How many blocks are in the box?"),
        ]:
            messages = [
                dict(role="system", content="Answer briefly."),
                dict(role="user", content=text),
            ]
            frame = token_frame(
                self.tokenizer,
                messages,
                state_spec,
                question,
                self.runtime.maximum_context_tokens - 4,
                self.spec.chat_template_options,
            )
            natural = self.prefill(frame, state_spec)
            zero = self.prefill(
                frame, state_spec, np.zeros(self.shape.hidden_size, dtype=np.float32)
            )
            perturbed = self.prefill(
                frame, state_spec, np.ones(self.shape.hidden_size, dtype=np.float32) * 0.01
            )
            zero_error = self._cache_error(natural.cache, zero.cache)
            lower = self._cache_error(natural.cache, perturbed.cache, range(state_spec.layer))
            upper = self._cache_error(
                natural.cache, perturbed.cache, range(state_spec.layer, self.shape.layers)
            )
            logits, native = self._forward(list(frame.base_ids))
            base_error = self._cache_error(natural.cache, native)
            a, ca = self._forward(list(frame.suffix_ids), natural.cache)
            b, cb = self._forward(list(frame.suffix_ids), native)
            maximum = max(
                zero_error,
                lower,
                base_error,
                float((a - b).abs().max().item()),
                self._cache_error(ca, cb),
            )
            ga = self._decode(a, ca, sampling, 42, len(frame.full_ids))
            gb = self._decode(b, cb, sampling, 42, len(frame.full_ids))
            require(
                maximum <= self.runtime.native_absolute_tolerance
                and ga["token_ids"] == gb["token_ids"],
                "Native/private-cache correspondence failed",
            )
            if state_spec.layer < self.shape.layers:
                require(upper > 0, "No downstream cache effect from perturbation")
            rows.append(
                dict(
                    maximum_native_error=maximum,
                    zero_error=zero_error,
                    lower_cache_error=lower,
                    upper_cache_difference=upper,
                    native_token_ids=gb["token_ids"],
                )
            )
            del natural, zero, perturbed, native, ca, cb, logits, a, b
        return dict(
            status="passed_neutral_backend_checks",
            backend="transformers",
            shape=asdict(self.shape),
            state=asdict(state_spec),
            absolute_tolerance=self.runtime.native_absolute_tolerance,
            checks=rows,
            semantic_validity_established=False,
        )

    def close(self):
        self.decoder = None
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
