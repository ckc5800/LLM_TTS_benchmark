from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc

LOGGER = logging.getLogger("tts_benchmark.grpc_bridge")


def _import_proto_modules() -> tuple[Any, Any, Path]:
    script = Path(__file__).resolve()
    root = script.parents[2]
    env_path = os.getenv("AICESS_TTS_PROTO_PY_DIR", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            root / "aicess-tts-integrated" / "admin" / "backend" / "app" / "grpc_client",
            root / "aicess-tts-integrated" / "api",
        ]
    )
    for base in candidates:
        if not (base / "proto" / "tts_pb2.py").exists():
            continue
        if not (base / "proto" / "tts_pb2_grpc.py").exists():
            continue
        for name in ("proto.tts_pb2_grpc", "proto.tts_pb2", "proto"):
            if name in sys.modules:
                del sys.modules[name]
        sys.path.insert(0, str(base))
        try:
            from proto import tts_pb2 as pb2  # type: ignore
            from proto import tts_pb2_grpc as pb2_grpc  # type: ignore
            return pb2, pb2_grpc, base
        except Exception:  # noqa: BLE001
            sys.path.pop(0)
    raise RuntimeError(
        "Could not import proto modules. Set AICESS_TTS_PROTO_PY_DIR to a directory "
        "that has proto/tts_pb2.py and proto/tts_pb2_grpc.py."
    )


tts_pb2, tts_pb2_grpc, PROTO_BASE = _import_proto_modules()


@dataclass
class ModelConfig:
    model_id: str
    voice_id: str
    name: str
    model_type: str
    run_model_key: str
    python_exe: str
    model_dir: str
    language: str
    gender: str
    sample_rate: int
    description: str
    enabled: bool
    timeout_s: int


class BenchmarkBridgeServicer(tts_pb2_grpc.TTSServiceServicer):
    def __init__(
        self,
        *,
        bench_root: Path,
        run_model_script: Path,
        output_dir: Path,
        model_instance: str,
        max_concurrent: int,
        max_text_length: int,
        keep_outputs: bool,
        default_timeout_s: int,
        models: list[ModelConfig],
        default_voice_id: str,
    ) -> None:
        self.bench_root = bench_root
        self.run_model_script = run_model_script
        self.output_dir = output_dir
        self.model_instance = model_instance
        self.max_concurrent = max(1, max_concurrent)
        self.max_text_length = max(1, max_text_length)
        self.keep_outputs = keep_outputs
        self.default_timeout_s = max(1, default_timeout_s)
        self.start_time = time.time()

        self.models_by_voice = {m.voice_id: m for m in models}
        self.models_by_id = {m.model_id: m for m in models}
        self.default_voice_id = default_voice_id if default_voice_id in self.models_by_voice else (
            models[0].voice_id if models else ""
        )

        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        self.active_requests = 0
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.total_ms = 0.0
        self.min_ms = 0.0
        self.max_ms = 0.0

        self.drain_enabled = False
        self.drain_started_at_ms = 0
        self.drain_start_success_count = 0
        self.drain_rejected_count = 0

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _rid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _has_field(msg: Any, field: str) -> bool:
        try:
            return msg.HasField(field)
        except Exception:  # noqa: BLE001
            return False

    def _err(self, code: int, message: str, request_id: str) -> Any:
        return tts_pb2.ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            timestamp_ms=self._now_ms(),
        )

    def _resolve_model(self, voice_id: str) -> ModelConfig | None:
        key = (voice_id or "").strip()
        if not key:
            return self.models_by_voice.get(self.default_voice_id)
        return self.models_by_voice.get(key)

    def _resolve_model_instance(self, model_instance: str) -> ModelConfig | None:
        key = (model_instance or "").strip()
        if not key or key == self.model_instance:
            return self.models_by_voice.get(self.default_voice_id)
        if key in self.models_by_id:
            return self.models_by_id[key]
        return self._resolve_model(key)

    @staticmethod
    def _is_remote_model_ref(model_dir: str) -> bool:
        raw = (model_dir or "").strip()
        if not raw:
            return False
        # Windows/Unix absolute path should stay strict local-path validation.
        if Path(raw).is_absolute() or raw.startswith(("\\\\", "/")):
            return False
        # HF style model id (owner/model) is allowed even if not a local dir.
        return "/" in raw and "\\" not in raw

    def _check_model_dir(self, model_dir: str) -> tuple[bool, str]:
        if Path(model_dir).exists():
            return True, "ready"
        if self._is_remote_model_ref(model_dir):
            return True, "remote-model-ref"
        return False, f"model dir not found: {model_dir}"

    def _model_ready(self, model: ModelConfig) -> tuple[bool, str]:
        if not model.enabled:
            return False, "disabled"
        if not self.run_model_script.exists():
            return False, f"runner not found: {self.run_model_script}"
        if not Path(model.python_exe).exists():
            return False, f"python not found: {model.python_exe}"
        ok, message = self._check_model_dir(model.model_dir)
        if not ok:
            return False, message
        return True, "ready"

    @staticmethod
    def _stdout_json(stdout_text: str) -> dict[str, Any]:
        last = None
        for line in stdout_text.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                last = line
        if not last:
            raise RuntimeError("runner stdout did not include json payload")
        return json.loads(last)

    def _voice_info(self, model: ModelConfig) -> Any:
        ready, status = self._model_ready(model)
        desc = model.description if ready else f"{model.description} ({status})"
        return tts_pb2.VoiceInfo(
            voice_id=model.voice_id,
            name=model.name,
            language=model.language,
            gender=model.gender,
            sample_rate=model.sample_rate,
            description=desc,
            enabled=model.enabled,
            ready=ready and model.enabled,
            model_instance=self.model_instance,
            supported_formats=[tts_pb2.AUDIO_FORMAT_WAV],
        )

    def _gpu(self) -> Any:
        try:
            import torch  # type: ignore
        except Exception:  # noqa: BLE001
            torch = None  # type: ignore
        if torch is None or not torch.cuda.is_available():
            return tts_pb2.GPUStatus(
                available=False,
                device="cpu",
                device_id=0,
                device_name="cpu",
                total_memory_mb=0,
                used_memory_mb=0,
                free_memory_mb=0,
                utilization_percent=0,
                memory_percent=0,
            )
        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory / (1024 * 1024))
        used = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        free = max(0, total - used)
        return tts_pb2.GPUStatus(
            available=True,
            device="cuda:0",
            device_id=0,
            device_name=str(props.name),
            total_memory_mb=total,
            used_memory_mb=used,
            free_memory_mb=free,
            utilization_percent=0,
            memory_percent=int((used / total) * 100) if total else 0,
        )

    @staticmethod
    def _wav_info(audio_bytes: bytes, fallback_sr: int = 24000) -> tuple[int, float]:
        sr = fallback_sr
        sec = 0.0
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            sec = wf.getnframes() / float(sr) if sr else 0.0
        return sr, sec

    async def _track_start(self) -> None:
        async with self.lock:
            self.active_requests += 1

    async def _track_end(self, ok: bool, elapsed_ms: float) -> None:
        async with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.total_requests += 1
            self.success_requests += 1 if ok else 0
            self.failed_requests += 0 if ok else 1
            self.total_ms += elapsed_ms
            self.min_ms = elapsed_ms if self.min_ms == 0.0 else min(self.min_ms, elapsed_ms)
            self.max_ms = max(self.max_ms, elapsed_ms)

    async def _invoke_runner(self, model: ModelConfig, text: str, request_id: str) -> tuple[bytes, int, float]:
        output_base = self.output_dir / f"{request_id}_{model.model_id}.wav"
        cmd = [
            model.python_exe,
            str(self.run_model_script),
            "--model",
            model.run_model_key,
            "--model-dir",
            model.model_dir,
            "--text",
            text,
            "--output-path",
            str(output_base),
            "--runs",
            "1",
        ]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.bench_root),
            env=env,
        )
        timeout_s = model.timeout_s if model.timeout_s > 0 else self.default_timeout_s
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"runner timeout after {timeout_s}s") from exc

        out = out_b.decode("utf-8", errors="replace")
        err = err_b.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError((err or out)[-1000:])

        payload = self._stdout_json(out)
        if payload.get("status") != "ok":
            raise RuntimeError(str(payload.get("error", "runner returned error")))

        result = (payload.get("results") or [None])[0]
        if result is None:
            raise RuntimeError("runner returned empty results")

        candidates: list[Path] = []
        output_wav = str(result.get("output_wav", "")).strip()
        if output_wav:
            p = Path(output_wav)
            if not p.is_absolute():
                p = (self.bench_root / p).resolve()
            candidates.append(p)
        candidates.append(Path(str(output_base).replace(".wav", "_0.wav")))
        candidates.extend(sorted(self.output_dir.glob(f"{request_id}_{model.model_id}_*.wav")))
        wav_path = next((p for p in candidates if p.exists()), None)
        if wav_path is None:
            raise RuntimeError("runner did not produce output wav")

        audio_data = wav_path.read_bytes()
        if not audio_data:
            raise RuntimeError("output wav is empty")
        sr = int(result.get("sample_rate") or model.sample_rate or 24000)
        dur = float(result.get("audio_duration_s") or 0.0)
        if dur <= 0:
            sr, dur = self._wav_info(audio_data, fallback_sr=sr)
        if not self.keep_outputs:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Failed to cleanup temp wav: %s", wav_path, exc_info=True)
        return audio_data, sr, dur

    async def SynthesizeText(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        start = time.perf_counter()
        await self._track_start()
        try:
            if self.drain_enabled:
                async with self.lock:
                    self.drain_rejected_count += 1
                return tts_pb2.SynthesizeResponse(
                    request_id=rid,
                    success=False,
                    error=self._err(tts_pb2.ERROR_POOL_EXHAUSTED, "drain mode is active", rid),
                    timestamp_ms=self._now_ms(),
                )

            text = (request.text or "").strip()
            if not text:
                return tts_pb2.SynthesizeResponse(
                    request_id=rid,
                    success=False,
                    error=self._err(tts_pb2.ERROR_EMPTY_TEXT, "text is empty", rid),
                    timestamp_ms=self._now_ms(),
                )
            if len(text) > self.max_text_length:
                return tts_pb2.SynthesizeResponse(
                    request_id=rid,
                    success=False,
                    error=self._err(tts_pb2.ERROR_TEXT_TOO_LONG, f"max={self.max_text_length}", rid),
                    timestamp_ms=self._now_ms(),
                )
            if request.format not in (tts_pb2.AUDIO_FORMAT_UNKNOWN, tts_pb2.AUDIO_FORMAT_WAV):
                return tts_pb2.SynthesizeResponse(
                    request_id=rid,
                    success=False,
                    error=self._err(tts_pb2.ERROR_INVALID_FORMAT, "only WAV is supported", rid),
                    timestamp_ms=self._now_ms(),
                )

            model = self._resolve_model(request.voice_id)
            if not model:
                return tts_pb2.SynthesizeResponse(
                    request_id=rid,
                    success=False,
                    error=self._err(tts_pb2.ERROR_VOICE_NOT_FOUND, f"voice not found: {request.voice_id}", rid),
                    timestamp_ms=self._now_ms(),
                )
            ready, msg = self._model_ready(model)
            if not ready:
                return tts_pb2.SynthesizeResponse(
                    request_id=rid,
                    success=False,
                    error=self._err(tts_pb2.ERROR_MODEL_NOT_LOADED, msg, rid),
                    timestamp_ms=self._now_ms(),
                )

            async with self.semaphore:
                audio_data, sr, dur = await self._invoke_runner(model, text, rid)
            elapsed = (time.perf_counter() - start) * 1000
            await self._track_end(True, elapsed)
            return tts_pb2.SynthesizeResponse(
                request_id=rid,
                success=True,
                audio_data=audio_data,
                format=tts_pb2.AUDIO_FORMAT_WAV,
                sample_rate=sr,
                duration_seconds=dur,
                data_size=len(audio_data),
                processing_time_ms=int(elapsed),
                timestamp_ms=self._now_ms(),
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            await self._track_end(False, elapsed)
            return tts_pb2.SynthesizeResponse(
                request_id=rid,
                success=False,
                error=self._err(tts_pb2.ERROR_SYNTHESIS_FAILED, str(exc), rid),
                timestamp_ms=self._now_ms(),
            )

    async def SynthesizeTextStreaming(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        resp = await self.SynthesizeText(request, context)
        if not resp.success:
            code = resp.error.code if resp.HasField("error") else tts_pb2.ERROR_INTERNAL
            msg = resp.error.message if resp.HasField("error") else "synthesis failed"
            yield tts_pb2.AudioChunk(
                request_id=rid,
                chunk_data=b"",
                chunk_index=0,
                is_last=True,
                has_error=True,
                error_code=code,
                error_message=msg,
                timestamp_ms=self._now_ms(),
            )
            return
        chunk_size = 64 * 1024
        for idx, offset in enumerate(range(0, len(resp.audio_data), chunk_size)):
            chunk = resp.audio_data[offset : offset + chunk_size]
            yield tts_pb2.AudioChunk(
                request_id=rid,
                chunk_data=chunk,
                chunk_index=idx,
                is_last=offset + chunk_size >= len(resp.audio_data),
                chunk_size=len(chunk),
                timestamp_ms=self._now_ms(),
            )

    async def GetVoices(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        language = request.language if self._has_field(request, "language") else ""
        gender = request.gender if self._has_field(request, "gender") else ""
        enabled_only = request.enabled_only if self._has_field(request, "enabled_only") else False
        voices = []
        for model in self.models_by_voice.values():
            if enabled_only and not model.enabled:
                continue
            if language and model.language.lower() != language.lower():
                continue
            if gender and model.gender.lower() != gender.lower():
                continue
            voices.append(self._voice_info(model))
        return tts_pb2.GetVoicesResponse(
            request_id=rid,
            voices=voices,
            total_count=len(voices),
            default_voice_id=self.default_voice_id,
            timestamp_ms=self._now_ms(),
        )

    async def GetVoiceInfo(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model(request.voice_id)
        if not model:
            return tts_pb2.GetVoiceInfoResponse(
                request_id=rid,
                found=False,
                error=self._err(tts_pb2.ERROR_VOICE_NOT_FOUND, f"voice not found: {request.voice_id}", rid),
                timestamp_ms=self._now_ms(),
            )
        return tts_pb2.GetVoiceInfoResponse(
            request_id=rid,
            found=True,
            voice=self._voice_info(model),
            timestamp_ms=self._now_ms(),
        )

    async def GetVoiceMemory(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        return tts_pb2.GetVoiceMemoryResponse(
            request_id=rid,
            voice_id=request.voice_id,
            prompt_memory_mb=0.0,
            model_shared_memory_mb=0.0,
            total_memory_mb=0.0,
            memory_breakdown=tts_pb2.MemoryBreakdown(
                wav_tensor_mb=0.0,
                features_mb=0.0,
                tokens_mb=0.0,
                text_mb=0.0,
                other_mb=0.0,
            ),
            timestamp_ms=self._now_ms(),
        )

    async def HealthCheck(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        include_details = bool(request.include_details)
        gpu = self._gpu()
        models = []
        healthy = False
        for model in self.models_by_voice.values():
            ready, msg = self._model_ready(model)
            ok = ready and model.enabled
            healthy = healthy or ok
            status = tts_pb2.HEALTH_STATUS_HEALTHY if ok else tts_pb2.HEALTH_STATUS_UNHEALTHY
            kwargs = {"gpu_status": gpu} if include_details else {}
            models.append(
                tts_pb2.ModelHealthStatus(
                    model_name=model.name,
                    model_type=model.model_type,
                    healthy=ok,
                    status=status,
                    status_message="OK" if ok else msg,
                    checks=tts_pb2.ModelHealthChecks(
                        model_loaded=ready,
                        voices_loaded=True,
                        config_valid=True,
                        gpu_available=gpu.available,
                        pool_healthy=ok,
                    ),
                    loaded_voices=[model.voice_id] if ok else [],
                    pool_total=self.max_concurrent,
                    pool_available=max(0, self.max_concurrent - self.active_requests),
                    pool_in_use=self.active_requests,
                    **kwargs,
                )
            )
        overall = tts_pb2.HEALTH_STATUS_HEALTHY if healthy else tts_pb2.HEALTH_STATUS_UNHEALTHY
        return tts_pb2.HealthCheckResponse(
            request_id=rid,
            healthy=healthy,
            status=overall,
            status_message="OK" if healthy else "no ready model",
            models=models,
            timestamp_ms=self._now_ms(),
            uptime_ms=int((time.time() - self.start_time) * 1000),
        )

    async def GetGpuStatus(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        gpu = self._gpu()
        return tts_pb2.GetGpuStatusResponse(
            request_id=rid,
            gpus=[gpu] if gpu.available else [],
            gpu_count=1 if gpu.available else 0,
            timestamp_ms=self._now_ms(),
        )

    async def GetServerStats(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        uptime = max(1.0, time.time() - self.start_time)
        avg = int(self.total_ms / self.total_requests) if self.total_requests else 0
        return tts_pb2.ServerStatsResponse(
            request_id=rid,
            active_requests=self.active_requests,
            queued_requests=0,
            active_batches=0,
            total_requests=self.total_requests,
            successful_requests=self.success_requests,
            failed_requests=self.failed_requests,
            avg_processing_time_ms=avg,
            min_processing_time_ms=int(self.min_ms) if self.total_requests else 0,
            max_processing_time_ms=int(self.max_ms) if self.total_requests else 0,
            requests_per_second=self.total_requests / uptime,
            uptime_seconds=int(uptime),
            start_time_ms=int(self.start_time * 1000),
            version="tts-benchmark-grpc-0.1.0",
            model_pool_total=self.max_concurrent,
            model_pool_available=max(0, self.max_concurrent - self.active_requests),
            timestamp_ms=self._now_ms(),
        )

    async def WarmupModel(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model_instance(request.model_instance)
        if not model:
            return tts_pb2.WarmupResponse(
                request_id=rid,
                success=False,
                model_instance=request.model_instance,
                message=f"model not found: {request.model_instance}",
                error=self._err(tts_pb2.ERROR_MODEL_NOT_FOUND, f"model not found: {request.model_instance}", rid),
                timestamp_ms=self._now_ms(),
            )
        ready, msg = self._model_ready(model)
        return tts_pb2.WarmupResponse(
            request_id=rid,
            success=ready,
            model_instance=model.model_id,
            message="ready" if ready else msg,
            duration_ms=0,
            error=None if ready else self._err(tts_pb2.ERROR_MODEL_NOT_LOADED, msg, rid),
            timestamp_ms=self._now_ms(),
        )

    async def WarmupAllModels(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        results = []
        ok_all = True
        for model in self.models_by_voice.values():
            ready, msg = self._model_ready(model)
            ok_all = ok_all and ready
            item = tts_pb2.WarmupResult(
                model_instance=model.model_id,
                success=ready,
                message="ready" if ready else msg,
                duration_ms=0,
            )
            if not ready:
                item.error.CopyFrom(self._err(tts_pb2.ERROR_MODEL_NOT_LOADED, msg, rid))
            results.append(item)
        return tts_pb2.WarmupAllResponse(
            request_id=rid,
            overall_success=ok_all,
            message="all ready" if ok_all else "one or more models are not ready",
            results=results,
            total_duration_ms=0,
            timestamp_ms=self._now_ms(),
        )

    async def GetLoadedModels(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        infos = []
        for model in self.models_by_voice.values():
            ready, _ = self._model_ready(model)
            status = "ready" if ready else ("disabled" if not model.enabled else "error")
            infos.append(
                tts_pb2.LoadedModelInfo(
                    model_instance=model.model_id,
                    model_type=model.model_type,
                    model_path=model.model_dir,
                    status=status,
                    loaded_at_ms=int(self.start_time * 1000),
                    memory_usage_mb=0.0,
                    voice_count=1,
                    loaded_voices=[model.voice_id] if ready else [],
                    supports_speed=False,
                    supports_pitch=False,
                    supports_volume=False,
                )
            )
        return tts_pb2.GetLoadedModelsResponse(request_id=rid, models=infos, timestamp_ms=self._now_ms())

    async def ReloadModel(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model_instance(request.model_instance)
        if not model:
            return tts_pb2.ReloadModelResponse(
                request_id=rid,
                success=False,
                message=f"model not found: {request.model_instance}",
                reload_time_ms=0,
                error=self._err(tts_pb2.ERROR_MODEL_NOT_FOUND, f"model not found: {request.model_instance}", rid),
            )
        ready, msg = self._model_ready(model)
        return tts_pb2.ReloadModelResponse(
            request_id=rid,
            success=ready,
            message="stateless runner does not require reload" if ready else msg,
            reload_time_ms=0,
            error=None if ready else self._err(tts_pb2.ERROR_MODEL_NOT_LOADED, msg, rid),
        )

    async def ReloadModelStreaming(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model_instance(request.model_instance)
        model_name = model.model_id if model else request.model_instance
        yield tts_pb2.ReloadProgress(
            request_id=rid,
            model_instance=model_name,
            phase=tts_pb2.RELOAD_PHASE_STARTED,
            attempt_number=1,
            max_attempts=1,
            phase_success=True,
            message="reload started",
            is_final=False,
            success=False,
            rolled_back=False,
            total_time_ms=0,
        )
        if not model:
            yield tts_pb2.ReloadProgress(
                request_id=rid,
                model_instance=model_name,
                phase=tts_pb2.RELOAD_PHASE_FAILED,
                attempt_number=1,
                max_attempts=1,
                phase_success=False,
                message=f"model not found: {request.model_instance}",
                is_final=True,
                success=False,
                rolled_back=False,
                total_time_ms=0,
                error=self._err(tts_pb2.ERROR_MODEL_NOT_FOUND, f"model not found: {request.model_instance}", rid),
            )
            return
        ready, msg = self._model_ready(model)
        yield tts_pb2.ReloadProgress(
            request_id=rid,
            model_instance=model.model_id,
            phase=tts_pb2.RELOAD_PHASE_COMPLETED if ready else tts_pb2.RELOAD_PHASE_FAILED,
            attempt_number=1,
            max_attempts=1,
            phase_success=ready,
            message="reload completed" if ready else msg,
            is_final=True,
            success=ready,
            rolled_back=False,
            total_time_ms=0,
            error=None if ready else self._err(tts_pb2.ERROR_MODEL_NOT_LOADED, msg, rid),
        )

    async def ReloadAllModels(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        results = []
        ok_all = True
        for model in self.models_by_voice.values():
            ready, msg = self._model_ready(model)
            ok_all = ok_all and ready
            results.append(
                tts_pb2.ModelReloadResult(
                    model_instance=model.model_id,
                    success=ready,
                    message="reload skipped (stateless)" if ready else msg,
                    reload_time_ms=0,
                    rolled_back=False,
                )
            )
        return tts_pb2.ReloadAllModelsResponse(
            request_id=rid,
            overall_success=ok_all,
            message="all ready" if ok_all else "one or more models are not ready",
            results=results,
            total_time_ms=0,
        )

    async def ReloadAllModelsStreaming(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        yield tts_pb2.ReloadProgress(
            request_id=rid,
            model_instance=self.model_instance,
            phase=tts_pb2.RELOAD_PHASE_STARTED,
            attempt_number=1,
            max_attempts=1,
            phase_success=True,
            message="reload all started",
            is_final=False,
            success=False,
            rolled_back=False,
            total_time_ms=0,
        )
        ok_all = all(self._model_ready(m)[0] for m in self.models_by_voice.values())
        yield tts_pb2.ReloadProgress(
            request_id=rid,
            model_instance=self.model_instance,
            phase=tts_pb2.RELOAD_PHASE_COMPLETED if ok_all else tts_pb2.RELOAD_PHASE_FAILED,
            attempt_number=1,
            max_attempts=1,
            phase_success=ok_all,
            message="reload all completed" if ok_all else "one or more models are not ready",
            is_final=True,
            success=ok_all,
            rolled_back=False,
            total_time_ms=0,
        )

    async def UpdateVoice(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model(request.voice_id)
        if not model:
            return tts_pb2.UpdateVoiceResponse(
                request_id=rid,
                success=False,
                message=f"voice not found: {request.voice_id}",
                error=self._err(tts_pb2.ERROR_VOICE_NOT_FOUND, f"voice not found: {request.voice_id}", rid),
            )
        if request.metadata.name:
            model.name = request.metadata.name
        if request.metadata.language:
            model.language = request.metadata.language
        if request.metadata.gender:
            model.gender = request.metadata.gender
        if request.metadata.description:
            model.description = request.metadata.description
        return tts_pb2.UpdateVoiceResponse(request_id=rid, success=True, message="voice metadata updated")

    async def SetVoiceEnabled(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model(request.voice_id)
        if not model:
            return tts_pb2.SetVoiceEnabledResponse(
                request_id=rid,
                success=False,
                message=f"voice not found: {request.voice_id}",
                error=self._err(tts_pb2.ERROR_VOICE_NOT_FOUND, f"voice not found: {request.voice_id}", rid),
            )
        model.enabled = bool(request.enabled)
        if not self.models_by_voice.get(self.default_voice_id, model).enabled:
            for v in self.models_by_voice.values():
                if v.enabled:
                    self.default_voice_id = v.voice_id
                    break
        return tts_pb2.SetVoiceEnabledResponse(request_id=rid, success=True, message="voice state updated")

    async def ReloadVoice(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        model = self._resolve_model(request.voice_id)
        if not model:
            return tts_pb2.ReloadVoiceResponse(
                request_id=rid,
                success=False,
                message=f"voice not found: {request.voice_id}",
                reload_time_ms=0,
                error=self._err(tts_pb2.ERROR_VOICE_NOT_FOUND, f"voice not found: {request.voice_id}", rid),
            )
        ready, msg = self._model_ready(model)
        return tts_pb2.ReloadVoiceResponse(
            request_id=rid,
            success=ready,
            message="voice ready" if ready else msg,
            reload_time_ms=0,
            error=None if ready else self._err(tts_pb2.ERROR_MODEL_NOT_LOADED, msg, rid),
        )

    async def ValidateConfig(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        results = [
            tts_pb2.ValidationResult(
                check_name="config_parse",
                passed=bool(self.models_by_voice),
                message=f"models={len(self.models_by_voice)}",
                details=[],
            )
        ]
        errors = []
        warnings = []
        if request.check_files and not self.run_model_script.exists():
            errors.append(
                tts_pb2.ConfigValidationError(
                    file=str(self.run_model_script),
                    path="run_model_script",
                    message="runner script not found",
                    code="RUNNER_NOT_FOUND",
                )
            )
        if request.check_files:
            for model in self.models_by_voice.values():
                ok, _ = self._check_model_dir(model.model_dir)
                if not ok:
                    errors.append(
                        tts_pb2.ConfigValidationError(
                            file=model.model_dir,
                            path=f"models.{model.model_id}.model_dir",
                            message="model dir not found",
                            code="MODEL_DIR_NOT_FOUND",
                        )
                    )
        if request.check_dependencies:
            for model in self.models_by_voice.values():
                if not Path(model.python_exe).exists():
                    errors.append(
                        tts_pb2.ConfigValidationError(
                            file=model.python_exe,
                            path=f"models.{model.model_id}.python_exe",
                            message="python executable not found",
                            code="PYTHON_NOT_FOUND",
                        )
                    )
        if request.check_gpu and not self._gpu().available:
            warnings.append(
                tts_pb2.ConfigValidationWarning(
                    file="runtime",
                    path="gpu",
                    message="CUDA GPU not detected",
                )
            )
        valid = len(errors) == 0
        return tts_pb2.ValidateConfigResponse(
            request_id=rid,
            valid=valid,
            message="validation passed" if valid else "validation failed",
            results=results,
            errors=errors,
            warnings=warnings,
            timestamp_ms=self._now_ms(),
        )

    async def GetPoolStatus(self, request, context):  # noqa: N802
        rid = request.request_id if self._has_field(request, "request_id") else self._rid()
        uptime = max(1.0, time.time() - self.start_time)
        return tts_pb2.PoolStatusResponse(
            request_id=rid,
            runtime="subprocess",
            total_slots=self.max_concurrent,
            active_count=self.active_requests,
            waiting_count=0,
            available_slots=max(0, self.max_concurrent - self.active_requests),
            thread_pool_size=self.max_concurrent,
            workers_alive=self.max_concurrent,
            submitted_count=self.total_requests,
            completed_count=self.success_requests,
            failed_count=self.failed_requests,
            synthesis_active=self.active_requests,
            synthesis_total=self.total_requests,
            synthesis_success=self.success_requests,
            synthesis_failed=self.failed_requests,
            uptime_seconds=uptime,
            overall_rps=self.total_requests / uptime,
            batch_enabled=False,
            batch_submitted=0,
            batch_processed=0,
            queue_size=0,
            max_queue_size=self.max_concurrent,
            queue_utilization=0.0,
            queue_rejected=self.drain_rejected_count,
            queue_timeout=0,
            prometheus_enabled=False,
            timestamp_ms=self._now_ms(),
        )

    async def StartDrain(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        async with self.lock:
            self.drain_enabled = True
            self.drain_started_at_ms = self._now_ms()
            self.drain_start_success_count = self.success_requests
            status = (
                tts_pb2.DRAIN_STATUS_DRAINED
                if request.force or self.active_requests == 0
                else tts_pb2.DRAIN_STATUS_DRAINING
            )
            return tts_pb2.DrainResponse(
                request_id=rid,
                success=True,
                status=status,
                active_requests=self.active_requests,
                completed_requests=max(0, self.success_requests - self.drain_start_success_count),
                rejected_requests=self.drain_rejected_count,
                message="drain mode enabled",
                timestamp_ms=self._now_ms(),
            )

    async def GetDrainStatus(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        if not self.drain_enabled:
            status = tts_pb2.DRAIN_STATUS_NOT_DRAINING
        elif self.active_requests > 0:
            status = tts_pb2.DRAIN_STATUS_DRAINING
        else:
            status = tts_pb2.DRAIN_STATUS_DRAINED
        return tts_pb2.DrainStatusResponse(
            request_id=rid,
            status=status,
            active_requests=self.active_requests,
            completed_requests=max(0, self.success_requests - self.drain_start_success_count),
            rejected_requests=self.drain_rejected_count,
            drain_started_at_ms=self.drain_started_at_ms,
            uptime_ms=int((time.time() - self.start_time) * 1000),
            timestamp_ms=self._now_ms(),
        )

    async def StopDrain(self, request, context):  # noqa: N802
        rid = request.request_id or self._rid()
        async with self.lock:
            self.drain_enabled = False
            self.drain_started_at_ms = 0
        return tts_pb2.StopDrainResponse(
            request_id=rid,
            success=True,
            message="drain mode disabled",
            timestamp_ms=self._now_ms(),
        )


def _set_default_methods() -> None:
    def unary(name: str):
        async def _method(self: BenchmarkBridgeServicer, request: Any, context: Any) -> Any:
            rid = getattr(request, "request_id", "") or self._rid()
            if name == "AddVoice":
                return tts_pb2.AddVoiceResponse(
                    request_id=rid,
                    success=False,
                    message="AddVoice is not supported",
                    voice_id="",
                    error=self._err(tts_pb2.ERROR_INVALID_PARAMETERS, "AddVoice is not supported", rid),
                )
            if name == "RemoveVoice":
                return tts_pb2.RemoveVoiceResponse(
                    request_id=rid,
                    success=False,
                    message="RemoveVoice is not supported",
                    error=self._err(tts_pb2.ERROR_INVALID_PARAMETERS, "RemoveVoice is not supported", rid),
                )
            if name == "SynthesizeBatch":
                return tts_pb2.BatchSynthesizeResponse(
                    request_id=rid,
                    batch_id=f"batch-{uuid.uuid4()}",
                    status=tts_pb2.BATCH_STATUS_FAILED,
                    message="Batch synthesis is not supported",
                    total_count=0,
                    timestamp_ms=self._now_ms(),
                )
            if name == "GetBatchStatus":
                return tts_pb2.BatchStatusResponse(
                    request_id=rid,
                    batch_id="",
                    status=tts_pb2.BATCH_STATUS_UNKNOWN,
                    progress_percent=0,
                    total_count=0,
                    completed_count=0,
                    failed_count=0,
                    processing_time_ms=0,
                    estimated_remaining_ms=0,
                    message="Batch synthesis is not supported",
                    results=[],
                    timestamp_ms=self._now_ms(),
                )
            if name == "GetBackupStatus":
                return tts_pb2.GetBackupStatusResponse(
                    request_id=rid,
                    can_rollback=False,
                    backup_exists=False,
                    backup_created_at="",
                    config_file_count=0,
                    samples_file_count=0,
                    backup_path="",
                    timestamp_ms=self._now_ms(),
                    backup_valid=False,
                    validation_errors=[],
                    validation_warnings=[],
                )
            if name == "CreateBackup":
                return tts_pb2.CreateBackupResponse(
                    request_id=rid,
                    success=False,
                    backup_id="",
                    backup_path="",
                    message="Backup is not supported",
                    timestamp_ms=self._now_ms(),
                )
            if name == "ListBackups":
                return tts_pb2.ListBackupsResponse(
                    request_id=rid,
                    backups=[],
                    timestamp_ms=self._now_ms(),
                )
            raise NotImplementedError(name)

        return _method

    for method_name in (
        "AddVoice",
        "RemoveVoice",
        "SynthesizeBatch",
        "GetBatchStatus",
        "GetBackupStatus",
        "CreateBackup",
        "ListBackups",
    ):
        setattr(BenchmarkBridgeServicer, method_name, unary(method_name))


def _expand(text: str, mapping: dict[str, str]) -> str:
    out = text
    for key, value in mapping.items():
        out = out.replace(f"${{{key}}}", value)
    return os.path.expandvars(out)


def _load_models(path: Path, mapping: dict[str, str]) -> tuple[list[ModelConfig], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid models config: {path}")
    models = []
    for row in rows:
        models.append(
            ModelConfig(
                model_id=str(row["id"]),
                voice_id=str(row["voice_id"]),
                name=str(row.get("name", row["id"])),
                model_type=str(row.get("model_type", "benchmark")),
                run_model_key=str(row["run_model_key"]),
                python_exe=str(Path(_expand(str(row["python_exe"]), mapping))),
                model_dir=str(Path(_expand(str(row["model_dir"]), mapping))),
                language=str(row.get("language", "unknown")),
                gender=str(row.get("gender", "unknown")),
                sample_rate=int(row.get("sample_rate", 24000)),
                description=str(row.get("description", "")),
                enabled=bool(row.get("enabled", True)),
                timeout_s=int(row.get("timeout_s", 1800)),
            )
        )
    default_voice_id = str(payload.get("default_voice_id", models[0].voice_id))
    return models, default_voice_id


async def _serve(args: argparse.Namespace) -> None:
    bench_root = Path(args.bench_root).resolve()
    run_model_script = Path(args.run_model_script).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "BENCH_ROOT": str(bench_root),
        "MODELS_ROOT": str(Path(args.models_root).resolve()),
        "PY_LIST_ROOT": str(Path(args.py_list_root).resolve()),
    }
    models, default_voice_id = _load_models(Path(args.models_config).resolve(), mapping)

    servicer = BenchmarkBridgeServicer(
        bench_root=bench_root,
        run_model_script=run_model_script,
        output_dir=output_dir,
        model_instance=args.model_instance,
        max_concurrent=args.max_concurrent,
        max_text_length=args.max_text_length,
        keep_outputs=args.keep_outputs,
        default_timeout_s=args.default_timeout_s,
        models=models,
        default_voice_id=default_voice_id,
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", args.max_send_message_bytes),
            ("grpc.max_receive_message_length", args.max_receive_message_bytes),
        ]
    )
    tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)
    bind = f"{args.host}:{args.port}"
    server.add_insecure_port(bind)
    LOGGER.info("Proto source: %s", PROTO_BASE)
    LOGGER.info("Models loaded: %d (enabled=%d)", len(models), len([m for m in models if m.enabled]))
    LOGGER.info("Listening on %s", bind)
    await server.start()
    await server.wait_for_termination()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS benchmark gRPC bridge")
    root = Path(__file__).resolve().parent
    bench = root.parent
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8320)
    parser.add_argument("--bench-root", default=str(bench))
    parser.add_argument("--run-model-script", default=str(bench / "adapters" / "run_model.py"))
    parser.add_argument("--models-config", default=str(root / "models.json"))
    parser.add_argument("--models-root", default=os.getenv("MODELS_ROOT", "D:/models"))
    parser.add_argument("--py-list-root", default=os.getenv("PY_LIST_ROOT", "D:/list"))
    parser.add_argument("--output-dir", default=str(bench / "outputs" / "grpc_bridge"))
    parser.add_argument("--model-instance", default="tts-benchmark-bridge")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--max-text-length", type=int, default=400)
    parser.add_argument("--default-timeout-s", type=int, default=1800)
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--max-send-message-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-receive-message-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    _set_default_methods()
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        LOGGER.info("Stopped")


if __name__ == "__main__":
    main()
