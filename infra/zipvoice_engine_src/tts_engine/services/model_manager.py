# -*- coding: utf-8 -*-
"""모델 매니저 - 모델 인스턴스 라이프사이클 관리.

모델 풀 관리, 로드/언로드, 상태 모니터링을 담당합니다.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional

from tts_engine.core.config import ModelInstanceConfig, ModelTypeConfig

if TYPE_CHECKING:
    from tts_engine.services.voice_manager import VoiceManager
from tts_engine.core.constants import HealthStatus
from tts_engine.core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    ModelUnloadError,
    PoolExhaustedError,
)
from tts_engine.core.logging import get_logger, log_model_event
from tts_engine.domain.types import ModelInfo
from tts_engine.models.base import AbstractTTSModel
from tts_engine.models.factory import ModelFactory

logger = get_logger(__name__)


class ModelPool:
    """모델 인스턴스 풀.

    동일 설정의 모델 인스턴스들을 풀로 관리합니다.
    각 요청에 사용 가능한 인스턴스를 할당하고, 사용 후 반환합니다.

    Attributes:
        instance_name: 모델 인스턴스 이름
        pool_size: 풀 크기
        models: 모델 인스턴스 목록
    """

    def __init__(self, instance_name: str, pool_size: int = 1):
        """풀 초기화.

        Args:
            instance_name: 모델 인스턴스 이름
            pool_size: 풀 크기
        """
        self.instance_name = instance_name
        self.pool_size = pool_size
        self._models: List[AbstractTTSModel] = []
        self._available: asyncio.Queue[AbstractTTSModel] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self, factory: ModelFactory) -> None:
        """풀을 초기화하고 모델들을 로드합니다.

        Args:
            factory: 모델 팩토리

        Raises:
            ModelLoadError: 모델 로드 실패
        """
        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "Initializing model pool",
                instance_name=self.instance_name,
                pool_size=self.pool_size,
            )

            for i in range(self.pool_size):
                try:
                    model = await factory.create(
                        self.instance_name,
                        auto_load=True,
                    )
                    self._models.append(model)
                    await self._available.put(model)

                    log_model_event(
                        "model_loaded",
                        model_name=f"{self.instance_name}_{i}",
                        pool_index=i,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to load model in pool",
                        instance_name=self.instance_name,
                        pool_index=i,
                        error=str(e),
                    )
                    raise ModelLoadError(
                        self.instance_name,
                        f"Failed to initialize model pool: {e}"
                    )

            self._initialized = True
            logger.info(
                "Model pool initialized",
                instance_name=self.instance_name,
                loaded_count=len(self._models),
            )

    async def shutdown(self) -> None:
        """풀을 종료하고 모든 모델을 언로드합니다."""
        async with self._lock:
            logger.info(
                "Shutting down model pool",
                instance_name=self.instance_name,
            )

            # 모든 모델이 반환될 때까지 대기 (타임아웃 적용)
            try:
                await asyncio.wait_for(
                    self._wait_all_returned(),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for models to return",
                    instance_name=self.instance_name,
                )

            # 모든 모델 언로드
            for i, model in enumerate(self._models):
                try:
                    await model.unload()
                    log_model_event(
                        "model_unloaded",
                        model_name=f"{self.instance_name}_{i}",
                    )
                except Exception as e:
                    logger.error(
                        "Error unloading model",
                        instance_name=self.instance_name,
                        pool_index=i,
                        error=str(e),
                    )

            self._models.clear()
            self._available = asyncio.Queue()
            self._initialized = False

            logger.info(
                "Model pool shutdown complete",
                instance_name=self.instance_name,
            )

    async def _wait_all_returned(self) -> None:
        """모든 모델이 풀에 반환될 때까지 대기."""
        while self._available.qsize() < len(self._models):
            await asyncio.sleep(0.1)

    async def acquire(self, timeout: Optional[float] = None) -> AbstractTTSModel:
        """풀에서 사용 가능한 모델을 획득합니다.

        Args:
            timeout: 대기 타임아웃 (초)

        Returns:
            사용 가능한 모델 인스턴스

        Raises:
            PoolExhaustedError: 타임아웃 내 사용 가능한 모델이 없음
        """
        try:
            if timeout is not None:
                model = await asyncio.wait_for(
                    self._available.get(),
                    timeout=timeout,
                )
            else:
                model = await self._available.get()

            logger.debug(
                "Model acquired from pool",
                instance_name=self.instance_name,
                model_name=model.name,
                available=self._available.qsize(),
            )

            return model

        except asyncio.TimeoutError:
            raise PoolExhaustedError(
                self.instance_name,
                self.pool_size,
            )

    async def release(self, model: AbstractTTSModel) -> None:
        """사용한 모델을 풀에 반환합니다.

        Args:
            model: 반환할 모델
        """
        await self._available.put(model)

        logger.debug(
            "Model released to pool",
            instance_name=self.instance_name,
            model_name=model.name,
            available=self._available.qsize(),
        )

    @asynccontextmanager
    async def get_model(
        self,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[AbstractTTSModel, None]:
        """컨텍스트 매니저로 모델을 획득/반환합니다.

        Args:
            timeout: 획득 타임아웃

        Yields:
            사용 가능한 모델 인스턴스

        Example:
            >>> async with pool.get_model(timeout=10) as model:
            ...     result = await model.synthesize(request)
        """
        model = await self.acquire(timeout)
        try:
            yield model
        finally:
            await self.release(model)

    @property
    def available_count(self) -> int:
        """현재 사용 가능한 모델 수."""
        return self._available.qsize()

    @property
    def total_count(self) -> int:
        """전체 모델 수."""
        return len(self._models)

    @property
    def in_use_count(self) -> int:
        """사용 중인 모델 수."""
        return self.total_count - self.available_count

    def get_status(self) -> dict:
        """풀 상태 정보를 반환합니다."""
        return {
            "instance_name": self.instance_name,
            "pool_size": self.pool_size,
            "total": self.total_count,
            "available": self.available_count,
            "in_use": self.in_use_count,
            "initialized": self._initialized,
        }


class ModelManager:
    """모델 매니저 - 모델 라이프사이클 통합 관리.

    여러 모델 인스턴스의 풀을 관리하고, 음성 ID를 기준으로
    적절한 모델을 할당합니다.

    Example:
        >>> manager = ModelManager(model_types, model_instances, voice_manager)
        >>> await manager.initialize()
        >>> async with manager.get_model_for_voice("kor_female_01") as model:
        ...     result = await model.synthesize(request)
        >>> await manager.shutdown()
    """

    def __init__(
        self,
        model_types: Dict[str, ModelTypeConfig],
        model_instances: Dict[str, ModelInstanceConfig],
        voice_manager: Optional[VoiceManager] = None,
        skip_pool_runtimes: Optional[List[str]] = None,
    ):
        """매니저 초기화.

        Args:
            model_types: 모델 타입 설정
            model_instances: 모델 인스턴스 설정
            voice_manager: 음성 매니저 (모델에서 음성 정보 조회용)
            skip_pool_runtimes: 모델 풀을 생성하지 않을 런타임 목록 (예: ["onnx"])
                               이 런타임들은 별도 Executor 세션 풀 사용
        """
        self._voice_manager = voice_manager
        self._factory = ModelFactory(
            model_types, model_instances, voice_manager=voice_manager
        )
        self._pools: Dict[str, ModelPool] = {}
        self._voice_to_instance: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._model_instances = model_instances
        self._skip_pool_runtimes = skip_pool_runtimes or []

        logger.info(
            "ModelManager created",
            model_types=list(model_types.keys()),
            model_instances=list(model_instances.keys()),
            voice_manager_attached=voice_manager is not None,
            skip_pool_runtimes=self._skip_pool_runtimes,
        )

    async def initialize(self) -> None:
        """모든 모델 풀을 초기화합니다."""
        async with self._lock:
            if self._initialized:
                logger.warning("ModelManager already initialized")
                return

            logger.info("Initializing ModelManager")

            # 설정 검증
            errors = self._factory.validate_configuration()
            if errors:
                logger.error(
                    "Configuration validation failed",
                    errors=errors,
                )
                raise ModelLoadError(
                    "configuration",
                    f"Configuration validation failed: {errors}"
                )

            # enabled=true 인스턴스만 풀 생성
            enabled_instances = self._factory.list_enabled_instances()

            if not enabled_instances:
                logger.warning("No enabled model instances found")

            logger.info(
                "Loading enabled model instances",
                enabled_instances=enabled_instances,
                all_instances=self._factory.list_instances(),
            )

            for instance_name in enabled_instances:
                try:
                    config = self._factory.get_instance_config(instance_name)

                    # 런타임 타입 확인 (options.runtime)
                    runtime = None
                    if config.options and hasattr(config.options, "runtime"):
                        runtime = config.options.runtime

                    # 스킵 런타임인 경우 모델 풀 생성 건너뛰기
                    if runtime and runtime.lower() in [r.lower() for r in self._skip_pool_runtimes]:
                        logger.info(
                            "Skipping model pool creation for runtime",
                            instance_name=instance_name,
                            runtime=runtime,
                            reason="Uses dedicated executor session pool",
                        )
                        continue

                    pool = ModelPool(
                        instance_name=instance_name,
                        pool_size=config.pool_size,
                    )
                    await pool.initialize(self._factory)
                    self._pools[instance_name] = pool

                except Exception as e:
                    logger.error(
                        "Failed to initialize pool",
                        instance_name=instance_name,
                        error=str(e),
                    )
                    # 이미 생성된 풀 정리
                    await self._cleanup_pools()
                    raise

            self._initialized = True
            logger.info(
                "ModelManager initialized",
                pools=list(self._pools.keys()),
            )

    async def shutdown(self) -> None:
        """모든 모델 풀을 종료합니다."""
        async with self._lock:
            logger.info("Shutting down ModelManager")
            await self._cleanup_pools()
            self._initialized = False
            logger.info("ModelManager shutdown complete")

    async def _cleanup_pools(self) -> None:
        """모든 풀을 정리합니다."""
        for pool in self._pools.values():
            try:
                await pool.shutdown()
            except Exception as e:
                logger.error(
                    "Error shutting down pool",
                    instance_name=pool.instance_name,
                    error=str(e),
                )
        self._pools.clear()

    def register_voice_mapping(self, voice_id: str, instance_name: str) -> None:
        """음성 ID와 모델 인스턴스를 매핑합니다.

        Args:
            voice_id: 음성 ID
            instance_name: 모델 인스턴스 이름

        Raises:
            ModelNotFoundError: 인스턴스가 존재하지 않는 경우
        """
        if instance_name not in self._pools:
            raise ModelNotFoundError(
                f"Model instance '{instance_name}' not found. "
                f"Available: {list(self._pools.keys())}"
            )

        self._voice_to_instance[voice_id] = instance_name
        logger.debug(
            "Voice mapping registered",
            voice_id=voice_id,
            instance_name=instance_name,
        )

    def get_instance_for_voice(self, voice_id: str) -> str:
        """음성 ID에 매핑된 모델 인스턴스 이름을 반환합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            모델 인스턴스 이름

        Raises:
            ModelNotFoundError: 매핑이 없는 경우
        """
        if voice_id not in self._voice_to_instance:
            raise ModelNotFoundError(
                f"No model instance mapped for voice '{voice_id}'"
            )
        return self._voice_to_instance[voice_id]

    @asynccontextmanager
    async def get_model_for_voice(
        self,
        voice_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[AbstractTTSModel, None]:
        """음성 ID에 매핑된 모델을 획득합니다.

        Args:
            voice_id: 음성 ID
            timeout: 획득 타임아웃

        Yields:
            모델 인스턴스

        Raises:
            ModelNotFoundError: 매핑된 인스턴스가 없는 경우
            PoolExhaustedError: 사용 가능한 모델이 없는 경우
        """
        instance_name = self.get_instance_for_voice(voice_id)
        async with self.get_model(instance_name, timeout) as model:
            yield model

    @asynccontextmanager
    async def get_model(
        self,
        instance_name: str,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[AbstractTTSModel, None]:
        """특정 인스턴스의 모델을 획득합니다.

        Args:
            instance_name: 모델 인스턴스 이름
            timeout: 획득 타임아웃

        Yields:
            모델 인스턴스

        Raises:
            ModelNotFoundError: 인스턴스가 없는 경우
            PoolExhaustedError: 사용 가능한 모델이 없는 경우
        """
        if instance_name not in self._pools:
            raise ModelNotFoundError(
                f"Model instance '{instance_name}' not found"
            )

        pool = self._pools[instance_name]
        async with pool.get_model(timeout) as model:
            yield model

    async def warmup(
        self,
        test_text: str,
        repeat_count: int = 3,
        instance_name: Optional[str] = None,
    ) -> Dict[str, bool]:
        """모델을 워밍업합니다.

        Args:
            test_text: 테스트 텍스트
            repeat_count: 반복 횟수
            instance_name: 특정 인스턴스 이름 (None이면 모든 인스턴스)

        Returns:
            {instance_name: success} 딕셔너리
        """
        # 대상 인스턴스 결정
        if instance_name:
            if instance_name not in self._pools:
                logger.error(
                    "Instance not found for warmup",
                    instance_name=instance_name,
                )
                return {instance_name: False}
            target_pools = {instance_name: self._pools[instance_name]}
        else:
            target_pools = self._pools

        logger.info(
            "Warming up models",
            repeat_count=repeat_count,
            target_instances=list(target_pools.keys()),
        )

        results = {}

        for name, pool in target_pools.items():
            try:
                async with pool.get_model(timeout=30) as model:
                    success = await model.warmup(test_text, repeat_count)
                    results[name] = success
            except Exception as e:
                logger.error(
                    "Warmup failed for instance",
                    instance_name=name,
                    error=str(e),
                )
                results[name] = False

        logger.info(
            "Warmup complete",
            results=results,
        )

        return results

    def get_model_info(self, instance_name: str) -> dict:
        """특정 인스턴스의 모델 정보를 딕셔너리로 반환합니다.

        Args:
            instance_name: 모델 인스턴스 이름

        Returns:
            모델 정보 딕셔너리
        """
        import time

        result = {
            "model_type": "unknown",
            "model_path": "",
            "loaded_at": 0,
            "memory_usage_mb": 0,
            "supports_speed": False,
            "supports_pitch": False,
            "supports_volume": True,
        }

        if instance_name not in self._pools:
            return result

        pool = self._pools[instance_name]
        if not pool._models:
            return result

        model = pool._models[0]
        model_info = model.get_info()

        if model_info:
            result["model_type"] = getattr(model_info, "model_type", "unknown")
            result["model_path"] = getattr(model_info, "model_path", "")
            result["loaded_at"] = getattr(model_info, "loaded_at", time.time())
            result["memory_usage_mb"] = getattr(model_info, "memory_mb", 0)

            # 모델 capabilities 확인
            if hasattr(model, "capabilities"):
                caps = model.capabilities
                result["supports_speed"] = getattr(caps, "supports_speed", False)
                result["supports_pitch"] = getattr(caps, "supports_pitch", False)
                result["supports_volume"] = getattr(caps, "supports_volume", True)

        return result

    def get_all_model_info(self) -> Dict[str, ModelInfo]:
        """모든 모델의 정보를 반환합니다."""
        return {
            name: self.get_model_info(name)
            for name in self._pools.keys()
            if self.get_model_info(name) is not None
        }

    def get_pool(self, instance_name: str) -> Optional[ModelPool]:
        """특정 풀을 반환합니다.

        Args:
            instance_name: 모델 인스턴스 이름

        Returns:
            ModelPool 또는 None
        """
        return self._pools.get(instance_name)

    def get_pool_status(self, instance_name: str) -> Optional[dict]:
        """특정 풀의 상태를 반환합니다."""
        if instance_name not in self._pools:
            return None
        return self._pools[instance_name].get_status()

    def get_all_pool_status(self) -> Dict[str, dict]:
        """모든 풀의 상태를 반환합니다."""
        return {
            name: pool.get_status()
            for name, pool in self._pools.items()
        }

    def health_check(self) -> dict:
        """매니저 상태를 확인합니다."""
        pool_statuses = self.get_all_pool_status()

        # 전체 상태 계산
        total_available = sum(p["available"] for p in pool_statuses.values())
        total_models = sum(p["total"] for p in pool_statuses.values())
        all_healthy = all(
            p["initialized"] and p["available"] > 0
            for p in pool_statuses.values()
        )

        if not self._initialized:
            status = HealthStatus.UNHEALTHY
        elif all_healthy:
            status = HealthStatus.HEALTHY
        elif total_available > 0:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return {
            "status": status.value,
            "initialized": self._initialized,
            "total_pools": len(self._pools),
            "total_models": total_models,
            "available_models": total_available,
            "pools": pool_statuses,
            "voice_mappings": len(self._voice_to_instance),
        }

    def list_instances(self) -> List[str]:
        """등록된 인스턴스 목록을 반환합니다."""
        return list(self._pools.keys())

    def list_model_instances(self) -> List[str]:
        """등록된 모델 인스턴스 목록을 반환합니다 (별칭)."""
        return self.list_instances()

    def list_mapped_voices(self) -> Dict[str, str]:
        """음성-인스턴스 매핑을 반환합니다."""
        return dict(self._voice_to_instance)

    @property
    def is_initialized(self) -> bool:
        """초기화 여부."""
        return self._initialized

    async def reload_model(
        self,
        instance_name: str,
        reload_voices: bool = False,
    ) -> None:
        """특정 모델 인스턴스를 리로드합니다.

        Args:
            instance_name: 모델 인스턴스 이름
            reload_voices: 음성도 함께 리로드할지 여부

        Raises:
            ModelNotFoundError: 인스턴스가 없는 경우
            ModelLoadError: 리로드 실패
        """
        if instance_name not in self._pools:
            raise ModelNotFoundError(
                f"Model instance '{instance_name}' not found"
            )

        logger.info(
            "Reloading model instance",
            instance_name=instance_name,
            reload_voices=reload_voices,
        )

        async with self._lock:
            # 기존 풀 종료
            old_pool = self._pools[instance_name]
            await old_pool.shutdown()

            # 새 풀 생성
            try:
                config = self._factory.get_instance_config(instance_name)
                new_pool = ModelPool(
                    instance_name=instance_name,
                    pool_size=config.pool_size,
                )
                await new_pool.initialize(self._factory)
                self._pools[instance_name] = new_pool

                logger.info(
                    "Model instance reloaded",
                    instance_name=instance_name,
                )

            except Exception as e:
                logger.error(
                    "Failed to reload model instance",
                    instance_name=instance_name,
                    error=str(e),
                )
                raise ModelLoadError(
                    instance_name,
                    f"Failed to reload: {e}"
                )

        # 음성 리로드
        if reload_voices and self._voice_manager:
            voices = self._voice_manager.list_voices_by_model(instance_name)
            for voice in voices:
                try:
                    await self._voice_manager.reload_voice(voice.voice_id)
                except Exception as e:
                    logger.warning(
                        "Failed to reload voice",
                        voice_id=voice.voice_id,
                        error=str(e),
                    )
