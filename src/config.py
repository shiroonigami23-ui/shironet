"""Configuration templates for GHOST-REF-ZERO integrations."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class NeonConfig:
    """Neon Postgres connection details."""

    database_url: str
    pooler_url: str
    branch: str


@dataclass(frozen=True)
class SupabaseConfig:
    """Supabase project credentials and storage settings."""

    url: str
    anon_key: str
    service_role_key: str
    bucket_models: str
    bucket_datasets: str


@dataclass(frozen=True)
class GhostRefZeroConfig:
    """Runtime controls for adversarial and distillation workflows."""

    mode: str
    adversarial_level: str
    distillation_cache_dir: str
    checkpoint_backend: str


@dataclass(frozen=True)
class AppConfig:
    neon: NeonConfig
    supabase: SupabaseConfig
    ghost: GhostRefZeroConfig


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def load_config() -> AppConfig:
    """Load all GHOST-REF-ZERO environment variables."""

    neon = NeonConfig(
        database_url=_env("NEON_DATABASE_URL"),
        pooler_url=_env("NEON_POOLER_URL"),
        branch=_env("NEON_BRANCH", "main"),
    )

    supabase = SupabaseConfig(
        url=_env("SUPABASE_URL"),
        anon_key=_env("SUPABASE_ANON_KEY"),
        service_role_key=_env("SUPABASE_SERVICE_ROLE_KEY"),
        bucket_models=_env("SUPABASE_BUCKET_MODELS", "shironet-models"),
        bucket_datasets=_env("SUPABASE_BUCKET_DATASETS", "shironet-datasets"),
    )

    ghost = GhostRefZeroConfig(
        mode=_env("GHOST_REF_ZERO_MODE", "strict"),
        adversarial_level=_env("GHOST_ADV_LEVEL", "medium"),
        distillation_cache_dir=_env("GHOST_DISTILL_CACHE", "./data/distill_cache"),
        checkpoint_backend=_env("GHOST_CHECKPOINT_BACKEND", "huggingface"),
    )

    return AppConfig(neon=neon, supabase=supabase, ghost=ghost)
