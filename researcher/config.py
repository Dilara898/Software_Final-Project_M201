from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ai/ paketinin oxuduqları, burada da elan edirik ki, sənədləşsin
    llm_provider: str = "gemini"
    web_search_provider: str = "tavily"

    # SE qatının öz parametrləri
    database_url: str
    log_level: str = "INFO"
    cache_ttl_seconds: int = 86_400
    per_source_timeout_seconds: float = 10.0
    max_sources_per_query: int = 3


@lru_cache
def get_settings() -> Settings:
    """Settings obyektini YALNIZ ilk çağırışda yaradır və keşləyir.

    Bu, modul import olunan anda deyil, faktiki istifadə anında
    DATABASE_URL kimi məcburi sahələrin yoxlanmasını təmin edir —
    beləliklə `import researcher.config` özü, .env mövcud olmasa belə,
    artıq uğursuz olmur; xəta yalnız settings faktiki lazım olanda çıxır.
    """
    return Settings()