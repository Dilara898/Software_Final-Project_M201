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


settings = Settings()