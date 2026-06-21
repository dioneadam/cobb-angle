"""
Carrega e expõe a configuração central do projeto.
"""
from pathlib import Path
import yaml


def load_config(path: str | Path = None) -> dict:
    """Carrega o config.yaml e retorna como dicionário."""
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Instância global reutilizável
_cfg: dict | None = None


def get_config() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
