"""graphsight-server: Zero-dependency local trace viewer and CLI."""

__version__ = "0.1.0"
__all__ = ["make_server", "main"]

from .server import make_server
from .cli import main
