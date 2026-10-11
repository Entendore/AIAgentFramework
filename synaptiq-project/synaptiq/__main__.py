"""CLI entry point for Synaptiq."""
import asyncio
import sys
import os
from pathlib import Path

from .ui.cli import SynaptiqCLI

def main():
    """Main CLI entry point."""
    cli = SynaptiqCLI()
    asyncio.run(cli.run())

if __name__ == "__main__":
    main()
