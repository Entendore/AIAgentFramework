"""Rich CLI interface for Synaptiq."""
import asyncio
import sys
import os
import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from ..core.brain import SynaptiqBrain

logger = logging.getLogger(__name__)

BANNER = """
███████╗██╗   ██╗███╗   ██╗ █████╗ ███████╗████████╗███████╗██╗
██╔════╝██║   ██║████╗  ██║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██║
███████╗██║   ██║██╔██╗ ██║███████║███████╗   ██║   █████╗  ██║
╚════██║██║   ██║██║╚███╔╝██╔══██║╚════██║   ██║   ██╔══╝  ██║
███████║╚██████╔╝██║ ╚████║╚█████╔╝███████║   ██║   ███████╗███████╗
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚════╝ ╚══════╝   ╚═╝   ╚══════╝╚══════╝
"""

class SynaptiqCLI:
    def __init__(self):
        self.console = Console()
        self.brain: Optional[SynaptiqBrain] = None

    async def run(self):
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.console.print("[bold red]Error: API_KEY environment variable not set.[/bold red]")
            sys.exit(1)
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"

        if len(sys.argv) > 1 and sys.argv[1] == "install":
            if len(sys.argv) < 3: self.console.print("[red]Usage: synaptiq install <git-url>[/red]"); return
            self.brain = SynaptiqBrain(api_key, provider=provider)
            await self.brain.graft_manager.install_from_git(sys.argv[2]); return

        self.brain = SynaptiqBrain(api_key, provider=provider)
        await self.brain.initialize()

        self.console.print(Panel(Text(BANNER, style="bold cyan"), border_style="blue"))
        self.console.print(f"[bold green]Synaptiq Cognitive Engine v{self.brain.version} Initialized[/bold green]")
        self.console.print("[dim]Type 'exit' to quit. Type 'help' for commands.[/dim]\n")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: Prompt.ask("[bold magenta]You ❯[/bold magenta]"))
                if user_input.lower() in ['exit', 'quit']:
                    await self.brain.shutdown(); break
                if user_input.lower() == 'help': self._show_help(); continue
                if user_input.lower() == 'status': self._show_status(); continue
                if user_input.lower() == 'memory': self._show_memory(); continue
                if user_input.lower() == 'tools': self._show_tools(); continue
                if user_input.lower() == 'extensions': self._show_extensions(); continue
                if user_input.lower() == 'session': self._show_session(); continue
                if not user_input.strip(): continue

                self.console.print("[bold cyan]Synaptiq ❯[/bold cyan] [dim]Thinking...[/dim]", end="\r")
                async for response in self.brain.think(user_input):
                    self.console.print(f"[bold cyan]Synaptiq ❯[/bold cyan] ")
                    self.console.print(Markdown(response))
            except KeyboardInterrupt: self.console.print("\n[bold yellow]Interrupted[/bold yellow]"); continue
            except Exception as e: self.console.print(f"\n[bold red]Error: {e}[/bold red]")

    def _show_help(self):
        table = Table(title="Synaptiq Commands")
        table.add_column("Command", style="cyan"); table.add_column("Description", style="green")
        table.add_row("help", "Show this help"); table.add_row("status", "Show brain status")
        table.add_row("memory", "Show memory stats"); table.add_row("tools", "List tools")
        table.add_row("extensions", "List grafts"); table.add_row("session", "Show session tree")
        self.console.print(table)

    def _show_status(self):
        status = self.brain.get_status()
        table = Table(title="Synaptiq Status")
        table.add_column("Property", style="cyan"); table.add_column("Value", style="green")
        for k, v in status.items(): table.add_row(k, str(v))
        self.console.print(table)

    def _show_memory(self):
        stats = self.brain.memory.get_stats()
        table = Table(title="Memory Statistics")
        table.add_column("Metric", style="cyan"); table.add_column("Value", style="green")
        for k, v in stats.items(): table.add_row(k, str(v))
        self.console.print(table)

    def _show_tools(self):
        tools = self.brain.registry.list_tools()
        table = Table(title="Registered Tools")
        table.add_column("Name", style="cyan"); table.add_column("Description", style="green")
        for n, d in tools.items(): table.add_row(n, d[:80])
        self.console.print(table)

    def _show_extensions(self):
        if not self.brain.graft_manager.loaded_grafts: self.console.print("[yellow]No extensions loaded.[/yellow]"); return
        table = Table(title="Loaded Grafts")
        table.add_column("Name", style="cyan"); table.add_column("Version", style="green")
        for n, g in self.brain.graft_manager.loaded_grafts.items(): table.add_row(n, getattr(g, 'version', 'unknown'))
        self.console.print(table)

    def _show_session(self):
        tree = self.brain.session.get_session_tree()
        table = Table(title="Session Tree")
        table.add_column("Node ID", style="cyan"); table.add_column("Role", style="green")
        table.add_column("Messages", style="yellow"); table.add_column("Active", style="magenta")
        for n in tree["nodes"]: table.add_row(n["id"], n["role"], str(n["message_count"]), "→" if n["is_active"] else "")
        self.console.print(table)
