#!/usr/bin/env python3
"""
Petco Retail Agent - CLI Interface.

Interactive command-line interface for the Petco shopping assistant.
"""

import argparse
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agent import PetcoRetailAgent, create_petco_agent
from multi_agent import create_petco_multi_agent

from orchestrator import LogLevel, setup_logging


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner():
    """Print welcome banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   🐕  Welcome to Petco Shopping Assistant  🐱                     ║
║                                                                    ║
║   Your AI-powered pet shopping companion                          ║
║                                                                    ║
╚═══════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    print(banner)


def print_help():
    """Print help information."""
    help_text = f"""
{Colors.YELLOW}{Colors.BOLD}Commands:{Colors.END}
  {Colors.GREEN}/help{Colors.END}     - Show this help message
  {Colors.GREEN}/tools{Colors.END}    - List available tools
  {Colors.GREEN}/debug{Colors.END}    - Toggle debug logging (shows tool calls)
  {Colors.GREEN}/clear{Colors.END}    - Clear screen
  {Colors.GREEN}/session{Colors.END}  - Show session info
  {Colors.GREEN}/quit{Colors.END}     - Exit the application

{Colors.YELLOW}{Colors.BOLD}Example queries:{Colors.END}
  • "Show me dog food options"
  • "I have a 2-year old golden retriever, what food do you recommend?"
  • "Add the first item to my cart"
  • "What's in my cart?"
  • "I'd like to checkout"
  • "Track my order #12345"
"""
    print(help_text)


def print_tools(tools: list):
    """Print available tools."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Available Tools ({len(tools)}):{Colors.END}\n")

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")[:80]
        print(f"  {Colors.GREEN}• {name}{Colors.END}")
        print(f"    {Colors.CYAN}{desc}{Colors.END}")

    print()


def print_session_info(agent: PetcoRetailAgent):
    """Print session information."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Session Info:{Colors.END}")
    print(f"  User ID:    {Colors.CYAN}{agent.user_id}{Colors.END}")
    print(f"  Session ID: {Colors.CYAN}{agent.session_id or 'N/A'}{Colors.END}")
    print()


async def main():
    """Main CLI loop."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Petco Shopping Assistant CLI")
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Use multi-agent (Plan-and-Execute) architecture instead of single agent",
    )
    args = parser.parse_args()

    # Setup logging - start with INFO to show some useful logs
    debug_mode = False
    setup_logging(level=LogLevel.INFO)

    print_banner()

    if args.multi_agent:
        print(f"{Colors.CYAN}Initializing multi-agent system (Plan-and-Execute)...{Colors.END}\n")
    else:
        print(f"{Colors.CYAN}Initializing agent...{Colors.END}\n")

    try:
        # Create and initialize agent
        if args.multi_agent:
            agent = await create_petco_multi_agent()
            agent_type = "Multi-Agent System"
        else:
            agent = await create_petco_agent()
            agent_type = "Agent"

        print(f"{Colors.GREEN}✓ {agent_type} ready!{Colors.END}")
        print(f"{Colors.CYAN}Connected to MCP with {len(agent.tools)} tools{Colors.END}")
        if args.multi_agent:
            print(f"{Colors.CYAN}Mode: Plan-and-Execute (Orchestrator + Executor){Colors.END}")
        print(f"\nType {Colors.GREEN}/help{Colors.END} for commands or start chatting!")
        print(f"Use {Colors.GREEN}/debug{Colors.END} to see detailed tool call logs.\n")

    except Exception as e:
        print(f"{Colors.RED}Failed to initialize agent: {e}{Colors.END}")
        print(f"\n{Colors.YELLOW}Make sure:{Colors.END}")
        print("  1. The MCP server is accessible")
        print("  2. Required environment variables are set (OPENAI_API_KEY)")
        print("  3. Docker services are running (if using memory/session)")
        return

    try:
        while True:
            try:
                # Get user input
                user_input = input(f"{Colors.GREEN}You:{Colors.END} ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()

                    if command == "/quit" or command == "/exit":
                        print(f"\n{Colors.CYAN}Thank you for shopping at Petco! 🐾{Colors.END}\n")
                        break

                    elif command == "/help":
                        print_help()
                        continue

                    elif command == "/tools":
                        print_tools(agent.tools)
                        continue

                    elif command == "/debug":
                        debug_mode = not debug_mode
                        if debug_mode:
                            setup_logging(level=LogLevel.DEBUG)
                            print(
                                f"{Colors.YELLOW}Debug mode ON - showing all tool call logs{Colors.END}\n"
                            )
                        else:
                            setup_logging(level=LogLevel.INFO)
                            print(f"{Colors.YELLOW}Debug mode OFF - reduced logging{Colors.END}\n")
                        continue

                    elif command == "/clear":
                        os.system("clear" if os.name == "posix" else "cls")
                        print_banner()
                        continue

                    elif command == "/session":
                        print_session_info(agent)
                        continue

                    else:
                        print(
                            f"{Colors.YELLOW}Unknown command. Type /help for available commands.{Colors.END}"
                        )
                        continue

                # Process message
                print(f"\n{Colors.CYAN}Thinking...{Colors.END}")

                response = await agent.chat(user_input)

                # Print response
                print(f"\n{Colors.BLUE}{Colors.BOLD}Petco Assistant:{Colors.END}")
                print(f"{response}\n")

            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYAN}Thank you for shopping at Petco! 🐾{Colors.END}\n")
                break

            except Exception as e:
                print(f"\n{Colors.RED}Error: {e}{Colors.END}\n")
                continue

    finally:
        # Cleanup
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
