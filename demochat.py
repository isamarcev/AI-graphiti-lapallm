#!/usr/bin/env python3
"""
Simple interactive chat with memory-enabled agent.

Usage:
    python demochat.py
"""

import asyncio
import sys
from datetime import datetime
from langchain_core.messages import HumanMessage

from config.settings import settings
from clients.graphiti_client import get_graphiti_client
from agent.graph import get_agent_app


# ANSI color codes for pretty output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_banner():
    """Print welcome banner."""
    banner = f"""
{Colors.CYAN}{'=' * 60}
{Colors.BOLD}🤖 LAPA AI CHAT - Агент з довготривалою пам'яттю{Colors.RESET}
{Colors.CYAN}{'=' * 60}{Colors.RESET}

{Colors.YELLOW}Команди:{Colors.RESET}
  {Colors.GREEN}/exit{Colors.RESET}   - Вийти з чату
  {Colors.GREEN}/clear{Colors.RESET}  - Очистити екран
  {Colors.GREEN}/stats{Colors.RESET}  - Показати статистику пам'яті
  {Colors.GREEN}/help{Colors.RESET}   - Показати цю довідку

{Colors.MAGENTA}Модель:{Colors.RESET} {settings.vllm_model_name}
{Colors.MAGENTA}Embeddings:{Colors.RESET} {settings.embedding_model_name}
{Colors.CYAN}{'=' * 60}{Colors.RESET}
"""
    print(banner)


def print_user(message: str):
    """Print user message."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Ви:{Colors.RESET} {message}")


def print_agent(message: str):
    """Print agent response."""
    print(f"{Colors.GREEN}{Colors.BOLD}Агент:{Colors.RESET} {message}\n")


def print_system(message: str):
    """Print system message."""
    print(f"{Colors.YELLOW}[Система]{Colors.RESET} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}[Помилка]{Colors.RESET} {message}")


async def show_stats(graphiti_client):
    """Show graph memory statistics."""
    try:
        stats = await graphiti_client.get_graph_stats()
        print(f"\n{Colors.CYAN}{'=' * 60}")
        print(f"{Colors.BOLD}📊 Статистика пам'яті{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"  Вузлів (Entity): {Colors.GREEN}{stats['node_count']}{Colors.RESET}")
        print(f"  Зв'язків (Relations): {Colors.GREEN}{stats['relationship_count']}{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")
    except Exception as e:
        print_error(f"Не вдалося отримати статистику: {e}")


async def main():
    """Main chat loop."""
    print_banner()

    # Get chat session name from user
    print(f"{Colors.MAGENTA}Введіть назву чату{Colors.RESET} (наприклад: робота, навчання, особисте):")
    session_name = input(f"{Colors.BOLD}> {Colors.RESET}").strip()

    if not session_name:
        session_name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print_system(f"Використовується автоматична назва: {session_name}")

    # Generate session ID and user ID
    session_id = f"session_{session_name}"
    user_id = "user_1"  # Can be customized if needed

    print_system(f"Чат почато: {Colors.BOLD}{session_name}{Colors.RESET}")
    print_system(f"Всі розмови будуть збережені в граф пам'яті")
    print_system(f"Введіть {Colors.GREEN}/help{Colors.RESET} для списку команд\n")

    # Initialize agent and graphiti
    try:
        print_system("Ініціалізація агента...")
        agent = get_agent_app()
        graphiti = await get_graphiti_client()
        print_system(f"✅ Агент готовий!\n")
    except Exception as e:
        print_error(f"Не вдалося ініціалізувати агента: {e}")
        return

    # Chat configuration
    config = {"configurable": {"thread_id": session_id}}
    message_count = 0

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BLUE}{Colors.BOLD}Ви: {Colors.RESET}").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/exit' or command == '/quit':
                    print_system(f"Дякую за розмову! Всього {message_count} повідомлень збережено.")
                    print_system("До побачення! 👋\n")
                    break

                elif command == '/clear':
                    # Clear screen
                    print('\033[2J\033[H', end='')
                    print_banner()
                    print_system(f"Продовжуємо чат: {Colors.BOLD}{session_name}{Colors.RESET}\n")
                    continue

                elif command == '/stats':
                    await show_stats(graphiti)
                    continue

                elif command == '/help':
                    print_banner()
                    continue

                else:
                    print_error(f"Невідома команда: {command}")
                    print_system(f"Введіть /help для списку команд")
                    continue

            # Process user message with agent
            try:
                # Clean user input from potential encoding issues
                try:
                    user_input = user_input.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    pass

                message = HumanMessage(content=user_input)

                # Invoke agent
                result = await agent.ainvoke(
                    {
                        "messages": [message],
                        "user_id": user_id,
                        "session_id": session_id,
                        "retrieved_context": None,
                        "timestamp": datetime.now(),
                        "current_query": None,
                        "needs_memory_update": False,
                        "search_results": None,
                        "message_count": message_count
                    },
                    config=config
                )

                # Get agent response
                agent_response = result['messages'][-1].content

                # Clean response from potential encoding issues
                try:
                    agent_response = agent_response.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    pass

                print_agent(agent_response)

                message_count += 1

            except UnicodeEncodeError as e:
                print_error(f"Помилка кодування тексту: {e}")
                print_system("Hosted API повернув некоректні символи. Спробуйте переформулювати запит.")
            except Exception as e:
                print_error(f"Помилка обробки повідомлення: {e}")
                print_system("Спробуйте ще раз або введіть /exit для виходу")

        except KeyboardInterrupt:
            print_system("\n\nПерервано користувачем (Ctrl+C)")
            print_system(f"Всього {message_count} повідомлень збережено.")
            print_system("До побачення! 👋\n")
            break

        except EOFError:
            print_system("\n\nКінець введення (Ctrl+D)")
            break

        except Exception as e:
            print_error(f"Несподівана помилка: {e}")
            print_system("Продовжуємо роботу...\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_system("\n\nДо побачення! 👋\n")
        sys.exit(0)
