#!/usr/bin/env python3
"""
Interactive chat з knowledge-centered agent (TEACH/SOLVE flow).

Використовує production agent через /text endpoint logic:
- TEACH: навчання агента новим фактам
- SOLVE: відповіді на питання з використанням збережених знань

Usage:
    python demochat.py

Legacy version: demochat_legacy.py
"""

import asyncio
import sys
from uuid import uuid4

from routers.text import process_text
from routers.schemas import TextRequest
from clients.graphiti_client import get_graphiti_client
from config.settings import settings


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
    GRAY = '\033[90m'


def print_banner():
    """Print welcome banner."""
    banner = f"""
{Colors.CYAN}{'=' * 70}
{Colors.BOLD}🤖 LAPA AI CHAT - Knowledge-Centered Agent{Colors.RESET}
{Colors.CYAN}{'=' * 70}{Colors.RESET}

{Colors.YELLOW}Цей агент може:{Colors.RESET}
  🎓 {Colors.GREEN}НАВЧАТИСЯ{Colors.RESET} - зберігає факти які ви йому повідомляєте
  💡 {Colors.GREEN}ВІДПОВІДАТИ{Colors.RESET} - використовує збережені знання для відповідей

{Colors.YELLOW}Команди:{Colors.RESET}
  {Colors.GREEN}/exit{Colors.RESET}   - Вийти з чату
  {Colors.GREEN}/clear{Colors.RESET}  - Очистити екран
  {Colors.GREEN}/stats{Colors.RESET}  - Показати статистику пам'яті
  {Colors.GREEN}/help{Colors.RESET}   - Показати цю довідку

{Colors.YELLOW}Приклади:{Colors.RESET}
  {Colors.GRAY}TEACH: "Мене звати Ігор, я з Києва"
  {Colors.GRAY}SOLVE: "Як мене звати?"
  {Colors.GRAY}TEACH: "Столиця України - Київ"
  {Colors.GRAY}SOLVE: "Яка столиця України?"{Colors.RESET}

{Colors.MAGENTA}Модель:{Colors.RESET} {settings.model_name}
{Colors.MAGENTA}Embeddings:{Colors.RESET} {'Hosted Qwen' if settings.use_hosted_embeddings else settings.embedding_model_name}
{Colors.MAGENTA}Reranking:{Colors.RESET} 'Disabled 🚀'
{Colors.CYAN}{'=' * 70}{Colors.RESET}
"""
    print(banner)


def print_user(message: str):
    """Print user message."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Ви:{Colors.RESET} {message}")


def print_agent(response: str, intent: str = None, references: list = None, reasoning: str = None):
    """Print agent response with metadata."""
    # Intent badge
    if intent == "teach":
        intent_badge = f"{Colors.YELLOW}[НАВЧАННЯ]{Colors.RESET}"
    elif intent == "solve":
        intent_badge = f"{Colors.CYAN}[ВІДПОВІДЬ]{Colors.RESET}"
    else:
        intent_badge = ""

    print(f"{Colors.GREEN}{Colors.BOLD}Агент:{Colors.RESET} {intent_badge}")
    print(f"{response}")

    # References
    if references and len(references) > 0:
        refs_str = ", ".join(references[:3])  # Show max 3
        if len(references) > 3:
            refs_str += f" (+{len(references) - 3} more)"
        print(f"{Colors.GRAY}📎 References: {refs_str}{Colors.RESET}")

    # Reasoning (if available)
    if reasoning:
        print(f"{Colors.GRAY}💭 Reasoning: {reasoning[:100]}...{Colors.RESET}")

    print()


def print_system(message: str):
    """Print system message."""
    print(f"{Colors.YELLOW}[Система]{Colors.RESET} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}[Помилка]{Colors.RESET} {message}")


async def show_stats():
    """Show graph memory statistics."""
    try:
        graphiti = await get_graphiti_client()
        stats = await graphiti.get_graph_stats()

        print(f"\n{Colors.CYAN}{'=' * 70}")
        print(f"{Colors.BOLD}📊 Статистика Knowledge Graph{Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
        print(f"  🔷 Вузлів (Entities): {Colors.GREEN}{stats['node_count']}{Colors.RESET}")
        print(f"  🔗 Зв'язків (Relations): {Colors.GREEN}{stats['relationship_count']}{Colors.RESET}")

        # Calculate knowledge richness
        if stats['node_count'] > 0:
            avg_relations = stats['relationship_count'] / stats['node_count']
            print(f"  📈 Середньо зв'язків на вузол: {Colors.GREEN}{avg_relations:.1f}{Colors.RESET}")

        print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")
    except Exception as e:
        print_error(f"Не вдалося отримати статистику: {e}")


async def main():
    """Main chat loop."""
    print_banner()

    # Get user ID
    print(f"{Colors.MAGENTA}Введіть ваш ID{Colors.RESET} (або залиште порожнім для 'demo_user'):")
    user_id = input(f"{Colors.BOLD}> {Colors.RESET}").strip()

    if not user_id:
        user_id = "demo_user"
        print_system(f"Використовується ID: {user_id}")

    print_system(f"Чат почато для користувача: {Colors.BOLD}{user_id}{Colors.RESET}")
    print_system(f"Всі знання будуть збережені в Graphiti knowledge graph")
    print_system(f"Введіть {Colors.GREEN}/help{Colors.RESET} для списку команд\n")

    # Initialize agent (check if it works)
    try:
        print_system("Ініціалізація knowledge-centered agent...")
        graphiti = await get_graphiti_client()
        print_system(f"✅ Агент готовий! Graphiti підключено.\n")
    except Exception as e:
        print_error(f"Не вдалося ініціалізувати агента: {e}")
        print_system("Перевірте чи запущено Neo4j: docker-compose up -d")
        return

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
                    print_system(f"Дякую за розмову! Всього {message_count} повідомлень оброблено.")
                    print_system("До побачення! 👋\n")
                    break

                elif command == '/clear':
                    # Clear screen
                    print('\033[2J\033[H', end='')
                    print_banner()
                    print_system(f"Продовжуємо чат для: {Colors.BOLD}{user_id}{Colors.RESET}\n")
                    continue

                elif command == '/stats':
                    await show_stats()
                    continue

                elif command == '/help':
                    print_banner()
                    continue

                else:
                    print_error(f"Невідома команда: {command}")
                    print_system(f"Введіть /help для списку команд")
                    continue

            # Process user message with knowledge-centered agent
            try:
                # Generate unique message UID
                msg_uid = f"msg_{uuid4().hex[:8]}"

                # Create request
                request = TextRequest(
                    uid=msg_uid,
                    text=user_input,
                    user_id=user_id
                )

                # Show processing indicator
                print(f"{Colors.GRAY}⏳ Обробка...{Colors.RESET}", end='\r')

                # Process through agent
                response = await process_text(request)

                # Clear processing indicator
                print(' ' * 50, end='\r')

                # Display response
                print_agent(
                    response=response.response,
                    intent=None,  # Intent not exposed in TextResponse
                    references=response.references,
                    reasoning=response.reasoning
                )

                message_count += 1

            except Exception as e:
                print_error(f"Помилка обробки повідомлення: {e}")
                print_system("Спробуйте ще раз або введіть /exit для виходу")
                import traceback
                print(f"{Colors.GRAY}{traceback.format_exc()}{Colors.RESET}")

        except KeyboardInterrupt:
            print_system("\n\nПерервано користувачем (Ctrl+C)")
            print_system(f"Всього {message_count} повідомлень оброблено.")
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