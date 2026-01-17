#!/usr/bin/env python3
"""
Entrypoint script для запуска uvicorn с оптимальным количеством workers.
Автоматически вычисляет workers на основе CPU cores: (CPU × 2) + 1
"""
import os
import multiprocessing
import subprocess
import sys


def get_optimal_workers() -> int:
    """
    Вычисляет оптимальное количество workers.

    Формула: (CPU cores × 2) + 1
    Можно переопределить через ENV: WORKERS

    Returns:
        int: Количество workers
    """
    # Проверяем ENV переменную
    env_workers = os.getenv("WORKERS")
    if env_workers:
        try:
            workers = int(env_workers)
            if workers > 0:
                print(f"✓ Using WORKERS from ENV: {workers}")
                return workers
            else:
                print(f"⚠ Invalid WORKERS value in ENV ({env_workers}), using auto-detection")
        except ValueError:
            print(f"⚠ Invalid WORKERS value in ENV ({env_workers}), using auto-detection")

    # Автоматическое вычисление
    cpu_count = multiprocessing.cpu_count()
    workers = (cpu_count * 2) + 1

    # Ограничиваем максимум 8 workers для стабильности
    max_workers = 8
    if workers > max_workers:
        print(f"ℹ Calculated {workers} workers, limiting to {max_workers} for stability")
        workers = max_workers

    print(f"✓ Auto-detected {cpu_count} CPU cores")
    print(f"✓ Starting with {workers} workers")

    return workers


def main():
    """Запускает uvicorn с оптимальным количеством workers."""
    workers = get_optimal_workers()

    # Формируем команду uvicorn
    cmd = [
        "uvicorn",
        "app:app",
        "--host", "0.0.0.0",
        "--port", "3000",
        "--workers", str(workers),
    ]

    print(f"🚀 Starting uvicorn: {' '.join(cmd)}")
    print("=" * 60)

    # Запускаем uvicorn
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Uvicorn failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠ Shutting down gracefully...")
        sys.exit(0)


if __name__ == "__main__":
    main()