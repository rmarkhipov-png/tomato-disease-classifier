# scripts/run_bot.py
import asyncio
import os
from leafdoctor.bot import run_bot

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

if not TOKEN:
    raise ValueError("Задай переменную окружения TELEGRAM_TOKEN")

asyncio.run(run_bot(TOKEN))