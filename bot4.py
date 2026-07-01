"""
bot.py — Telegram-бот на aiogram 3.x.

Пользователь присылает фото листа — бот отвечает диагнозом.

Запуск:
    1) создай бота у @BotFather, получи токен;
    2) положи токен в файл .env (см. .env.example):  TELEGRAM_BOT_TOKEN=123:ABC...
    3) python -m leafdoctor.bot

Токен читается ТОЛЬКО из окружения/.env и НИКОГДА не хранится в коде.
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from .infer import format_result, predict

logging.basicConfig(level=logging.INFO)
dp = Dispatcher()

WELCOME = (
    "🍅 <b>LeafDoctor</b> — ИИ-диагностика болезней листьев томата.\n\n"
    "Пришлите фотографию листа, и я определю болезнь "
    "(10 классов, включая «здоровый лист»).\n\n"
    "Модель: ResNet18, точность на тесте ≈ 99%."
)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(WELCOME)


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer("Просто отправьте фото листа томата 🍃")


@dp.message(F.photo)
async def on_photo(message: Message, bot: Bot):
    await message.chat.do("typing")
    # берём самое большое разрешение
    photo = message.photo[-1]
    buf = await bot.download(photo)  # io.BytesIO
    data = buf.read()
    try:
        preds = predict(data, topk=3)
    except Exception as e:
        logging.exception("inference failed")
        await message.answer(f"Не получилось обработать изображение: {e}")
        return
    await message.answer(format_result(preds))


@dp.message(F.document)
async def on_document(message: Message, bot: Bot):
    doc = message.document
    if not (doc.mime_type or "").startswith("image/"):
        await message.answer("Пришлите изображение (JPEG/PNG) как фото.")
        return
    buf = await bot.download(doc)
    preds = predict(buf.read(), topk=3)
    await message.answer(format_result(preds))


@dp.message()
async def fallback(message: Message):
    await message.answer("Отправьте, пожалуйста, фотографию листа 🍃")


async def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN. Создайте файл .env и впишите токен "
            "(см. .env.example). Токен берётся у @BotFather."
        )
    # прогрев модели до старта polling, чтобы первый ответ не тормозил
    from .infer import _load
    _load()
    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    logging.info("LeafDoctor bot запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
