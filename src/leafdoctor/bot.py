# src/leafdoctor/bot.py
import asyncio
import logging
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart, Command

from leafdoctor.infer import predict_image
from leafdoctor.labels import get_label, is_healthy

logger = logging.getLogger(__name__)


def build_bot(token: str) -> tuple[Bot, Dispatcher]:
    bot = Bot(token=token)
    dp  = Dispatcher()

    @dp.message(CommandStart())
    async def cmd_start(message: Message):
        await message.answer(
            "🌿 *LeafDoctor* — диагностика болезней томата\n\n"
            "Отправь фото листа, и я определю болезнь.\n"
            "Поддерживаю 9 болезней + здоровый лист.",
            parse_mode="Markdown"
        )

    @dp.message(Command("help"))
    async def cmd_help(message: Message):
        await message.answer(
            "📋 *Команды:*\n"
            "/start — начало работы\n"
            "/help  — эта справка\n\n"
            "📸 Просто отправь фото листа томата!",
            parse_mode="Markdown"
        )

    @dp.message(F.photo)
    async def handle_photo(message: Message):
        await message.answer("🔍 Анализирую фото...")

        # Скачиваем фото
        photo = message.photo[-1]  # лучшее качество
        file  = await bot.get_file(photo.file_id)
        data  = await bot.download_file(file.file_path)
        img_bytes = data.read()

        # Предсказание
        results = predict_image(img_bytes, topk=3)

        # Формируем ответ
        top      = results[0]
        idx      = top["class_idx"]
        label    = get_label(idx)
        conf     = top["confidence"] * 100
        healthy  = is_healthy(idx)

        if healthy:
            header = f"✅ *{label}* ({conf:.1f}%)\n\nПризнаков болезни не обнаружено."
        else:
            header = f"⚠️ *{label}* ({conf:.1f}%)\n\nОбнаружена болезнь!"

        lines = [header, "\n📊 *Топ-3 варианта:*"]
        for r in results:
            lines.append(f"  • {get_label(r['class_idx'])}: {r['confidence']*100:.1f}%")

        await message.answer("\n".join(lines), parse_mode="Markdown")

    @dp.message(F.document)
    async def handle_doc(message: Message):
        await message.answer("📎 Пожалуйста, отправь именно *фото*, не файл.", parse_mode="Markdown")

    @dp.message()
    async def handle_text(message: Message):
        await message.answer("📸 Отправь фото листа томата для диагностики.")

    return bot, dp


async def run_bot(token: str):
    logging.basicConfig(level=logging.INFO)
    bot, dp = build_bot(token)
    logger.info("Бот запущен...")
    await dp.start_polling(bot)