# LeafDoctor — техническая документация

Подробное руководство по проекту: что сделано, как устроен пайплайн, как обучать
модель самому и как ставить эксперименты с разными вариантами обучения.

> Краткое README с быстрым стартом — в [`README.md`](README.md). Этот документ — глубже.

---

## Содержание
1. [TL;DR — что сделано и результат](#1-tldr)
2. [Архитектура пайплайна](#2-архитектура-пайплайна)
3. [Поток данных от фото до класса](#3-поток-данных)
4. [Установка с нуля](#4-установка-с-нуля)
5. [Полный цикл: prepare → train → eval → infer](#5-полный-цикл)
6. [Как обучать самому: конфиг и все параметры](#6-как-обучать-самому)
7. [Как тестировать разные варианты обучения (эксперименты)](#7-эксперименты)
8. [Версионирование артефактов и манифест](#8-версионирование)
9. [Метрики и почему F1 macro, а не accuracy](#9-метрики)
10. [Воспроизводимость](#10-воспроизводимость)
11. [Инференс: CLI и FastAPI](#11-инференс)
12. [Troubleshooting](#12-troubleshooting)
13. [Текущие результаты v0.1.0](#13-результаты)

---

## 1. TL;DR

**Задача.** Классификация изображений (компьютерное зрение): фото листа томата →
класс (здоров / конкретная болезнь) + уверенность. Single-label, multi-class, 10 классов.

**Модель.** Свёрточная нейросеть (CNN), transfer learning на **ResNet18** из torchvision:
заменён последний `fc`-слой под 10 классов и дообучен. Поддержаны также EfficientNet-B0
и собственная маленькая CNN с нуля — для сравнения.

**Данные.** PlantVillage (HuggingFace), подмножество «томат»: **10 классов, 17 956 фото**.
Детерминированный стратифицированный split 80/10/10 → 14364 / 1796 / 1796.

**Результат (test, версия v0.1.0):**

| Метрика | Модель | Baseline (мажоритарный класс) |
|---|---|---|
| **F1 macro** (главная) | **0.9938** | 0.0460 |
| Accuracy | 0.9950 | 0.2984 |

Обучение на RTX 5070 Ti (8 эпох, AMP) — несколько минут.

---

## 2. Архитектура пайплайна

Пакет `src/leafdoctor/` — каждый модуль с одной зоной ответственности:

| Модуль | Назначение |
|---|---|
| `labels.py` | **Единый источник правды по классам.** Список классов = отсортированные имена папок; сохраняется в `classes.json` и копируется в каждый артефакт. И split, и train, и eval, и инференс читают индексы только отсюда → соответствие «класс ↔ индекс» никогда не разъезжается. |
| `config.py` | Загрузка `TrainConfig` из TOML + точечные оверрайды из CLI (`--set key=value` и именованные флаги). |
| `prepare_data.py` | Скачивание PlantVillage с HF, фильтр по культуре, материализация в ImageFolder, стратифицированный split → CSV. |
| `data.py` | Трансформы (аугментации для train / детерминированные для val-test), `Dataset` из split-CSV, фабрика `DataLoader`. |
| `model.py` | `build_model(name, num_classes)`: resnet18 / efficientnet_b0 / small_cnn. Заморозка/разморозка backbone. |
| `train.py` | Двухфазное обучение, AMP, scheduler, class weights, выбор лучшей эпохи по val F1 macro, сохранение версии. |
| `eval.py` | Метрики: F1 macro, accuracy, per-class P/R/F1, confusion matrix, baseline. Используется и из train, и как отдельный скрипт. |
| `version.py` | Сохранение артефакта + `manifest.json` (провенанс), резолвинг активной версии, загрузка весов. |
| `infer.py` | CLI-инференс: фото → класс + top-k. |
| `api.py` | (Опционально) FastAPI-сервис `/predict`. |
| `bot.py` | (Опционально) Telegram-бот на aiogram: фото → диагноз. |
| `display.py` | Человекочитаемые русские названия классов для бота/API. |

**Принцип:** конфиг — единственная точка настройки обучения; `labels.py` — единственная
точка истины по классам; `version.py` — единственная точка работы с артефактами.

---

## 3. Поток данных

```
                  HuggingFace: GVJahnavi/PlantVillage_dataset
                                   │  prepare_data.py
                                   ▼
        data/raw/<Tomato___class>/<sha>.jpg      (ImageFolder, 10 папок)
                                   │  стратиф. split 80/10/10 (seed)
                                   ▼
        data/splits/{train,val,test}.csv  +  data/splits/classes.json
                                   │  data.py (transforms + DataLoader)
                                   ▼
   train.py ──► ResNet18 (fc → 10) ──► фаза 1 (заморозка) → фаза 2 (fine-tune, cosine, AMP)
                                   │  выбор лучшей эпохи по val F1 macro
                                   ▼
        artifacts/plant-leaf-tomato/v0.1.0/
            model.safetensors · classes.json · metrics.json
            confusion_matrix.png · manifest.json
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                       ▼
        eval.py (метрики на test)            infer.py / api.py (инференс)
```

**Почему `classes.json` копируется в артефакт:** инференс не зависит от наличия `data/`.
Модель самодостаточна — содержит и веса, и порядок классов, и метаданные.

---

## 4. Установка с нуля

Требования: Python 3.12, GPU NVIDIA с драйвером под CUDA 12.x (проект тестировался на
RTX 5070 Ti, Blackwell).

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# 1) torch + torchvision из индекса PyTorch под CUDA 12.8 (cu128).
#    ВАЖНО: версии torch и torchvision должны совпадать по CUDA-суффиксу.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) тяжёлый стек для обучения (в файле НЕТ torch — он не перетирается)
pip install -r requirements-train.txt

# 3) сам пакет editable — чтобы работали команды `python -m leafdoctor.*`
pip install -e .
```

Проверка GPU:
```bash
python -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
# ожидается: 2.11.0+cu128 0.26.0+cu128 True
```

> **Почему cu128, а не cu126.** В системе уже стоял `torch 2.11.0+cu128`; torchvision
> обязан совпадать по CUDA-суффиксу, иначе ломается загрузка C++-расширений. Если ставите
> с нуля — берите любую согласованную пару с одного индекса.

Альтернатива без `pip install -e .`: запускать модули с `PYTHONPATH=src`, например
`PYTHONPATH=src python -m leafdoctor.train ...`.

---

## 5. Полный цикл

```bash
# 1) Данные: скачать томат + сделать split (идемпотентно — повтор не перекачивает)
python -m leafdoctor.prepare_data

# 2) Обучение по конфигу
python -m leafdoctor.train --config configs/train.toml

# 3) Оценка сохранённой версии на тесте
python -m leafdoctor.eval --version v0.1.0

# 4) Инференс на одном фото
python -m leafdoctor.infer data/raw/Tomato___healthy/<любой>.jpg
```

`train.py` уже в конце сам прогоняет оценку на test и рисует confusion matrix — отдельный
`eval.py` нужен, чтобы переоценить уже сохранённую версию (например, на другом split или
просто для воспроизведения метрик из весов).

---

## 6. Как обучать самому

### 6.1 Конфиг `configs/train.toml`

Все параметры обучения — в одном файле. Полный список полей:

| Поле | По умолч. | Что делает |
|---|---|---|
| `data_dir` | `data/raw` | ImageFolder-структура с изображениями. |
| `splits_dir` | `data/splits` | Где лежат `train/val/test.csv` и `classes.json`. |
| `out_dir` | `artifacts` | Корень версионированных артефактов. |
| `project_name` | `plant-leaf-tomato` | Имя проекта (подпапка в `out_dir`). |
| `version` | `v0.1.0` | Семвер-метка сохраняемого артефакта. **Меняйте для каждого нового эксперимента**, иначе перезапишете. |
| `model_name` | `resnet18` | `resnet18` \| `efficientnet_b0` \| `small_cnn`. |
| `img_size` | `224` | Размер входа сети. |
| `epochs` | `8` | Всего эпох (включая фазу заморозки). |
| `freeze_backbone_epochs` | `2` | Сколько первых эпох учить только голову (fc). `0` = сразу обучать всю сеть. |
| `batch_size` | `64` | Размер батча. Упёрлись в память GPU — уменьшайте. |
| `lr` | `5e-4` | LR фазы fine-tune (вся сеть). |
| `head_lr` | `1e-3` | LR фазы заморозки (только голова). |
| `weight_decay` | `1e-4` | L2-регуляризация. |
| `optimizer` | `adamw` | `adamw` \| `sgd`. |
| `scheduler` | `cosine` | `cosine` \| `step` \| `none` (действует в фазе 2). |
| `amp` | `true` | Mixed precision на GPU (быстрее, меньше памяти). |
| `class_weights` | `false` | Взвешивать CrossEntropy обратной частотой классов (борьба с дисбалансом). |
| `seed` | `42` | Фиксирует split-загрузку и инициализацию. |
| `num_workers` | `8` | Потоков DataLoader. |

### 6.2 Двухфазная стратегия обучения

ResNet18 предобучен на ImageNet. Стратегия transfer learning:

- **Фаза 1 (эпохи `0 .. freeze_backbone_epochs-1`):** backbone заморожен
  (`requires_grad=False`), учится только новая голова `fc` с `head_lr`. Голова быстро
  «настраивается» на новые классы, не разрушая полезные признаки backbone.
- **Фаза 2 (остальные эпохи):** backbone размораживается, дообучается вся сеть с меньшим
  `lr` и cosine-затуханием. Тонкая подстройка под домен (листья).

Лучшая по **val F1 macro** эпоха сохраняется как артефакт (а не последняя — защита от
переобучения на последних эпохах).

### 6.3 Точечные оверрайды без правки файла

```bash
# именованные шорткаты
python -m leafdoctor.train --epochs 12 --lr 0.0003 --batch_size 32

# универсальный --set key=value (можно несколько раз) — для любого поля конфига
python -m leafdoctor.train --set class_weights=true --set scheduler=step --set version=v0.2.0

# выбор устройства вручную
python -m leafdoctor.train --device cpu
```

Оверрайды из CLI имеют приоритет над TOML. Неизвестное поле → понятная ошибка.

---

## 7. Эксперименты

Каждый эксперимент — **своя `version`**, чтобы артефакты не перезаписывались и их можно
было сравнивать. Метрики каждого лежат в `artifacts/<project>/<version>/metrics.json`.

### 7.1 Сравнить архитектуры

```bash
python -m leafdoctor.train --model_name resnet18        --set version=v0.1.0
python -m leafdoctor.train --model_name efficientnet_b0 --set version=v0.2.0
python -m leafdoctor.train --model_name small_cnn       --set version=v0.3.0 --epochs 20
```
`small_cnn` обучается с нуля (без предобучения) — ему нужно больше эпох. Это честное
сравнение «transfer learning vs обучение с нуля».

### 7.2 Влияние дисбаланса классов

Классы томата несбалансированы (от 298 до 4285 фото в train). Проверьте class weights:
```bash
python -m leafdoctor.train --set class_weights=true --set version=v0.4.0-cw
```
Смотрите, поднялся ли **per-class recall у редких классов** (mosaic_virus, Leaf_Mold) —
именно там class weights дают эффект, даже если общий accuracy почти не меняется.

### 7.3 Влияние заморозки backbone

```bash
python -m leafdoctor.train --set freeze_backbone_epochs=0 --set version=v0.5.0-nofreeze
python -m leafdoctor.train --set freeze_backbone_epochs=4 --set version=v0.6.0-freeze4
```

### 7.4 LR / scheduler / оптимизатор

```bash
python -m leafdoctor.train --lr 0.001  --set version=v0.7.0-lr1e3
python -m leafdoctor.train --set scheduler=step --set version=v0.8.0-step
python -m leafdoctor.train --set optimizer=sgd --lr 0.01 --set version=v0.9.0-sgd
```

### 7.5 Скрипт сравнения версий

Быстро свести метрики всех обученных версий в таблицу:
```bash
python - <<'PY'
import json, glob, os
rows=[]
for m in sorted(glob.glob("artifacts/*/v*/manifest.json")):
    d=json.load(open(m))
    t=d["metrics"]["test"]
    rows.append((d["version"], d["model_name"],
                 round(t["f1_macro"],4), round(t["accuracy"],4),
                 d["config"].get("class_weights"), d["config"].get("epochs")))
print(f"{'version':<18}{'model':<16}{'f1_macro':<10}{'acc':<8}{'cw':<7}{'ep'}")
for r in rows:
    print(f"{r[0]:<18}{r[1]:<16}{r[2]:<10}{r[3]:<8}{str(r[4]):<7}{r[5]}")
PY
```

### 7.6 Эксперимент на другой культуре

`prepare_data` умеет любую культуру PlantVillage (Potato, Grape, Apple, …):
```bash
python -m leafdoctor.prepare_data --crop Potato --splits-dir data/splits_potato
python -m leafdoctor.train --set splits_dir=data/splits_potato \
    --set project_name=plant-leaf-potato --set version=v0.1.0
```

> Полезно при экспериментах фиксировать `seed` и менять только один фактор за раз —
> тогда разница в метриках объясняется именно этим фактором.

---

## 8. Версионирование

Каждый артефакт — `artifacts/<project_name>/<version>/`:

| Файл | Содержимое |
|---|---|
| `model.safetensors` | Веса (формат safetensors, безопасная сериализация). |
| `classes.json` | Список классов в порядке индексов — источник правды для инференса. |
| `metrics.json` | Метрики val (лучшая эпоха) + test + история по эпохам. |
| `confusion_matrix.png` | Матрица ошибок на тесте. |
| `manifest.json` | **Провенанс** (см. ниже). |

`manifest.json` содержит: `project`, `version`, `created_utc`, `git_commit`, `model_name`,
`num_classes`, `classes`, `class_to_idx`, полный снимок `config`, `data_sha256`
(sha256 от отсортированного списка путей данных — фиксирует состав датасета), `metrics`,
`framework` (версии torch/torchvision/CUDA).

**Зачем:** по манифесту можно точно сказать, на каких данных, каким кодом (git-commit) и
с какой конфигурацией получена модель — это и есть «воспроизводимость по-взрослому».

**Резолвинг активной версии** (для `eval`/`infer`/`api`):
```
явный --version  →  env LEAFDOCTOR_VERSION  →  самая свежая семвер-папка vX.Y.Z
```
```bash
python -m leafdoctor.infer leaf.jpg                      # самая свежая версия
python -m leafdoctor.infer leaf.jpg --version v0.2.0     # конкретная
LEAFDOCTOR_VERSION=v0.2.0 python -m leafdoctor.infer leaf.jpg
```

---

## 9. Метрики

**Главная метрика — F1 macro, а не accuracy.** При дисбалансе классов accuracy
вводит в заблуждение: модель, всегда предсказывающая мажоритарный класс
(`Tomato_Yellow_Leaf_Curl_Virus`, ~30% выборки), даёт **accuracy 0.298**, но
**F1 macro лишь 0.046** — потому что F1 macro усредняется по классам и штрафует за
игнорирование редких. Поэтому именно его мы максимизируем и по нему выбираем лучшую эпоху.

`eval.py` печатает и сохраняет:
- **F1 macro** (среднее F1 по классам, без учёта их размера);
- **accuracy** (доля верных);
- **per-class precision / recall / F1 / support** (`sklearn.classification_report`);
- **baseline** мажоритарного класса — для честного контекста «насколько модель лучше тупого угадывания»;
- **confusion matrix** (PNG) — видно, какие классы путаются между собой.

---

## 10. Воспроизводимость

Что гарантирует повторяемость результата:
- **`seed`** фиксирует стратифицированный split (`prepare_data`) и инициализацию весов головы (`train`).
- **Детерминированный split** записан в CSV — один и тот же на всех запусках.
- **`classes.json`** — единый порядок классов на всех этапах.
- **`manifest.json`** — git-commit + `data_sha256` + снимок конфига: точный «слепок» эксперимента.

> Полная битовая детерминированность GPU-операций здесь не форсируется (это замедляет
> обучение и редко нужно в учебном проекте). Метрики воспроизводятся с точностью до
> небольшого шума; сам split и состав данных — строго детерминированы.

---

## 11. Инференс

### CLI
```bash
python -m leafdoctor.infer path/to/leaf.jpg                 # человекочитаемый вывод
python -m leafdoctor.infer path/to/leaf.jpg --topk 5 --json # JSON
python -m leafdoctor.infer path/to/leaf.jpg --version v0.2.0
```
Вывод: класс с максимальной уверенностью + top-k вероятностей (softmax). Картинка
проходит те же val/test-трансформы (resize 256 → center-crop 224 → ImageNet-нормализация).

### FastAPI (опционально)
```bash
pip install -r requirements.txt          # лёгкий стек (fastapi, uvicorn, multipart, pillow, safetensors)
uvicorn leafdoctor.api:app --port 8000
```
```bash
curl http://localhost:8000/health
curl -F "file=@leaf.jpg" "http://localhost:8000/predict?topk=3"
```
Версия модели для сервиса задаётся через env: `LEAFDOCTOR_VERSION`, `LEAFDOCTOR_PROJECT`,
`LEAFDOCTOR_OUT_DIR`.

### Конфигурация через `.env`

Токен бота и выбор версии модели читаются из переменных окружения, которые удобно
держать в файле **`.env`** в корне проекта (загружается автоматически через
`python-dotenv`, см. `settings.py`). Файл `.env` **не коммитится** (в `.gitignore`),
в репозитории лежит только шаблон **`.env.example`**.

```bash
cp .env.example .env      # и впишите свои значения
```
```ini
# .env
TELEGRAM_BOT_TOKEN=123456789:AA-ваш-токен
LEAFDOCTOR_VERSION=v0.1.0
# LEAFDOCTOR_PROJECT=plant-leaf-tomato
# LEAFDOCTOR_OUT_DIR=artifacts
```
> Реальные переменные окружения имеют **приоритет** над `.env` (`override=False`),
> поэтому в продакшене можно не держать `.env`, а задавать всё снаружи (docker/systemd).
> `.env` подхватывают и бот, и FastAPI-сервис.

### Telegram-бот (aiogram)
```bash
pip install -r requirements.txt              # включает aiogram и python-dotenv
cp .env.example .env                          # впишите TELEGRAM_BOT_TOKEN
python -m leafdoctor.bot
# либо без .env:  export TELEGRAM_BOT_TOKEN="123456:ABC..."  &&  python -m leafdoctor.bot
```
Как устроен:
- Модель загружается **один раз при старте** (`Predictor`) и держится в памяти; версия
  резолвится так же, как у CLI/API (env `LEAFDOCTOR_VERSION` → самая свежая).
- На фото/изображение-файл бот скачивает байты, запускает инференс **в отдельном потоке**
  (`asyncio.to_thread`), чтобы не блокировать event loop, и отвечает классом + уверенностью
  + top-3. Названия классов — дружелюбные русские (`display.py`), с эмодзи по уровню
  уверенности (✅ ≥80%, 🤔 ≥50%, ⚠️ ниже).
- Команды: `/start` (приветствие), `/help` (версия модели + список распознаваемых классов).

Запуск с конкретной версией модели:
```bash
LEAFDOCTOR_VERSION=v0.2.0 TELEGRAM_BOT_TOKEN=... python -m leafdoctor.bot
```

---

## 12. Troubleshooting

| Симптом | Причина / решение |
|---|---|
| `torch.cuda.is_available() == False` | torch установлен в CPU-версии. Переустановите torch+torchvision с `--index-url .../cu128`. |
| `undefined symbol` / ошибка при `import torchvision` | torch и torchvision из разных CUDA-индексов. Поставьте обе из одного индекса. |
| `CUDA out of memory` | Уменьшите `batch_size` (например 32 или 16), убедитесь, что `amp=true`. |
| `BuilderConfig 'color' not found` | Это про старое зеркало `mohanty/PlantVillage` (поломанная карточка). Проект использует `GVJahnavi/PlantVillage_dataset` — обновите код, если меняли. |
| `Нет сохранённых версий` при инференсе | Сначала обучите модель (`train.py`) — артефакт создаётся им. |
| Медленный DataLoader | Увеличьте `num_workers`; на Windows/WSL иногда помогает уменьшить. |
| Хочу заново скачать данные | Удалите `data/raw/.materialized` (или весь `data/raw/`) и запустите `prepare_data`. |
| `Не задан TELEGRAM_BOT_TOKEN` | Создайте бота у [@BotFather](https://t.me/BotFather), скопируйте токен и задайте `export TELEGRAM_BOT_TOKEN=...`. |
| Бот не отвечает | Проверьте, что процесс `leafdoctor.bot` запущен и нет другого polling с тем же токеном (Telegram допускает только один). |

---

## 13. Результаты v0.1.0

**Конфигурация:** ResNet18, 8 эпох (2 заморозки + 6 fine-tune), batch 64, AdamW,
cosine LR, AMP, без class weights, seed 42.

**Данные:** 10 классов томата, 17 956 фото, split 14364 / 1796 / 1796.

**Тест:**

| Метрика | Значение |
|---|---|
| **F1 macro** | **0.9938** |
| Accuracy | 0.9950 |
| Baseline F1 macro (мажоритарный класс) | 0.0460 |
| Baseline accuracy | 0.2984 |

**Per-class F1 (test):** все ≥ 0.975.

| Класс | precision | recall | f1 | support |
|---|---|---|---|---|
| Bacterial_spot | 1.000 | 1.000 | 1.000 | 213 |
| Early_blight | 0.970 | 0.980 | 0.975 | 100 |
| Late_blight | 0.995 | 0.979 | 0.987 | 190 |
| Leaf_Mold | 1.000 | 1.000 | 1.000 | 76 |
| Septoria_leaf_spot | 0.978 | 0.994 | 0.986 | 177 |
| Spider_mites | 1.000 | 1.000 | 1.000 | 168 |
| Target_Spot | 1.000 | 0.986 | 0.993 | 140 |
| Yellow_Leaf_Curl_Virus | 1.000 | 1.000 | 1.000 | 536 |
| mosaic_virus | 1.000 | 1.000 | 1.000 | 38 |
| healthy | 0.994 | 1.000 | 0.997 | 158 |

Всего ошибок: **9 из 1796** (≈0.5%). Кривая обучения (val F1 macro по эпохам):
0.827 → 0.852 → 0.963 → 0.977 → 0.986 → 0.985 → 0.994 → **0.994**.

Артефакт: `artifacts/plant-leaf-tomato/v0.1.0/` (git-commit и полный конфиг — в `manifest.json`).
