# Разработка

## Windows: текущая рабочая машина

Из корня репозитория:

```powershell
./scripts/setup.ps1
./.venv/Scripts/python.exe -m wan_dit_lab.cli doctor
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check .
./.venv/Scripts/python.exe -m ruff format --check .
./.venv/Scripts/python.exe -m wan_dit_lab.cli bench-smoke
```

В VS Code открой именно этот репозиторий как папку и выбери `.venv/Scripts/python.exe`.

Окружение создано с `--system-site-packages`, чтобы использовать уже работающие CUDA/PyTorch/Triton без повторного скачивания. Это удобная локальная среда, но не полностью изолированная и не lock-файл. Изменения глобальных пакетов могут на неё влиять. Перед бенчмарком сохраняй вывод `doctor` вместе с результатом.

Проверенная исходная среда: Python 3.14.3, torch 2.10.0+cu128, triton-windows 3.7.1.post27, safetensors 0.8.0, RTX 4060 Laptop GPU. Эти версии фиксируют локальную проверку, а не обещание совместимости с оригинальным pipeline Wan. Его зависимости проверяются отдельно на этапе интеграции. На Linux пакет Triton отличается; Windows-настройки не переносить механически.

`bench-smoke` — небольшой синтетический FP16 GEMM для проверки измерительного стенда. Он сообщает GPU-время через CUDA events, wall-time с синхронизацией и пиковую выделенную память. Это не бенчмарк Wan и не доказательство ускорения модели. Код выполняется после прогрева, случайный seed зафиксирован. Для сравнения реализаций потребуются реальные одинаковые входы и отдельная проверка точности.

## Учебный рабочий процесс

Сначала описать математику и формы, затем реализовать компонент, сравнить с оригиналом, после этого измерять. Не заполнять всю архитектуру заранее заглушками, выдаваемыми за работающую модель.

Веса хранить в `checkpoints/`, дампы и замеры — в `artifacts/`, видео — в `outputs/`; эти каталоги исключены из Git. Не коммитить токены Hugging Face и `.env`.

Следующий этап: зафиксировать конкретный checkpoint и upstream revision в `configs/`, затем реализовать один блок. Пока модель не реализована, тесты проверяют только доступность вычислительной среды.

## Веса и оригинальный код

Ревизии теперь закреплены в `configs/upstream.json`. Полная загрузка/возобновление и проверка:

```powershell
./.venv/Scripts/python.exe scripts/conversion/prepare_weights.py
```

Комплект включает DiT, VAE, UMT5 encoder и tokenizer, около 17.6 ГБ. После загрузки создаётся `artifacts/weights-manifest.json` с размерами, SHA256 и формами тензоров DiT. Для LFS-файлов SHA256 сверяется с официальными метаданными Hub. Файлы `.pth` пока не исполняются и не загружаются в GPU: загрузку их state_dict следует делать через `torch.load(..., weights_only=True, map_location="cpu")`.

Оригинальный код находится в `reference/Wan2.1`, на закреплённом commit. Это отдельный reference checkout; не импортируем его пакет автоматически и не устанавливаем все его зависимости в учебную среду до соответствующего этапа. Для восстановления checkout на другой машине используй code_url и code_revision из `configs/upstream.json`.

Начать работу с весами собственного блока:

```python
from wan_dit_lab.models.checkpoint import model_config, load_block_weights

config = model_config()
weights = load_block_weights(0)  # CPU, только первый блок
print(config)
print({name: tuple(value.shape) for name, value in weights.items()})
# После реализации совместимого блока:
# my_block.load_state_dict(weights, strict=True)
```

Полный запуск генерации пока не реализован. Подготовленные файлы и загрузчик дают основу для написания своего DiT, а не обещание уже работающего pipeline.
