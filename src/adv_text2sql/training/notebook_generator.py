"""
notebook_generator — генерирует Colab-ноутбук для FT под конкретную БД.

Берёт `lora_finetune_2.ipynb` как референс и пересобирает его с подстановкой
параметров: пути к Drive, db_id, base_model. Это не template-engine типа Jinja
для ipynb — мы целиком формируем структуру ячеек программно (так проще
поддерживать чем jinja-templated json).

Использование:
    notebook_path = generate_finetune_notebook(
        db_id="card_games",
        base_model="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        drive_data_dir="text2sql/card_games",
    )
"""
from __future__ import annotations

import json
from pathlib import Path


def generate_finetune_notebook(
    db_id: str,
    base_model: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    drive_data_dir: str = "text2sql",
    output_dir: str | Path = "notebooks",
    r: int = 16,
    lora_alpha: int = 32,
    lr: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_seq_len: int = 4096,
) -> Path:
    """Создаёт notebooks/auto_train_<db_id>.ipynb с заполненными параметрами.

    Args:
        db_id: имя БД, используется в путях к train.jsonl/val.jsonl на Drive.
        base_model: HF-ID базовой модели (см. lora_finetune_2.ipynb).
        drive_data_dir: папка в MyDrive где лежат train/val.jsonl.
        output_dir: куда писать .ipynb.
        r, lora_alpha, lr, ... — гиперпараметры FT.
    """
    cells = [
        _md_cell(f"# LoRA fine-tune для `{db_id}`\n\n"
                 f"Автосгенерированный ноутбук. Параметры:\n"
                 f"- base_model: `{base_model}`\n- r={r}, lora_alpha={lora_alpha}, "
                 f"lr={lr}, epochs={num_epochs}, batch={batch_size}×{grad_accum}\n\n"
                 f"**Запуск**: Runtime → Run all. Время: ~1 час на A100."),
        _code_cell(
            "# 0. Install unsloth\n"
            "!pip install -q --upgrade pip\n"
            "!pip install -q --upgrade unsloth unsloth_zoo\n"
            "print('✓ install done — Runtime → Restart session')"
        ),
        _md_cell("## 1. Подключение Drive и загрузка датасета"),
        _code_cell(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            f"!cp /content/drive/MyDrive/{drive_data_dir}/{db_id}/*.jsonl /content/\n"
            "!ls -la /content/*.jsonl"
        ),
        _md_cell("## 2. Загрузка базовой модели в 4-bit"),
        _code_cell(
            "from unsloth import FastLanguageModel\n"
            f"MAX_SEQ_LEN = {max_seq_len}\n\n"
            "model, tokenizer = FastLanguageModel.from_pretrained(\n"
            f"    model_name='{base_model}',\n"
            "    max_seq_length=MAX_SEQ_LEN,\n"
            "    dtype=None,\n"
            "    load_in_4bit=True,\n"
            ")\n"
            f"print('✓ {base_model} loaded')"
        ),
        _md_cell("## 3. LoRA wrap"),
        _code_cell(
            "model = FastLanguageModel.get_peft_model(\n"
            "    model,\n"
            f"    r={r},\n"
            f"    lora_alpha={lora_alpha},\n"
            "    target_modules=['q_proj','k_proj','v_proj','o_proj',\n"
            "                    'gate_proj','up_proj','down_proj'],\n"
            "    lora_dropout=0.05, bias='none',\n"
            "    use_gradient_checkpointing='unsloth', random_state=42,\n"
            ")\n"
            "model.print_trainable_parameters()"
        ),
        _md_cell("## 4. Chat-format dataset"),
        _code_cell(
            "from datasets import load_dataset\n"
            "raw = load_dataset('json', data_files={\n"
            "    'train': 'train.jsonl', 'val': 'val.jsonl'})\n\n"
            "def fmt(ex):\n"
            "    text = tokenizer.apply_chat_template(\n"
            "        ex['messages'], tokenize=False, add_generation_prompt=False)\n"
            "    return {'text': text}\n"
            "ds = raw.map(fmt, remove_columns=['messages', '_meta'])\n"
            "print('train:', len(ds['train']), 'val:', len(ds['val']))"
        ),
        _md_cell("## 5. Trainer"),
        _code_cell(
            "from trl import SFTTrainer, SFTConfig\n"
            f"CKPT_DIR = '/content/drive/MyDrive/{drive_data_dir}/{db_id}/checkpoints'\n\n"
            "trainer = SFTTrainer(\n"
            "    model=model, tokenizer=tokenizer,\n"
            "    train_dataset=ds['train'], eval_dataset=ds['val'],\n"
            "    args=SFTConfig(\n"
            "        output_dir=CKPT_DIR,\n"
            f"        per_device_train_batch_size={batch_size},\n"
            f"        gradient_accumulation_steps={grad_accum},\n"
            f"        num_train_epochs={num_epochs},\n"
            f"        learning_rate={lr},\n"
            "        warmup_ratio=0.05, bf16=True,\n"
            "        logging_steps=10, save_strategy='steps', save_steps=100,\n"
            "        save_total_limit=5, eval_strategy='steps', eval_steps=100,\n"
            f"        max_seq_length={max_seq_len}, dataset_text_field='text',\n"
            "        packing=True, report_to='none', seed=42,\n"
            "    ),\n"
            ")\n"
            "trainer.train()"
        ),
        _md_cell("## 6. Save adapter to Drive"),
        _code_cell(
            f"ADAPTER_DIR = '/content/drive/MyDrive/{drive_data_dir}/{db_id}/adapter'\n"
            "model.save_pretrained(ADAPTER_DIR)\n"
            "tokenizer.save_pretrained(ADAPTER_DIR)\n"
            "print(f'✓ adapter at {ADAPTER_DIR}')"
        ),
        _md_cell("## 7. (Опционально) GGUF-экспорт для LM Studio fallback"),
        _code_cell(
            f"model.save_pretrained_gguf('qwen25-coder-{db_id}', tokenizer,\n"
            "                            quantization_method='q4_k_m')\n"
            "import glob, shutil\n"
            "from pathlib import Path\n"
            f"DRIVE = '/content/drive/MyDrive/{drive_data_dir}/{db_id}'\n"
            "for f in glob.glob('/content/**/*.gguf', recursive=True):\n"
            "    if '/content/drive/' in f: continue\n"
            "    dst = Path(DRIVE) / Path(f).name\n"
            "    if not dst.exists(): shutil.copy(f, dst)\n"
            "    print(f'✓ saved {dst.name}')"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"auto_train_{db_id}.ipynb"
    out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Cell helpers
# ─────────────────────────────────────────────────────────────────────────────


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}
