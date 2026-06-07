# Прогон старых run1-адаптеров на РУССКОМ BIRD

Этот сниппет проверяет гипотезу языкового рассинхрона из run1:
старые модели обучались на **русской** синтетике → должны лучше отвечать
на **русских** вопросах, чем на английских (на которых получили 6-22%).

## Подготовка
1. Загрузи `data/bird_large_ru.json` на Drive → `MyDrive/text2sql/bird_large_ru.json`
2. Открой Colab (A100), новый ноутбук или текущий eval-ноутбук.

## Сниппет (одна большая ячейка)

```python
# Если ноутбук свежий — сначала install + перезапуск сеанса:
# !pip install -q --upgrade unsloth unsloth_zoo

from unsloth import FastLanguageModel
import torch, json, gc, os, glob
from peft import PeftModel
from google.colab import drive
drive.mount('/content/drive')

BASE_NEW = '/content/drive/MyDrive/text2sql'              # новые tasks/данные
BASE_OLD = '/content/drive/MyDrive/text2sql_finetune'     # старые run1-адаптеры
MAX_SEQ_LEN = 12288
BASE_MODEL = 'unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit'

# Адаптеры из run1 (если они ещё лежат на Drive)
OLD_EXPERIMENTS = ['exp_gpt41', 'exp_gpt41_mini', 'exp_gpt41_nano',
                   'exp_gpt4o_mini', 'exp_codex_mini']

# Русский BIRD
bird = json.load(open(f'{BASE_NEW}/bird_large_ru.json'))
questions = [q for q in bird if q['db_id'] == 'card_games']
print(f'Вопросов (RU): {len(questions)}')

# Системный промпт из любого train.jsonl run1
def find_system_prompt():
    for p in (glob.glob(f'{BASE_OLD}/**/train.jsonl', recursive=True)
              + glob.glob(f'{BASE_NEW}/**/train.jsonl', recursive=True)):
        try:
            return json.loads(open(p).readline())['messages'][0]['content']
        except Exception:
            continue
    raise FileNotFoundError('train.jsonl не найден')

SYSTEM_PROMPT = find_system_prompt()

SQL_PROMPT_TEMPLATE = '''
Преобразуй следующий запрос в SQL:
Запрос: {user_query}

Требования:
1. Только SQL, диалект {sql_dialect} - совместимый синтаксис
2. Обязательные комментарии перед запросом
3. Четкое соответствие запросу
4. Используй LIKE для имён собственных.
5. DISTINCT только сразу после SELECT.
НЕ ОФОРМЛЯЙ НИКАК СВОЙ ОТВЕТ. ТВОЙ ОТВЕТ - ТОЛЬКО ЗАПРОС, НИЧЕГО БОЛЕЕ!

SQL запрос:'''

USE_EVIDENCE = True

def run_inference(model, tok, tag, out_base):
    FastLanguageModel.for_inference(model)
    preds = []
    for i, q in enumerate(questions):
        question = q['question']
        if USE_EVIDENCE and q.get('evidence'):
            question = f"{q['question']} (подсказка: {q['evidence']})"
        user_msg = SQL_PROMPT_TEMPLATE.format(
            user_query=question, sql_dialect='PostgreSQL')
        msgs = [{'role':'system','content':SYSTEM_PROMPT},
                {'role':'user','content':user_msg}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors='pt', truncation=True,
                  max_length=MAX_SEQ_LEN).to('cuda')
        out = model.generate(
            input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
            max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
        sql = tok.decode(out[0][enc['input_ids'].shape[1]:],
                         skip_special_tokens=True).strip()
        sql = sql.replace('```sql','').replace('```','').strip()
        preds.append({'question_id': str(q['question_id']),
                      'question': q.get('question_en', q['question']),  # для diff с прошлым прогоном
                      'gold_sql': q['SQL'],
                      'predicted_sql': sql,
                      'difficulty': q.get('difficulty','unknown')})
        if i < 2: print(f'  Q: {q["question"][:45]}\n  A: {sql[:70]}')
    os.makedirs(f'{out_base}/{tag}_ru', exist_ok=True)
    json.dump(preds, open(f'{out_base}/{tag}_ru/eval_predictions.json','w'),
              ensure_ascii=False, indent=2)
    print(f'✓ {len(preds)} → {tag}_ru/eval_predictions.json')

# BASELINE на русских вопросах
print(f'\n{"="*55}\nbaseline_ru\n{"="*55}')
model, tok = FastLanguageModel.from_pretrained(
    BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
run_inference(model, tok, 'baseline', BASE_NEW)
del model, tok; torch.cuda.empty_cache(); gc.collect()

# Старые run1-адаптеры на русских вопросах
for exp in OLD_EXPERIMENTS:
    adapter = f'{BASE_OLD}/{exp}/adapter'
    if not os.path.exists(adapter):
        print(f'⏭  {exp}: нет {adapter}'); continue
    print(f'\n{"="*55}\n{exp}_ru\n{"="*55}')
    try:
        model, tok = FastLanguageModel.from_pretrained(
            BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
        model = PeftModel.from_pretrained(model, adapter)
        run_inference(model, tok, exp, BASE_NEW)
    except Exception as e:
        print(f'✗ {exp} УПАЛ: {e}')
    finally:
        torch.cuda.empty_cache(); gc.collect()
```

## После прогона

Скачай `eval_predictions.json` из каждой папки `<exp>_ru/` локально в `data/exp/<exp>_ru/`,
потом запусти финальное сравнение. Сравни с run1 на английском —
если на русском accuracy выше, языковая гипотеза подтверждена.

```bash
uv run --env-file .env python scripts/compare_generators.py
```

(Чтобы скрипт увидел `<exp>_ru/`, добавь их в реестр или в EXPERIMENTS словаре скрипта.)
