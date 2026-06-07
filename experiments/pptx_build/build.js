/**
 * AutoText2SQL presentation builder.
 * Theme: Midnight Executive (navy/ice blue/white).
 * Style: научный, без эмодзи, чёткие таблицы и нумерация.
 *
 * Run: node build.js
 * Output: ../AutoText2SQL.pptx
 */
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10" x 5.625"
pres.author = "AutoText2SQL";
pres.title = "AutoText2SQL — Autonomous Text2SQL pipeline";

// ─── Palette: Midnight Executive ─────────────────────────────────────────
const NAVY = "1E2761";
const NAVY_DARK = "131A47";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const ACCENT = "F96167"; // coral для главных чисел и акцентов
const MUTED = "64748B";
const LIGHT_BG = "F7F8FC";
const BORDER = "D6DBEC";

// ─── Helpers ─────────────────────────────────────────────────────────────
function addHeader(slide, title, subtitle) {
  // верхняя navy-полоса
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.95,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  slide.addText(title, {
    x: 0.5, y: 0.15, w: 9, h: 0.55,
    fontSize: 24, fontFace: "Calibri", bold: true,
    color: WHITE, margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 0.62, w: 9, h: 0.3,
      fontSize: 12, fontFace: "Calibri",
      color: ICE, margin: 0,
    });
  }
}

function addFooter(slide, page, total) {
  slide.addText(`AutoText2SQL  ·  DeepPavlov`, {
    x: 0.4, y: 5.32, w: 5, h: 0.25,
    fontSize: 9, color: MUTED, fontFace: "Calibri",
  });
  slide.addText(`${page} / ${total}`, {
    x: 8.8, y: 5.32, w: 0.8, h: 0.25,
    fontSize: 9, color: MUTED, fontFace: "Calibri", align: "right",
  });
}

function bg(slide, color) { slide.background = { color: color || WHITE }; }

const TOTAL = 14;

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 1 — Титульный (dark)
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, NAVY);

  // декоративная вертикальная акцент-полоса слева
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });

  s.addText("AutoText2SQL", {
    x: 0.6, y: 1.3, w: 9, h: 0.9,
    fontSize: 54, fontFace: "Georgia", bold: true, color: WHITE, margin: 0,
  });
  s.addText("Автоматизированное создание Text2SQL-инструментов под конкретную БД", {
    x: 0.6, y: 2.3, w: 9, h: 0.7,
    fontSize: 20, fontFace: "Calibri", color: ICE, margin: 0,
  });
  s.addText("От разведки схемы до production-сервиса в одном CLI", {
    x: 0.6, y: 2.95, w: 9, h: 0.45,
    fontSize: 14, fontFace: "Calibri", italic: true, color: ICE, margin: 0,
  });

  // нижний блок с лабораторией
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 4.5, w: 4, h: 0.05, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });
  s.addText("DeepPavlov · 2026", {
    x: 0.6, y: 4.65, w: 5, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: WHITE, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 2 — Проблема
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Проблема", "Универсальные Text2SQL не знают вашу БД");

  // Левая колонка: проблема
  s.addText("Universal Text2SQL", {
    x: 0.5, y: 1.2, w: 4.5, h: 0.4,
    fontSize: 18, fontFace: "Calibri", bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "Обучены на BIRD / Spider / WikiSQL", options: { bullet: true, breakLine: true } },
    { text: "Не знают конкретную пользовательскую БД", options: { bullet: true, breakLine: true } },
    { text: "Галлюцинируют имена колонок", options: { bullet: true, breakLine: true } },
    { text: "Неправильные JOIN-связи", options: { bullet: true, breakLine: true } },
    { text: "Игнорируют domain-специфику (форматы дат, бизнес-правила)", options: { bullet: true } },
  ], {
    x: 0.5, y: 1.65, w: 4.5, h: 3,
    fontSize: 13, fontFace: "Calibri", color: "333333", paraSpaceAfter: 4,
  });

  // Правая колонка: альтернатива
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.2, w: 4.3, h: 3.5,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addText("Альтернатива: FT под каждую БД", {
    x: 5.4, y: 1.35, w: 4, h: 0.4,
    fontSize: 16, fontFace: "Calibri", bold: true, color: NAVY, margin: 0,
  });
  s.addText([
    { text: "Дни-недели ручной разметки", options: { bullet: true, breakLine: true } },
    { text: "Требуются экспертные SQL-разработчики", options: { bullet: true, breakLine: true } },
    { text: "Под каждую новую БД — заново", options: { bullet: true, breakLine: true } },
    { text: "Не масштабируется на 10+ БД", options: { bullet: true } },
  ], {
    x: 5.4, y: 1.85, w: 4, h: 2.5,
    fontSize: 13, fontFace: "Calibri", color: "333333", paraSpaceAfter: 4,
  });

  // Главный месседж
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.85, w: 9, h: 0.4,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Нужен способ автоматически адаптировать Text2SQL под произвольную БД", {
    x: 0.5, y: 4.85, w: 9, h: 0.4,
    fontSize: 13, fontFace: "Calibri", bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });

  addFooter(s, 2, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 3 — Идея: 5-стадийный пайплайн
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Решение", "Автономный пайплайн: 5 стадий, ни одного ручного шага");

  const stages = [
    { name: "PROFILE",  sub: "read DB metadata" },
    { name: "GENERATE", sub: "LLM + templates" },
    { name: "VALIDATE", sub: "exec filter" },
    { name: "FINE-TUNE", sub: "LoRA / Qwen-7B" },
    { name: "SERVE",    sub: "vLLM / MCP" },
  ];

  const startX = 0.4;
  const boxW = 1.7, boxH = 1.4, gap = 0.18;

  stages.forEach((st, i) => {
    const x = startX + i * (boxW + gap);
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.6, w: boxW, h: boxH,
      fill: { color: NAVY }, line: { color: NAVY, width: 0 },
    });
    // Number badge
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.1, y: 1.7, w: 0.35, h: 0.35,
      fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
    });
    s.addText(String(i + 1), {
      x: x + 0.1, y: 1.7, w: 0.35, h: 0.35,
      fontSize: 14, bold: true, color: WHITE,
      align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
    });
    // Name
    s.addText(st.name, {
      x, y: 2.15, w: boxW, h: 0.4,
      fontSize: 14, fontFace: "Calibri", bold: true, color: WHITE,
      align: "center", valign: "middle", margin: 0,
    });
    // Subtitle
    s.addText(st.sub, {
      x, y: 2.55, w: boxW, h: 0.4,
      fontSize: 10, fontFace: "Calibri", color: ICE,
      align: "center", valign: "middle", italic: true, margin: 0,
    });

    // Arrow между блоками
    if (i < stages.length - 1) {
      s.addShape(pres.shapes.RIGHT_TRIANGLE, {
        x: x + boxW + 0.02, y: 2.2, w: 0.14, h: 0.2,
        fill: { color: NAVY }, line: { color: NAVY, width: 0 }, rotate: 90,
      });
    }
  });

  // Входы / выходы под пайплайном
  s.addText("Вход", {
    x: 0.5, y: 3.5, w: 1.5, h: 0.35,
    fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  s.addText("postgresql://...", {
    x: 0.5, y: 3.8, w: 4, h: 0.35,
    fontSize: 12, color: "333333", fontFace: "Consolas", margin: 0,
  });

  s.addText("Выход", {
    x: 5.5, y: 3.5, w: 1.5, h: 0.35,
    fontSize: 12, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  s.addText("LoRA-адаптер + MCP-сервис", {
    x: 5.5, y: 3.8, w: 4, h: 0.35,
    fontSize: 12, color: "333333", fontFace: "Consolas", margin: 0,
  });

  // Time stat
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9, h: 0.7,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addText([
    { text: "~30 минут", options: { fontSize: 22, bold: true, color: ACCENT } },
    { text: "  от URL базы до работающего сервиса. Идемпотентно, артефакты на диске, независимый дебаг.", options: { fontSize: 12, color: "333333" } },
  ], {
    x: 0.7, y: 4.4, w: 8.6, h: 0.7,
    fontFace: "Calibri", valign: "middle", margin: 0,
  });

  addFooter(s, 3, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 4 — Архитектура
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Архитектура", "6 модулей, ~5000 строк, 35 unit-тестов");

  const modules = [
    { name: "profiler/",  desc: "DBConnector, SchemaExtractor, StatsCollector, SampleCollector — вычитка метаданных PG" },
    { name: "synth/",     desc: "LLMSyntheticGenerator + TemplateSyntheticGenerator (25 шаблонов) — генерация Q-SQL" },
    { name: "synth/Validator", desc: "Parse (sqlglot) + Whitelist (SELECT) + Execute на реальной БД + опц. LLM-judge" },
    { name: "training/",  desc: "DatasetBuilder (chat-format JSONL) + notebook_generator (Colab template)" },
    { name: "serve/",     desc: "Dockerfile.vllm + LoRA hot-swap + Text2SQLClient SDK + QueryLogger" },
    { name: "heal/",      desc: "HealCollector (failed.jsonl → дедуп) + HealSQLGenerator (GPT-4 корректор)" },
  ];

  modules.forEach((m, i) => {
    const y = 1.15 + i * 0.65;
    // Module card
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 9, h: 0.55,
      fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
    });
    // Left accent
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y, w: 0.08, h: 0.55,
      fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
    });
    // Module name
    s.addText(m.name, {
      x: 0.75, y, w: 2.1, h: 0.55,
      fontSize: 13, fontFace: "Consolas", bold: true, color: NAVY,
      valign: "middle", margin: 0,
    });
    // Description
    s.addText(m.desc, {
      x: 2.9, y, w: 6.5, h: 0.55,
      fontSize: 11, fontFace: "Calibri", color: "333333",
      valign: "middle", margin: 0,
    });
  });

  // Bottom note
  s.addText("Артефакты передаются через файлы на диске → идемпотентность, возобновляемость, независимый дебаг каждой стадии", {
    x: 0.5, y: 5.07, w: 9, h: 0.3,
    fontSize: 10, italic: true, color: MUTED, fontFace: "Calibri", margin: 0,
  });

  addFooter(s, 4, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 5 — Стадия 1: Profiler
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Стадия 1: Profiler", "Что мы знаем про БД");

  // Левая колонка — что собираем
  s.addText("Собираемые метаданные", {
    x: 0.5, y: 1.15, w: 4.5, h: 0.35,
    fontSize: 16, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  s.addText([
    { text: "Структура: таблицы, колонки, типы, FK", options: { bullet: true, breakLine: true } },
    { text: "Статистика numeric: min / max / avg / median", options: { bullet: true, breakLine: true } },
    { text: "Статистика categorical: top-5 значений + частоты", options: { bullet: true, breakLine: true } },
    { text: "Date: min / max диапазон", options: { bullet: true, breakLine: true } },
    { text: "Sample-rows: 3 реальные строки / таблица", options: { bullet: true, breakLine: true } },
    { text: "Low-cardinality (enum) — полный список значений", options: { bullet: true } },
  ], {
    x: 0.5, y: 1.55, w: 4.5, h: 3,
    fontSize: 12, color: "333333", fontFace: "Calibri", paraSpaceAfter: 4,
  });

  // Правая колонка — пример profile.json
  s.addText("Пример profile.json", {
    x: 5.2, y: 1.15, w: 4.3, h: 0.35,
    fontSize: 16, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.55, w: 4.3, h: 3.05,
    fill: { color: NAVY_DARK }, line: { color: NAVY_DARK, width: 0 },
  });
  s.addText([
    { text: '"schema_str": "TABLE cards\\n  id BIGINT PK\\n  artist TEXT ..."', options: { breakLine: true } },
    { text: '"relationships_str": "cards.uuid → foreign_data.uuid"', options: { breakLine: true } },
    { text: '"column_stats_str":', options: { breakLine: true } },
    { text: '   "cards.artist: 252 distinct,', options: { breakLine: true } },
    { text: '    top=[John Avon:1128, Kev Walker:997]"', options: { breakLine: true } },
    { text: '"sample_rows": {"cards": [{...}, {...}, {...}]}', options: { breakLine: true } },
    { text: '"low_cardinality_values":', options: { breakLine: true } },
    { text: '   {"cards.bordercolor":', options: { breakLine: true } },
    { text: '    ["black","white","gold","silver"]}', options: { breakLine: true } },
    { text: '"complexity_score": 27.82', options: {} },
  ], {
    x: 5.3, y: 1.65, w: 4.1, h: 2.85,
    fontSize: 10, color: ICE, fontFace: "Consolas", margin: 0,
  });

  // Stat bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.85, w: 9, h: 0.4,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addText([
    { text: "card_games (115 колонок) → profile.json за ~30 секунд / ~120 SELECT-ов к системным таблицам", options: { fontSize: 11, color: "333333" } },
  ], {
    x: 0.7, y: 4.85, w: 8.6, h: 0.4,
    fontFace: "Calibri", valign: "middle", margin: 0,
  });

  addFooter(s, 5, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 6 — Стадия 2: Генерация
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Стадия 2: Генерация", "Два пути — стабильность + разнообразие");

  // Left card: Template
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.15, w: 4.3, h: 3.7,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.15, w: 4.3, h: 0.5,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Template generator", {
    x: 0.5, y: 1.15, w: 4.3, h: 0.5,
    fontSize: 16, bold: true, color: WHITE,
    align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
  });
  s.addText([
    { text: "25 SQL-шаблонов разной сложности", options: { bullet: true, breakLine: true } },
    { text: "100% валидный SQL — нет галлюцинаций", options: { bullet: true, breakLine: true } },
    { text: "Бесплатно — без API-вызовов", options: { bullet: true, breakLine: true } },
    { text: "Категории: lookup, count, top-N, JOIN, GROUP BY, CTE, window", options: { bullet: true, breakLine: true } },
    { text: "Подставляет реальные имена и значения из profile.json", options: { bullet: true } },
  ], {
    x: 0.7, y: 1.8, w: 4, h: 2.8,
    fontSize: 11, color: "333333", fontFace: "Calibri", paraSpaceAfter: 4,
  });

  // Right card: LLM
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.15, w: 4.3, h: 3.7,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.15, w: 4.3, h: 0.5,
    fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });
  s.addText("LLM generator", {
    x: 5.2, y: 1.15, w: 4.3, h: 0.5,
    fontSize: 16, bold: true, color: WHITE,
    align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
  });
  s.addText([
    { text: "GPT-4 / 4o / 4.1 / codex-mini как teacher-модели", options: { bullet: true, breakLine: true } },
    { text: "Естественные natural-language вопросы", options: { bullet: true, breakLine: true } },
    { text: "Сложные SQL: CTE, окна, подзапросы", options: { bullet: true, breakLine: true } },
    { text: "Параметр language: ru / en (train==inference invariant)", options: { bullet: true, breakLine: true } },
    { text: "Стоит денег, требует execution-валидации", options: { bullet: true } },
  ], {
    x: 5.4, y: 1.8, w: 4, h: 2.8,
    fontSize: 11, color: "333333", fontFace: "Calibri", paraSpaceAfter: 4,
  });

  // Bottom strategy
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.95, w: 9, h: 0.4,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Используем оба источника параллельно — Template даёт фундамент, LLM — разнообразие", {
    x: 0.5, y: 4.95, w: 9, h: 0.4,
    fontSize: 12, color: WHITE, bold: true,
    align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
  });

  addFooter(s, 6, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 7 — Стадия 3: Валидация
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Стадия 3: Валидация", "Защита от ошибочных примеров в три уровня");

  const filters = [
    { num: "1", title: "Parse",        desc: "sqlglot разбирает SQL как PostgreSQL. Отсев синтаксических ошибок." },
    { num: "2", title: "Whitelist",    desc: "Только SELECT. Запрет DROP / DELETE / UPDATE / TRUNCATE." },
    { num: "3", title: "Execute",      desc: "Выполнение на реальной БД. Должен вернуть ≥1 строку, без ошибок." },
    { num: "4", title: "LLM Judge",    desc: "Опц. GPT-4 проверяет семантическое соответствие question↔SQL." },
  ];

  filters.forEach((f, i) => {
    const y = 1.2 + i * 0.78;
    // Number badge
    s.addShape(pres.shapes.OVAL, {
      x: 0.6, y: y + 0.1, w: 0.5, h: 0.5,
      fill: { color: NAVY }, line: { color: NAVY, width: 0 },
    });
    s.addText(f.num, {
      x: 0.6, y: y + 0.1, w: 0.5, h: 0.5,
      fontSize: 18, bold: true, color: WHITE,
      align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
    });
    // Title
    s.addText(f.title, {
      x: 1.3, y, w: 2.5, h: 0.35,
      fontSize: 16, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
    });
    // Desc
    s.addText(f.desc, {
      x: 1.3, y: y + 0.35, w: 8, h: 0.4,
      fontSize: 11, color: "333333", fontFace: "Calibri", margin: 0,
    });
  });

  // Result stat
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.85, w: 9, h: 0.4,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addText([
    { text: "Реальный фильтр: ", options: { fontSize: 11, color: "333333" } },
    { text: "510 raw → 468 passed (93%)", options: { fontSize: 12, bold: true, color: ACCENT } },
    { text: "   reasons: sql_error=23, too_few_rows=11", options: { fontSize: 11, color: MUTED } },
  ], {
    x: 0.7, y: 4.85, w: 8.6, h: 0.4,
    fontFace: "Calibri", valign: "middle", margin: 0,
  });

  addFooter(s, 7, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 8 — Стадия 4: Fine-tuning
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Стадия 4: Fine-tuning", "LoRA-адаптеры на Qwen2.5-Coder-7B");

  // Левая колонка: таблица гиперпараметров
  s.addTable([
    [
      { text: "Параметр", options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } },
      { text: "Значение", options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 11 } },
    ],
    ["База",              "Qwen2.5-Coder-7B-Instruct (4-bit)"],
    ["Метод",             "LoRA"],
    ["r / α",             "16 / 32"],
    ["Learning rate",     "2e-4"],
    ["Epochs",            "1"],
    ["Batch × grad_accum", "1 × 8"],
    ["max_seq_length",    "12288"],
    ["Target modules",    "q/k/v/o/gate/up/down_proj"],
    ["GPU",               "A100 40GB (Colab)"],
    ["Acceleration",      "unsloth ~2×"],
  ], {
    x: 0.5, y: 1.15, w: 5.3,
    fontSize: 10, fontFace: "Calibri", color: "333333",
    border: { pt: 0.5, color: BORDER },
    rowH: 0.32,
  });

  // Правая колонка: ключевой инсайт
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.0, y: 1.15, w: 3.5, h: 3.7,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.0, y: 1.15, w: 0.08, h: 3.7,
    fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });

  s.addText("Ключевой инвариант", {
    x: 6.2, y: 1.3, w: 3.2, h: 0.35,
    fontSize: 14, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  s.addText("train == inference", {
    x: 6.2, y: 1.65, w: 3.2, h: 0.35,
    fontSize: 13, italic: true, color: ACCENT, fontFace: "Consolas", margin: 0,
  });
  s.addText([
    { text: "Системный промпт со схемой БД ≈ 10 000 токенов", options: { bullet: true, breakLine: true } },
    { text: "На контексте 4096 ответ обрезался — модель не училась SQL", options: { bullet: true, breakLine: true } },
    { text: "Подняли до 12288 — обучение работает корректно", options: { bullet: true, breakLine: true } },
    { text: "Eval отключён в trainer — OOM-фикс (val активации без чекпойнтинга)", options: { bullet: true } },
  ], {
    x: 6.2, y: 2.1, w: 3.2, h: 2.6,
    fontSize: 10, color: "333333", fontFace: "Calibri", paraSpaceAfter: 4,
  });

  // Bottom
  s.addText("~40 минут на одну модель × 5 генераторов = ~3.5 часа на полный эксперимент", {
    x: 0.5, y: 5.05, w: 9, h: 0.3,
    fontSize: 11, italic: true, color: MUTED, fontFace: "Calibri", margin: 0,
  });

  addFooter(s, 8, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 9 — Стадия 5: Production
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Стадия 5: Production", "От адаптера к работающему сервису");

  const items = [
    { title: "vLLM в Docker",
      desc: "LoRA hot-swap без merge; экономия диска; быстрая загрузка адаптеров" },
    { title: "MCP-сервер (FastMCP)",
      desc: "Интеграция с AI-агентами (Claude, GPT) через Model Control Protocol" },
    { title: "Python SDK",
      desc: "Text2SQLClient(api_url).query(question) → {sql, results}" },
    { title: "QueryLogger middleware",
      desc: "queries.jsonl + failed.jsonl — основа для self-healing цикла" },
    { title: "Hot-start через profile.json",
      desc: "<50 ms вместо 30+ секунд (нет повторного чтения схемы)" },
    { title: "Self-healing loop (heal/)",
      desc: "failed.jsonl → GPT-4 корректор → corrections.jsonl → retraining" },
  ];

  // 2-колоночная сетка
  items.forEach((it, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6;
    const y = 1.2 + row * 1.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.4, h: 1.15,
      fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.08, h: 1.15,
      fill: { color: NAVY }, line: { color: NAVY, width: 0 },
    });
    s.addText(it.title, {
      x: x + 0.2, y: y + 0.1, w: 4.1, h: 0.35,
      fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
    });
    s.addText(it.desc, {
      x: x + 0.2, y: y + 0.45, w: 4.1, h: 0.65,
      fontSize: 10, color: "333333", fontFace: "Calibri", margin: 0,
    });
  });

  addFooter(s, 9, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 10 — Постановка эксперимента
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Эксперимент", "Какая модель-генератор синтетики работает лучше всего?");

  // Левая половина: setup
  s.addText("Setup", {
    x: 0.5, y: 1.15, w: 4.5, h: 0.35,
    fontSize: 16, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  s.addTable([
    [{ text: "Параметр", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
     { text: "Значение", options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ["БД", "card_games (6 таблиц, 115 колонок)"],
    ["Test", "BIRD card_games subset"],
    ["Test-size", "31 реальный вопрос"],
    ["Метрика", "Execution accuracy"],
    ["Сравнение", "Result sets (set equality)"],
    ["Base FT", "Qwen-Coder-7B (4-bit)"],
    ["Train size", "≈ 500–1000 пар / эксп."],
  ], {
    x: 0.5, y: 1.55, w: 4.5,
    fontSize: 10, fontFace: "Calibri", color: "333333",
    border: { pt: 0.5, color: BORDER },
    rowH: 0.32,
  });

  // Правая половина: 5 генераторов
  s.addText("5 генераторов синтетики", {
    x: 5.2, y: 1.15, w: 4.3, h: 0.35,
    fontSize: 16, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
  });
  const gens = [
    { name: "gpt-4.1",            tier: "топ" },
    { name: "gpt-4o-mini",        tier: "средний" },
    { name: "gpt-5.1-codex-mini", tier: "code-spec" },
    { name: "gpt-4.1-mini",       tier: "средний" },
    { name: "gpt-4.1-nano",       tier: "бюджетный" },
  ];
  gens.forEach((g, i) => {
    const y = 1.6 + i * 0.5;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y, w: 4.3, h: 0.4,
      fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
    });
    s.addText(g.name, {
      x: 5.4, y, w: 2.7, h: 0.4,
      fontSize: 12, bold: true, color: NAVY, fontFace: "Consolas",
      valign: "middle", margin: 0,
    });
    s.addText(g.tier, {
      x: 8.0, y, w: 1.3, h: 0.4,
      fontSize: 11, italic: true, color: ACCENT, fontFace: "Calibri",
      valign: "middle", align: "right", margin: 0,
    });
  });

  // Вопрос
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.85, w: 9, h: 0.4,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText("Главный вопрос: нужна ли сильная и дорогая LLM для генерации синтетики?", {
    x: 0.5, y: 4.85, w: 9, h: 0.4,
    fontSize: 12, bold: true, color: WHITE,
    align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
  });

  addFooter(s, 10, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 11 — Главная таблица результатов
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Результаты", "Три итерации: RU → RU+EN → EN");

  const hdrFill = { color: NAVY };
  const hdrOpt = { bold: true, color: WHITE, fill: hdrFill, fontSize: 11, align: "center" };
  const accentOpt = { bold: true, color: ACCENT, fontSize: 11, align: "center" };
  const cellOpt = { fontSize: 11, align: "center", color: "333333" };
  const labelOpt = { fontSize: 11, bold: true, color: NAVY };

  s.addTable([
    [
      { text: "Генератор", options: hdrOpt },
      { text: "Эксп. 1 (RU)", options: hdrOpt },
      { text: "Эксп. 2 (RU+EN)", options: hdrOpt },
      { text: "Эксп. 3 (EN)", options: hdrOpt },
    ],
    [
      { text: "baseline (без FT)", options: labelOpt },
      { text: "19.4%", options: cellOpt },
      { text: "19.4%", options: cellOpt },
      { text: "32.3%", options: accentOpt },
    ],
    [
      { text: "gpt-4.1", options: labelOpt },
      { text: "22.6%", options: cellOpt },
      { text: "39.4%", options: accentOpt },
      { text: "32.3%", options: cellOpt },
    ],
    [
      { text: "gpt-4o-mini", options: labelOpt },
      { text: "12.9%", options: cellOpt },
      { text: "29.0%", options: accentOpt },
      { text: "25.8%", options: cellOpt },
    ],
    [
      { text: "gpt-5.1-codex-mini", options: labelOpt },
      { text: "19.4%", options: cellOpt },
      { text: "22.6%", options: cellOpt },
      { text: "25.8%", options: cellOpt },
    ],
    [
      { text: "gpt-4.1-mini", options: labelOpt },
      { text: "6.5%", options: cellOpt },
      { text: "19.4%", options: cellOpt },
      { text: "—", options: cellOpt },
    ],
    [
      { text: "gpt-4.1-nano", options: labelOpt },
      { text: "16.1%", options: cellOpt },
      { text: "16.1%", options: cellOpt },
      { text: "35.5%", options: accentOpt },
    ],
  ], {
    x: 0.5, y: 1.15, w: 9,
    fontFace: "Calibri",
    border: { pt: 0.5, color: BORDER },
    rowH: 0.36,
  });

  // Footnote с точкой отсчёта
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9, h: 0.85,
    fill: { color: LIGHT_BG }, line: { color: BORDER, width: 1 },
  });
  s.addText([
    { text: "Точка отсчёта (classical FT на полном BIRD train): ", options: { fontSize: 12, color: "333333" } },
    { text: "38.7%", options: { fontSize: 14, bold: true, color: NAVY } },
    { text: "  →  Эксп. 2 / gpt-4.1 = ", options: { fontSize: 12, color: "333333" } },
    { text: "39.4%", options: { fontSize: 14, bold: true, color: ACCENT } },
    { text: " (обходит при 10× меньше данных, без человеческой разметки)", options: { fontSize: 12, color: "333333" } },
  ], {
    x: 0.7, y: 4.55, w: 8.6, h: 0.6,
    fontFace: "Calibri", valign: "middle", margin: 0,
  });

  addFooter(s, 11, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 12 — Ключевые выводы
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Ключевые выводы", "5 наблюдений из трёх итераций");

  const findings = [
    {
      title: "Сравнялись с classical full-BIRD FT",
      body: "Эксп. 2 / gpt-4.1 (39.4%) ≈ classical FT (38.7%) при 10× меньше данных и без разметки",
    },
    {
      title: "Языковой инвариант train↔test критичен",
      body: "Эксп. 1 → 2: добавление EN-синтетики дало +17 п.п. для gpt-4.1 (с 22.6% до 39.4%)",
    },
    {
      title: "Evidence в инференсе — мощный самостоятельный рычаг",
      body: "baseline без FT: 19.4% → 32.3% (+12.9 п.п.) только от правильной подачи подсказок",
    },
    {
      title: "Сила генератора важна, но нелинейно",
      body: "На сильном baseline дешёвая nano парадоксально лучшая (35.5%) — её SQL ближе к BIRD-стилю",
    },
    {
      title: "Эксп. 1 — методологическая находка, не failure",
      body: "Зафиксировали языковой рассинхрон как явление и устранили в Эксп. 2",
    },
  ];

  findings.forEach((f, i) => {
    const y = 1.15 + i * 0.78;
    // Number badge
    s.addShape(pres.shapes.OVAL, {
      x: 0.5, y: y + 0.05, w: 0.5, h: 0.5,
      fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
    });
    s.addText(String(i + 1), {
      x: 0.5, y: y + 0.05, w: 0.5, h: 0.5,
      fontSize: 16, bold: true, color: WHITE,
      align: "center", valign: "middle", fontFace: "Calibri", margin: 0,
    });
    // Title
    s.addText(f.title, {
      x: 1.2, y, w: 8.3, h: 0.32,
      fontSize: 14, bold: true, color: NAVY, fontFace: "Calibri", margin: 0,
    });
    // Body
    s.addText(f.body, {
      x: 1.2, y: y + 0.32, w: 8.3, h: 0.4,
      fontSize: 11, color: "333333", fontFace: "Calibri", margin: 0,
    });
  });

  addFooter(s, 12, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 13 — Сравнение с classical
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, WHITE);
  addHeader(s, "Сравнение с classical full-BIRD FT", "Главный научный contribution");

  s.addTable([
    [
      { text: "Параметр", options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 12 } },
      { text: "Classical FT (Эксп. 0)", options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 12 } },
      { text: "AutoText2SQL (Эксп. 2)", options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 12 } },
    ],
    [
      { text: "Объём данных", options: { fontSize: 11, bold: true, color: NAVY } },
      { text: "≈ 10 000 человеческих пар", options: { fontSize: 11, color: "333333" } },
      { text: "1000 автогенерированных пар", options: { fontSize: 11, color: ACCENT, bold: true } },
    ],
    [
      { text: "Разметка", options: { fontSize: 11, bold: true, color: NAVY } },
      { text: "Ручная (дни — недели)", options: { fontSize: 11, color: "333333" } },
      { text: "Полностью автоматическая", options: { fontSize: 11, color: ACCENT, bold: true } },
    ],
    [
      { text: "Применимость", options: { fontSize: 11, bold: true, color: NAVY } },
      { text: "Только под BIRD-БД", options: { fontSize: 11, color: "333333" } },
      { text: "Любая PostgreSQL-БД", options: { fontSize: 11, color: ACCENT, bold: true } },
    ],
    [
      { text: "Время с нуля под новую БД", options: { fontSize: 11, bold: true, color: NAVY } },
      { text: "Недели разметки + обучение", options: { fontSize: 11, color: "333333" } },
      { text: "≈ 30 минут end-to-end", options: { fontSize: 11, color: ACCENT, bold: true } },
    ],
    [
      { text: "Accuracy на BIRD card_games", options: { fontSize: 11, bold: true, color: NAVY } },
      { text: "38.7%", options: { fontSize: 13, bold: true, color: "333333", align: "center" } },
      { text: "39.4%", options: { fontSize: 13, bold: true, color: ACCENT, align: "center" } },
    ],
  ], {
    x: 0.5, y: 1.15, w: 9,
    fontFace: "Calibri",
    border: { pt: 0.5, color: BORDER },
    rowH: 0.5,
  });

  // Main message
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.4, w: 9, h: 0.85,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText([
    { text: "Для одной БД автоматически сгенерированной синтетики достаточно, ", options: { fontSize: 13, color: WHITE } },
    { text: "чтобы воспроизвести качество classical FT по всему BIRD, ", options: { fontSize: 13, color: WHITE } },
    { text: "при этом снять зависимость от человеческой разметки", options: { fontSize: 13, color: WHITE, bold: true } },
  ], {
    x: 0.7, y: 4.55, w: 8.6, h: 0.6,
    fontFace: "Calibri", valign: "middle", margin: 0,
  });

  addFooter(s, 13, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════
// СЛАЙД 14 — Что дальше + Q&A (dark)
// ═══════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  bg(s, NAVY);

  // Декор-акцент
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });

  s.addText("Что сделано · Что дальше", {
    x: 0.6, y: 0.5, w: 9, h: 0.55,
    fontSize: 28, bold: true, color: WHITE, fontFace: "Georgia", margin: 0,
  });

  // Левая колонка: Done
  s.addText("Сделано", {
    x: 0.6, y: 1.3, w: 4.2, h: 0.4,
    fontSize: 18, bold: true, color: ACCENT, fontFace: "Calibri", margin: 0,
  });
  s.addText([
    { text: "Полный автономный пайплайн (CLI, 7 команд)", options: { bullet: true, breakLine: true } },
    { text: "3 итерации экспериментов", options: { bullet: true, breakLine: true } },
    { text: "vLLM-серверинг + MCP-обёртка", options: { bullet: true, breakLine: true } },
    { text: "Self-healing loop через failed-queries", options: { bullet: true, breakLine: true } },
    { text: "35 unit-тестов + документация", options: { bullet: true, breakLine: true } },
    { text: "Сопоставимость с classical full-BIRD FT", options: { bullet: true } },
  ], {
    x: 0.6, y: 1.7, w: 4.2, h: 3.2,
    fontSize: 12, color: ICE, fontFace: "Calibri", paraSpaceAfter: 5,
  });

  // Правая колонка: Future
  s.addText("Дальше", {
    x: 5.2, y: 1.3, w: 4.3, h: 0.4,
    fontSize: 18, bold: true, color: ACCENT, fontFace: "Calibri", margin: 0,
  });
  s.addText([
    { text: "Прогон на enterprise-БД (30+ таблиц)", options: { bullet: true, breakLine: true } },
    { text: "DPO / RLAIF поверх SFT-адаптера", options: { bullet: true, breakLine: true } },
    { text: "Адаптивный target_count под complexity_score", options: { bullet: true, breakLine: true } },
    { text: "Redact PII в low_cardinality перед отправкой в LLM", options: { bullet: true, breakLine: true } },
    { text: "Интеграция с реальной BI-системой", options: { bullet: true, breakLine: true } },
    { text: "Расширение на MySQL / Snowflake / BigQuery", options: { bullet: true } },
  ], {
    x: 5.2, y: 1.7, w: 4.3, h: 3.2,
    fontSize: 12, color: ICE, fontFace: "Calibri", paraSpaceAfter: 5,
  });

  // Q&A bar внизу
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 5.05, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
  });
  s.addText("Q & A", {
    x: 0.6, y: 5.15, w: 9, h: 0.4,
    fontSize: 18, bold: true, color: WHITE, fontFace: "Georgia", margin: 0,
  });
}

// Save
pres.writeFile({ fileName: "../AutoText2SQL.pptx" })
  .then(name => console.log("✓ saved:", name))
  .catch(err => { console.error("ERROR:", err); process.exit(1); });
