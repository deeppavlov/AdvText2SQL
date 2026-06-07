"""
Top-level CLI for the autonomous Text2SQL pipeline.

Команды соответствуют стадиям пайплайна из дорожной карты:

    profile        — извлечь профиль БД (Week 1)
    generate       — сгенерировать синтетический Q-SQL датасет (Week 2)
    build-dataset  — собрать train.jsonl/val.jsonl из профиля + синтетики (Week 2)
    train          — подготовить ноутбук для FT в Colab (Week 2)
    serve          — поднять inference-сервис (Week 3)
    heal           — собрать failed_queries и подготовить retraining (Week 3)
    init           — orchestrator: profile → generate → build-dataset → train

Каждый subcommand читает артефакт предыдущей стадии с диска (`data/profiles/...`,
`data/synthetic/...`), пишет следующий — стадии идемпотентны и независимы.
"""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="text2sql",
    help="Autonomous Text2SQL pipeline: DB → synthetic dataset → fine-tune → serve.",
    no_args_is_help=True,
)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Week 1
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def profile(
    db_url: str = typer.Option(..., "--db-url", help="PostgreSQL connection string"),
    out: str = typer.Option(
        "data/profiles",
        "--out",
        help="Directory to write <db_id>/profile.json",
    ),
    sample_size: int = typer.Option(
        500,
        "--sample-size",
        help="Rows per table for type detection (smaller = faster, less accurate)",
    ),
    skip_stats: bool = typer.Option(
        False,
        "--skip-stats",
        help="Skip column statistics collection (still extracts schema + relationships)",
    ),
) -> None:
    """Извлечь профиль БД → data/profiles/<db_id>/profile.json."""
    from adv_text2sql.profiler.cli import run_profile

    run_profile(db_url=db_url, out_dir=out, sample_size=sample_size, skip_stats=skip_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Week 2 (stubs — implement in week 2)
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def generate(
    profile_path: str = typer.Option(..., "--profile", help="Path to profile.json"),
    count: str = typer.Option("auto", "--count", help="Target sample count or 'auto'"),
    generator: str = typer.Option(
        "template",
        "--generator",
        help="Generator type: template | llm | both",
    ),
    llm_model: str = typer.Option("gpt-4o", "--llm-model", help="LLM model id (for --generator=llm|both)"),
    judge: bool = typer.Option(False, "--judge", help="Enable LLM-judge filter"),
    auto_resize: bool = typer.Option(
        False, "--auto-resize", help="Double target_count if <40% pass validation"
    ),
    out: str = typer.Option("data/synthetic", "--out"),
    db_url: str = typer.Option(
        None, "--db-url", help="PG URL для execute-валидации (без него — только parse+whitelist)"
    ),
    language: str = typer.Option(
        "ru", "--language", help="Язык вопросов: ru | en (train==inference!)"
    ),
) -> None:
    """Сгенерировать + отфильтровать синтетический Q-SQL датасет."""
    from adv_text2sql.synth.cli import run_generate

    stats = run_generate(
        profile_path=profile_path,
        count=count,
        generator=generator,
        llm_model=llm_model,
        judge=judge,
        auto_resize=auto_resize,
        out_dir=out,
        db_url=db_url,
        language=language,
    )
    console.print(f"\n[bold green]Done.[/bold green] Stats: {stats}")


@app.command("build-dataset")
def build_dataset(
    profile_path: str = typer.Option(..., "--profile", help="Path to profile.json"),
    synthetic: str = typer.Option(..., "--synthetic", help="Path to *_validated.jsonl"),
    out: str = typer.Option("data/finetune", "--out"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
) -> None:
    """Собрать train.jsonl/val.jsonl в chat-format для FT."""
    from adv_text2sql.training.dataset_builder import run_build_dataset

    run_build_dataset(
        profile_path=profile_path,
        synthetic_path=synthetic,
        out_dir=out,
        val_ratio=val_ratio,
    )


@app.command()
def train(
    db_id: str = typer.Option(..., "--db-id"),
    base_model: str = typer.Option(
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "--base-model"
    ),
    drive_dir: str = typer.Option("text2sql_finetune", "--drive-dir"),
    output_dir: str = typer.Option("notebooks", "--output-dir"),
) -> None:
    """Сгенерировать Colab-ноутбук для FT под конкретную БД.

    После генерации: залить train.jsonl/val.jsonl на Drive в
    `MyDrive/<drive_dir>/<db_id>/`, открыть .ipynb в Colab → Run all.
    """
    from adv_text2sql.training.notebook_generator import generate_finetune_notebook

    path = generate_finetune_notebook(
        db_id=db_id,
        base_model=base_model,
        drive_data_dir=drive_dir,
        output_dir=output_dir,
    )
    console.print(
        f"[bold green]✓ Notebook generated[/bold green]: {path}\n"
        f"[bold]Next:[/bold]\n"
        f"  1. Залей data/finetune/{db_id}/train.jsonl + val.jsonl на Drive в\n"
        f"     MyDrive/{drive_dir}/{db_id}/\n"
        f"  2. Открой {path} в Colab → Runtime: GPU → Run all\n"
        f"  3. Через ~1 час получишь LoRA-адаптер в MyDrive/{drive_dir}/{db_id}/adapter/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Week 3 (stubs)
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    profile_path: str = typer.Option(..., "--profile"),
    model_url: str = typer.Option("http://localhost:8000/v1", "--model-url"),
    model_name: str = typer.Option("text2sql", "--model-name"),
    db_url: str = typer.Option(None, "--db-url", help="Если задан — клиент будет исполнять SQL"),
    question: str = typer.Option(None, "--question", help="Одноразовый запрос (без -q входит в REPL)"),
) -> None:
    """REPL-клиент поверх vLLM-сервера (или одиночный запрос через --question).

    Сам vLLM запускается через docker-compose (см. src/adv_text2sql/serve/docker-compose.yml).
    """
    from adv_text2sql.profiler.profile import Profile
    from adv_text2sql.serve.client import Text2SQLClient

    profile = Profile.load_json(profile_path)
    client = Text2SQLClient(
        profile=profile,
        api_url=model_url,
        model_name=model_name,
        execute_db_url=db_url,
    )

    def ask(q: str) -> None:
        result = client.query(q, execute=db_url is not None)
        console.print(f"[bold]SQL:[/bold]\n{result.sql}")
        if result.error:
            console.print(f"[red]Error:[/red] {result.error}")
        elif result.results is not None:
            console.print(f"[green]{len(result.results)} rows in {result.latency_ms:.0f}ms[/green]")
            for row in result.results[:5]:
                console.print(f"  {row}")
        else:
            console.print(f"[dim]Generated in {result.latency_ms:.0f}ms[/dim]")

    if question:
        ask(question)
        return

    console.print(f"[bold cyan]Text2SQL REPL[/bold cyan] for {profile.db_id}. Empty line to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ask(q)


@app.command("eval")
def eval_cmd(
    profile_path: str = typer.Option(..., "--profile"),
    gold_path: str = typer.Option(..., "--gold", help="BIRD/Ambrosia gold JSON"),
    model_url: str = typer.Option("http://localhost:8000/v1", "--model-url"),
    model_name: str = typer.Option("text2sql", "--model-name"),
    db_url: str = typer.Option(..., "--db-url"),
    limit: int = typer.Option(50, "--limit"),
    use_evidence: bool = typer.Option(False, "--use-evidence"),
    out: str = typer.Option("experiments", "--out"),
) -> None:
    """Прогон FT-модели на BIRD subset → execution accuracy report."""
    from adv_text2sql.eval.cli import run_eval_cmd

    run_eval_cmd(
        profile_path=profile_path,
        gold_path=gold_path,
        model_url=model_url,
        model_name=model_name,
        db_url=db_url,
        limit=limit,
        use_evidence=use_evidence,
        out_dir=out,
    )


@app.command()
def heal(
    profile_path: str = typer.Option(..., "--profile"),
    db_url: str = typer.Option(..., "--db-url", help="PG URL для validation"),
    min_samples: int = typer.Option(50, "--min-samples"),
    judge_model: str = typer.Option("gpt-4o", "--judge-model"),
    out: str = typer.Option("data/heal", "--out"),
) -> None:
    """Собрать failed_queries → корректирующая генерация → corrections.jsonl."""
    import asyncio

    from adv_text2sql.heal.collector import HealCollector
    from adv_text2sql.heal.sql_generator import HealConfig, HealSQLGenerator
    from adv_text2sql.profiler.profile import Profile

    profile = Profile.load_json(profile_path)
    collector = HealCollector(min_samples=min_samples)
    result = collector.collect(db_id=profile.db_id)

    console.print(
        f"[bold cyan]Heal stage[/bold cyan]\n"
        f"  total failed = {result.total_failed}\n"
        f"  after dedup  = {result.after_dedup_exact}"
    )
    if len(result.candidates) < min_samples:
        console.print(
            f"[yellow]Only {len(result.candidates)} candidates < {min_samples} min_samples — skipping[/yellow]"
        )
        raise typer.Exit(code=0)

    config = HealConfig(model_name=judge_model)
    generator = HealSQLGenerator(profile, db_url, config)
    out_path = f"{out}/{profile.db_id}_corrections.jsonl"
    healing = asyncio.run(generator.heal(result.candidates, out_path))

    console.print(
        f"[bold green]✓ Healing done[/bold green]\n"
        f"  healed       = {healing.healed} / {healing.total_input} ({healing.heal_rate*100:.1f}%)\n"
        f"  corrections  = {healing.corrections_path}\n"
        f"  [bold]Next:[/bold] merge with previous validated.jsonl + rerun build-dataset + train"
    )


@app.command()
def init(
    db_url: str = typer.Option(..., "--db-url"),
    skip_generate: bool = typer.Option(False, "--skip-generate"),
    skip_train: bool = typer.Option(False, "--skip-train"),
    generator: str = typer.Option("template", "--generator"),
    count: str = typer.Option("auto", "--count"),
) -> None:
    """Orchestrator: profile → generate → build-dataset → train (notebook gen).

    Полный flow для новой БД. Не запускает train.jsonl на Colab автоматически —
    выводит инструкцию пользователю что делать дальше.
    """
    from adv_text2sql.profiler.cli import run_profile
    from adv_text2sql.profiler.profile import Profile
    from adv_text2sql.synth.cli import run_generate
    from adv_text2sql.training.dataset_builder import run_build_dataset
    from adv_text2sql.training.notebook_generator import generate_finetune_notebook

    # Stage 1
    console.rule("[bold]Stage 1: PROFILE")
    profile_path = run_profile(db_url=db_url, out_dir="data/profiles")
    profile = Profile.load_json(profile_path)

    if skip_generate:
        console.print("[yellow]Skipping generate + train (--skip-generate)[/yellow]")
        return

    # Stage 2
    console.rule("[bold]Stage 2: GENERATE")
    gen_stats = run_generate(
        profile_path=str(profile_path),
        count=count,
        generator=generator,
        out_dir="data/synthetic",
        db_url=db_url,
        auto_resize=True,
    )

    # Stage 3
    console.rule("[bold]Stage 3: BUILD-DATASET")
    run_build_dataset(
        profile_path=str(profile_path),
        synthetic_path=gen_stats["passed_path"],
        out_dir="data/finetune",
    )

    if skip_train:
        console.print("[yellow]Skipping notebook generation (--skip-train)[/yellow]")
        return

    # Stage 4
    console.rule("[bold]Stage 4: TRAIN (notebook)")
    nb = generate_finetune_notebook(db_id=profile.db_id)
    console.print(
        f"\n[bold green]✓ Pipeline ready.[/bold green]\n"
        f"  profile  = data/profiles/{profile.db_id}/profile.json\n"
        f"  synthetic = {gen_stats['passed_path']}\n"
        f"  dataset  = data/finetune/{profile.db_id}/\n"
        f"  notebook = {nb}\n\n"
        f"[bold]Финальные шаги:[/bold]\n"
        f"  1. Залей data/finetune/{profile.db_id}/{{train,val}}.jsonl на Drive в text2sql_finetune/{profile.db_id}/\n"
        f"  2. Открой {nb} в Colab → Runtime: GPU → Run all (~1 час)\n"
        f"  3. Скачай adapter и запусти docker-compose up -d (см. src/adv_text2sql/serve/)"
    )


if __name__ == "__main__":
    app()
