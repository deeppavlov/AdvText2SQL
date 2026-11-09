"""Real GEPA, DSPy MIPROv2 and TextGrad; all task calls use one shared evaluator."""

import json
import random
from unittest.mock import patch

from atomic_artifacts import write_json
import prompt_comparison as campaign
from static_hybrid_runtime import append


def constraints(session):
    common = (
        "Optimize transferable instructions only. Do not copy dataset questions, "
        "gold answers, schema names, literal values or particular-question rules. "
        "Do not change the algorithm, call count, feature mask, output contract or question. "
        "Do not add placeholders, braces, fenced code or new worked examples. "
    )
    if session.slot == "sql":
        return (
            common
            + "Return SQL rules in at most 6000 characters; PostgreSQL and SQL-only output are immutable."
        )
    return (
        common
        + "Keep exactly one [DATABASE_SCHEMA] marker. Improve only the ambiguity taxonomy "
        "rule body, at most 12000 characters. The protected JSON output fields are "
        "is_ambiguous (boolean), ambiguity_type (A1/A3/null), reason. "
        "Existing seed examples may be retained; do not copy additional dataset questions."
    )


def valid_or_record(session, value):
    try:
        campaign.check(value, session.manifest, session.slot, session.fit)
        # Validate rendering/context size before any task request.
        campaign.render(
            value, session.manifest, session.slot, session.fit, session.inner[0]
        )
        return True
    except ValueError as error:
        append(
            session.folder / "invalid_candidates.jsonl",
            {"candidate": value, "error": str(error)},
        )
        return False


def evaluate_inner(session, value):
    if valid_or_record(session, value):
        for item in session.inner:
            session.evaluate(item, value)


def gepa_search(session):
    import gepa
    from gepa.core.adapter import EvaluationBatch
    from gepa.logging.logger import StdOutLogger

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            if set(candidate) != {"rules"}:
                raise ValueError("Only rules are mutable")
            allowed = session.fit + session.inner
            if any(not any(i == j for j in allowed) for i in batch):
                raise ValueError("GEPA attempted to access external dev")
            value = campaign.candidate(candidate["rules"])
            rows = (
                [session.evaluate(i, value) for i in batch]
                if valid_or_record(session, value)
                else [
                    {
                        "score": 0.0,
                        "prediction": "error",
                        "error": "invalid_instruction",
                    }
                    for i in batch
                ]
            )
            traces = (
                [{"item": i, "result": r} for i, r in zip(batch, rows)]
                if capture_traces
                else None
            )
            return EvaluationBatch(
                outputs=rows, scores=[r["score"] for r in rows], trajectories=traces
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            if components_to_update != ["rules"]:
                raise ValueError("Only rules are mutable")
            items = [t["item"] for t in eval_batch.trajectories or []]
            return {
                "rules": session.feedback(items, campaign.candidate(candidate["rules"]))
            }

    template = (
        constraints(session) + "\nCurrent instructions:\n<curr_param>\n"
        "Optimization-fit feedback only:\n<side_info>\n"
        "Return just the proposed rules in one ``` block."
    )
    result = gepa.optimize(
        seed_candidate={"rules": session.manifest["seed_instructions"][session.slot]},
        trainset=session.fit,
        valset=session.inner,
        adapter=Adapter(),
        reflection_lm=session.reflect,
        reflection_prompt_template=template,
        reflection_minibatch_size=4,
        max_metric_calls=session.manifest["limits"]["examples"],
        stop_callbacks=lambda state: session.ledger.data["teacher_calls"]
        >= session.manifest["limits"]["teacher_calls"],
        use_merge=False,
        run_dir=str(session.output / "gepa"),
        seed=session.job["seed"],
        raise_on_exception=True,
        display_progress_bar=False,
        logger=StdOutLogger(),
    )
    write_json(session.output / "gepa_result.json", result.to_dict())


def textgrad_search(session):
    import textgrad as tg
    from textgrad.autograd.string_based_ops import StringBasedFunction

    class Engine(tg.EngineLM):
        model_string = session.manifest["teacher"]["model"]

        def generate(self, prompt, system_prompt=None, **kwargs):
            return session.reflect(
                "Follow the backward/update output format requested below. Constraints apply only to the updated instruction variable, not gradient feedback.\n"
                + constraints(session)
                + "\n"
                + (system_prompt or "")
                + "\n"
                + prompt
            )

        def __call__(self, *args, **kwargs):
            return self.generate(*args, **kwargs)

    engine = Engine()
    variable = tg.Variable(
        session.manifest["seed_instructions"][session.slot],
        requires_grad=True,
        role_description=f"general {session.slot} instruction rules",
    )
    optimizer = tg.TGD(
        parameters=[variable], engine=engine, constraints=[constraints(session)]
    )
    rng = random.Random(session.job["seed"])
    for step in range(12):
        current = campaign.candidate(variable.value)
        if not valid_or_record(session, current):
            variable.set_value(session.manifest["seed_instructions"][session.slot])
            continue
        batch = rng.sample(session.fit, min(4, len(session.fit)))

        def measured(instruction):
            # Actual executable metric + safe fit feedback, not an LLM-judged loss.
            return json.dumps(
                session.feedback(batch, campaign.candidate(instruction.value)),
                ensure_ascii=False,
            )

        fn = StringBasedFunction(
            measured,
            "Improve measured fit correctness of unchanged Text-to-SQL pipeline",
        )
        optimizer.zero_grad()
        loss = fn(
            {"instruction": variable},
            response_role_description="execution/ambiguity correctness and optimization-fit diagnostics",
        )
        loss.backward(engine=engine)
        try:
            optimizer.step()
        except IndexError:
            append(
                session.folder / "invalid_candidates.jsonl",
                {"step": step + 1, "error": "textgrad_update_tags_missing"},
            )
            continue
        value = campaign.candidate(variable.value)
        append(
            session.folder / "textgrad_steps.jsonl",
            {"step": step + 1, "candidate": value},
        )
        evaluate_inner(session, value)


def mipro_search(session):
    import dspy

    class Teacher(dspy.BaseLM):
        def __init__(self):
            super().__init__(
                model=session.manifest["teacher"]["model"],
                cache=False,
                temperature=0,
                max_tokens=2048,
                num_retries=0,
            )

        def __call__(self, prompt=None, *, messages=None, **kwargs):
            if kwargs.get("n", 1) != 1:
                raise ValueError("One accounted response per teacher call")
            # DSPy formats and parses its own structured output. The provider
            # transport itself stays in our native, budgeted API adapter.
            text = (
                "Follow the DSPy output sections requested in the messages below. Constraints apply only to proposed_instruction, not observations/summary.\n"
                + constraints(session)
                + "\n"
                + json.dumps(
                    messages if messages is not None else prompt, ensure_ascii=False
                )
            )
            return [session.reflect(text)]

        def forward(self, *args, **kwargs):
            raise RuntimeError("Use accounted teacher __call__")

    class Signature(dspy.Signature):
        """Convert a question using the frozen schema/task and the supplied rules."""

        qid: str = dspy.InputField()
        question: str = dspy.InputField()
        evidence: str = dspy.InputField()
        schema_text: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Program(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(
                Signature.with_instructions(
                    session.manifest["seed_instructions"][session.slot]
                )
            )

        def forward(self, qid, question, evidence, schema_text):
            lookup = {campaign.item_id(i): i for i in session.fit + session.inner}
            if qid not in lookup:
                raise ValueError("MIPRO attempted to access a non-training item")
            item = lookup[qid]
            if question != item["question"] or evidence != item.get("evidence", ""):
                raise ValueError("MIPRO may not rewrite the task question")
            demos = [str(d.qid) for d in self.predict.demos]
            value = campaign.candidate(self.predict.signature.instructions, demos)
            if session.job["method"] == "mipro_zero" and demos:
                raise ValueError("Zero-shot search cannot use train demonstrations")
            if not valid_or_record(session, value):
                return dspy.Prediction(answer="error", score=0.0)
            result = session.evaluate(item, value)
            return dspy.Prediction(answer=result["prediction"], score=result["score"])

    def examples(items):
        return [
            dspy.Example(
                qid=campaign.item_id(i),
                question=i["question"],
                evidence=i.get("evidence", ""),
                schema_text=json.dumps(
                    session.manifest["schema_audit"]["schemas"][session.benchmark][
                        i["db_id"]
                    ]["rows"],
                    ensure_ascii=False,
                ),
                answer=i["SQL"]
                if session.slot == "sql"
                else ("ambiguous" if i["SQL"] == "ambiguous" else "unambiguous"),
            ).with_inputs("qid", "question", "evidence", "schema_text")
            for i in items
        ]

    teacher = Teacher()

    # Disable alternate JSON adapters/provider fallbacks: all teacher traffic
    # must pass through the budget ledger, never an implicit SDK client.
    def labeled_sets(
        student, num_candidate_sets, trainset, max_labeled_demos, rng=None, **kwargs
    ):
        # Official MIPRO helper samples randint(1, 0) for labeled-only mode.
        # Keep its proposer/Optuna search; explicitly build bounded labeled sets
        # with official LabeledFewShot instead of implicit bootstrap trajectories.
        rng = rng or random.Random(session.job["seed"])
        result = {i: [] for i, _ in enumerate(student.predictors())}
        for n in range(num_candidate_sets):
            rows = list(trainset)
            rng.shuffle(rows)
            program = dspy.LabeledFewShot(k=0 if n == 0 else max_labeled_demos).compile(
                student, trainset=rows, sample=False
            )
            for i, predictor in enumerate(program.predictors()):
                result[i].append(predictor.demos)
        return result

    with (
        dspy.context(
            lm=teacher, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
        ),
        patch(
            "dspy.teleprompt.mipro_optimizer_v2.create_n_fewshot_demo_sets",
            labeled_sets,
        ),
    ):
        optimizer = dspy.MIPROv2(
            metric=lambda gold, pred, trace=None: pred.score,
            prompt_model=teacher,
            task_model=teacher,
            auto=None,
            num_candidates=8,
            max_bootstrapped_demos=0,
            max_labeled_demos=2 if session.job["method"] == "mipro_few" else 0,
            num_threads=1,
            max_errors=1,
            seed=session.job["seed"],
            init_temperature=0,
            log_dir=str(session.output / "mipro"),
        )
        program = optimizer.compile(
            Program(),
            trainset=examples(session.fit),
            valset=examples(session.inner),
            num_trials=12,
            minibatch=True,
            minibatch_size=min(8, len(session.inner)),
            minibatch_full_eval_steps=3,
            program_aware_proposer=False,
            data_aware_proposer=True,
            view_data_batch_size=8,
            tip_aware_proposer=True,
            fewshot_aware_proposer=True,
            provide_traceback=False,
        )
        value = campaign.candidate(
            program.predict.signature.instructions,
            [str(d.qid) for d in program.predict.demos],
        )
        evaluate_inner(session, value)
        write_json(session.output / "mipro_result.json", {"candidate": value})


def optimize(session):
    {
        "gepa_zero": gepa_search,
        "textgrad_zero": textgrad_search,
        "mipro_zero": mipro_search,
        "mipro_few": mipro_search,
    }[session.job["method"]](session)
