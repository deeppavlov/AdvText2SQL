"""Persistent hard call limits; reserve unknown request costs before network IO."""

from time import monotonic
import json
import uuid

from atomic_artifacts import write_json


class BudgetStop(BaseException):
    """Never convert exhausted infrastructure budgets into negative model labels."""


class AccountingError(BaseException):
    """Malformed provider accounting is a failure, not a completed budget stop."""


class Ledger:
    def __init__(self, folder, limits, *, clock=monotonic):
        self.path = folder / "budget.json"
        self.limits, self.clock = limits, clock
        self.data = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {
                "task_calls": 0,
                "teacher_calls": 0,
                "tokens": 0,
                "examples": 0,
                "active_seconds": 0,
                "pending": {},
                "example_keys": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "task_tokens": 0,
                "teacher_tokens": 0,
            }
        )
        self.started, self.prior_seconds = clock(), self.data["active_seconds"]
        self.save()

    def save(self):
        self.data["active_seconds"] = self.prior_seconds + self.clock() - self.started
        write_json(self.path, self.data)

    def check_time(self):
        self.save()
        if self.data["active_seconds"] >= self.limits["active_seconds"]:
            raise BudgetStop("Active-time budget exhausted")

    def example(self, key):
        self.check_time()
        if key in self.data["example_keys"]:
            return
        if self.data["examples"] >= self.limits["examples"]:
            raise BudgetStop("Task-example budget exhausted")
        self.data["examples"] += 1
        self.data["example_keys"].append(key)
        self.save()

    def reserve(self, role, messages, output_limit):
        self.check_time()
        key = role + "_calls"
        reserve = (
            len(json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8"))
            + output_limit
            + 1024
        )
        outstanding = sum(v["reserved"] for v in self.data["pending"].values())
        if (
            self.data[key] >= self.limits[key]
            or self.data["tokens"] + outstanding + reserve > self.limits["tokens"]
        ):
            raise BudgetStop(f"{role} call/token budget exhausted")
        ticket = uuid.uuid4().hex
        self.data[key] += 1
        self.data["pending"][ticket] = {"role": role, "reserved": reserve}
        self.save()
        return ticket

    def settle(self, ticket, prompt_tokens, completion_tokens):
        if any(type(v) is not int or v < 0 for v in (prompt_tokens, completion_tokens)):
            raise AccountingError("Invalid usage; reservation retained for inspection")
        request = self.data["pending"].pop(ticket)
        reservation = request["reserved"]
        used = prompt_tokens + completion_tokens
        self.data["tokens"] += used
        self.data["prompt_tokens"] += prompt_tokens
        self.data["completion_tokens"] += completion_tokens
        self.data[request["role"] + "_tokens"] += used
        self.save()
        if used > reservation:
            raise AccountingError(
                "Actual token usage exceeded conservative reservation; inspect before continuing"
            )
