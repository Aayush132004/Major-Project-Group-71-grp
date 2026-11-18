"""Declarative pipeline model (Section 5.1).

A pipeline is a DAG of typed operators written in a DocETL-like YAML.  Every
operator names its inputs; when ``inputs`` is omitted the previous operator
is used, so the common retrieve -> rerank -> generate chain stays short.

Operator kinds
--------------
retrieve       ANN probe (``top_k``, ``index`` hnsw|ivf|flat, optional
               ``where`` => filtered vector probe)
filter         SQL predicate over corpus metadata, evaluated in DuckDB
rerank         cross-encoder rescoring, keeps ``top_k``
map            LLM call per record (extract / summarise), drops "NONE"
generate       one LLM call producing the final answer from the records
sql_generate   text-to-SQL over the schema in context (``fewshot_k``)
sql_repair     execute, and on error/empty result ask the LLM to fix
sql_execute    run the SQL and attach result rows
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import yaml

KINDS = {"retrieve", "filter", "rerank", "map", "generate",
         "sql_generate", "sql_repair", "sql_execute"}
LLM_KINDS = {"map", "generate", "sql_generate", "sql_repair"}
CHEAP_KINDS = KINDS - LLM_KINDS


class PipelineError(ValueError):
    pass


@dataclass
class Operator:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.params:
            d["params"] = copy.deepcopy(self.params)
        if self.prompt:
            d["prompt"] = self.prompt
        if self.inputs:
            d["inputs"] = list(self.inputs)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Operator":
        if d.get("kind") not in KINDS:
            raise PipelineError(f"operator '{d.get('name')}' has unknown kind '{d.get('kind')}'")
        return cls(name=str(d["name"]), kind=d["kind"], params=dict(d.get("params") or {}),
                   prompt=str(d.get("prompt") or ""), inputs=list(d.get("inputs") or []))

    @property
    def uses_llm(self) -> bool:
        return self.kind in LLM_KINDS


@dataclass
class Pipeline:
    name: str
    workload: str
    operators: list[Operator] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)   # directives applied so far

    # ---------------------------------------------------------- structure
    def op(self, name: str) -> Operator:
        for o in self.operators:
            if o.name == name:
                return o
        raise KeyError(name)

    def find(self, kind: str) -> Operator | None:
        return next((o for o in self.operators if o.kind == kind), None)

    def index_of(self, name: str) -> int:
        return next(i for i, o in enumerate(self.operators) if o.name == name)

    def kinds(self) -> list[str]:
        return [o.kind for o in self.topological()]

    def resolved_inputs(self, op: Operator) -> list[str]:
        if op.inputs:
            return op.inputs
        i = self.index_of(op.name)
        return [self.operators[i - 1].name] if i > 0 else []

    def topological(self) -> list[Operator]:
        """Kahn's algorithm; raises on cycles or dangling inputs."""
        names = {o.name for o in self.operators}
        deps = {o.name: set(self.resolved_inputs(o)) for o in self.operators}
        for n, ds in deps.items():
            missing = ds - names
            if missing:
                raise PipelineError(f"operator '{n}' reads unknown input(s) {sorted(missing)}")
        order, ready = [], [o.name for o in self.operators if not deps[o.name]]
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m, ds in deps.items():
                if n in ds:
                    ds.discard(n)
                    if not ds and m not in order and m not in ready:
                        ready.append(m)
        if len(order) != len(self.operators):
            raise PipelineError("pipeline graph has a cycle")
        by_name = {o.name: o for o in self.operators}
        return [by_name[n] for n in order]

    def validate(self) -> None:
        seen = set()
        for o in self.operators:
            if o.name in seen:
                raise PipelineError(f"duplicate operator name '{o.name}'")
            seen.add(o.name)
        self.topological()
        terminal = self.topological()[-1].kind
        if terminal not in {"generate", "sql_execute", "map"}:
            raise PipelineError(f"pipeline must end in an answer-producing operator, got '{terminal}'")

    # ---------------------------------------------------------- editing
    def clone(self) -> "Pipeline":
        return Pipeline(self.name, self.workload,
                        [Operator.from_dict(o.to_dict()) for o in self.operators],
                        list(self.lineage))

    def insert_after(self, anchor: str, op: Operator) -> None:
        """Insert ``op`` downstream of ``anchor`` and re-point anchor's readers."""
        i = self.index_of(anchor)
        repointed = False
        for other in self.operators[i + 1:]:
            if anchor in other.inputs:
                other.inputs = [op.name if x == anchor else x for x in other.inputs]
                repointed = True
        if repointed:
            op.inputs = [anchor]
        self.operators.insert(i + 1, op)

    def insert_before(self, anchor: str, op: Operator) -> None:
        i = self.index_of(anchor)
        anchor_op = self.operators[i]
        if anchor_op.inputs:
            op.inputs = list(anchor_op.inputs)
            anchor_op.inputs = [op.name]
        self.operators.insert(i, op)

    def remove(self, name: str) -> None:
        i = self.index_of(name)
        gone = self.operators.pop(i)
        upstream = gone.inputs or ([self.operators[i - 1].name] if i > 0 else [])
        for other in self.operators:
            if name in other.inputs:
                new_inputs: list[str] = []
                for x in other.inputs:
                    new_inputs.extend(upstream if x == name else [x])
                other.inputs = new_inputs

    def unique_name(self, base: str) -> str:
        names = {o.name for o in self.operators}
        if base not in names:
            return base
        k = 2
        while f"{base}_{k}" in names:
            k += 1
        return f"{base}_{k}"

    # ---------------------------------------------------------- identity
    def structural_dict(self) -> dict:
        return {"ops": [o.to_dict() for o in self.topological()]}

    def fingerprint(self) -> str:
        blob = json.dumps(self.structural_dict(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        parts = []
        for o in self.topological():
            p = o.params
            if o.kind == "retrieve":
                extra = f"k={p.get('top_k')},{p.get('index', 'hnsw')}" + (",where" if p.get("where") else "")
            elif o.kind == "rerank":
                extra = f"k={p.get('top_k')}"
            elif o.kind == "sql_generate":
                extra = f"shots={p.get('fewshot_k', 0)}"
            else:
                extra = ""
            parts.append(f"{o.kind}({extra})" if extra else o.kind)
        return " -> ".join(parts)

    # ---------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {"pipeline": self.name, "workload": self.workload,
                "lineage": list(self.lineage),
                "operators": [o.to_dict() for o in self.operators]}

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, width=100)

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        p = cls(name=d.get("pipeline", "pipeline"), workload=d.get("workload", ""),
                operators=[Operator.from_dict(o) for o in d.get("operators", [])],
                lineage=list(d.get("lineage") or []))
        p.validate()
        return p

    @classmethod
    def from_yaml(cls, text: str) -> "Pipeline":
        return cls.from_dict(yaml.safe_load(text))
