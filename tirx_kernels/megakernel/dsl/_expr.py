# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Private expression language used while normalizing MoE lowering plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tvm.script import tirx as T


class UnresolvedExpression(ValueError):
    """Raised when an expression is not available in an evaluation environment."""


class Expr:
    """Base class for the deliberately narrow private MoE expression language."""

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def lower(self, env: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def to_data(self) -> object:
        raise NotImplementedError

    def __add__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("+", self, as_expr(other))

    def __radd__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("+", as_expr(other), self)

    def __sub__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("-", self, as_expr(other))

    def __rsub__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("-", as_expr(other), self)

    def __mul__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("*", self, as_expr(other))

    def __rmul__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("*", as_expr(other), self)

    def __floordiv__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("//", self, as_expr(other))

    def __rfloordiv__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("//", as_expr(other), self)

    def __mod__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("%", self, as_expr(other))

    def __rmod__(self, other: ExprLike) -> BinaryExpr:
        return BinaryExpr("%", as_expr(other), self)


ExprLike = Expr | int


def _lookup(env: Mapping[str, Any], namespace: str, name: str) -> Any:
    values = env.get(namespace)
    if isinstance(values, Mapping) and name in values:
        return values[name]
    if name in env:
        return env[name]
    raise UnresolvedExpression(f"{namespace} value {name!r} is not available")


@dataclass(frozen=True)
class ConstExpr(Expr):
    value: int

    def __post_init__(self):
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("ConstExpr only accepts integer values")

    def evaluate(self, env: Mapping[str, Any]) -> int:
        del env
        return self.value

    def lower(self, env: Mapping[str, Any]) -> int:
        del env
        return self.value

    def to_data(self) -> object:
        return {"const": self.value}


@dataclass(frozen=True)
class VarExpr(Expr):
    name: str

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return _lookup(env, "vars", self.name)

    def lower(self, env: Mapping[str, Any]) -> Any:
        return _lookup(env, "vars", self.name)

    def to_data(self) -> object:
        return {"var": self.name}


@dataclass(frozen=True)
class TileIndexExpr(Expr):
    task: str
    axis: int

    def __post_init__(self):
        if self.axis not in (0, 1, 2):
            raise ValueError(f"tile axis must be 0, 1, or 2; got {self.axis}")

    def _resolve(self, env: Mapping[str, Any]) -> Any:
        tiles = env.get("tiles")
        if isinstance(tiles, Mapping):
            if self.task in tiles:
                return tiles[self.task][self.axis]
            if (self.task, self.axis) in tiles:
                return tiles[(self.task, self.axis)]
        raise UnresolvedExpression(f"tile index ({self.task!r}, axis={self.axis}) is not available")

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self._resolve(env)

    def lower(self, env: Mapping[str, Any]) -> Any:
        return self._resolve(env)

    def to_data(self) -> object:
        return {"tile": [self.task, self.axis]}


@dataclass(frozen=True)
class ScalarLoadExpr(Expr):
    tensor: str
    index: Expr
    dtype: str
    tensor_shape: tuple[Expr, ...]

    def _load(self, env: Mapping[str, Any], *, lower: bool) -> Any:
        tensor = _lookup(env, "tensors", self.tensor)
        index = self.index.lower(env) if lower else self.index.evaluate(env)
        return tensor[index]

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self._load(env, lower=False)

    def lower(self, env: Mapping[str, Any]) -> Any:
        return self._load(env, lower=True)

    def to_data(self) -> object:
        return {"scalar_load": [self.tensor, self.index.to_data()]}


@dataclass(frozen=True)
class BinaryExpr(Expr):
    op: str
    lhs: Expr
    rhs: Expr

    def __post_init__(self):
        if self.op not in {"+", "-", "*", "//", "%"}:
            raise ValueError(f"unsupported binary operator: {self.op!r}")

    def _apply(self, lhs: Any, rhs: Any, *, lower: bool) -> Any:
        if self.op == "+":
            return lhs + rhs
        if self.op == "-":
            return lhs - rhs
        if self.op == "*":
            return lhs * rhs
        if self.op == "//":
            return lhs // rhs
        return lhs % rhs

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        return self._apply(self.lhs.evaluate(env), self.rhs.evaluate(env), lower=False)

    def lower(self, env: Mapping[str, Any]) -> Any:
        return self._apply(self.lhs.lower(env), self.rhs.lower(env), lower=True)

    def to_data(self) -> object:
        return {self.op: [self.lhs.to_data(), self.rhs.to_data()]}


@dataclass(frozen=True)
class CeilDivExpr(Expr):
    lhs: Expr
    rhs: Expr

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        lhs = self.lhs.evaluate(env)
        rhs = self.rhs.evaluate(env)
        return (lhs + rhs - 1) // rhs

    def lower(self, env: Mapping[str, Any]) -> Any:
        lhs = self.lhs.lower(env)
        rhs = self.rhs.lower(env)
        return T.truncdiv(lhs + rhs - 1, rhs)

    def to_data(self) -> object:
        return {"ceildiv": [self.lhs.to_data(), self.rhs.to_data()]}


def as_expr(value: ExprLike) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        if callable(value):
            raise TypeError("callables are not valid lowering expressions")
        raise TypeError(f"expected an integer or Expr, got {type(value).__name__}")
    return ConstExpr(value)


def ceildiv(lhs: ExprLike, rhs: ExprLike) -> CeilDivExpr:
    return CeilDivExpr(as_expr(lhs), as_expr(rhs))


def walk_expr(expr: Expr):
    """Yield an expression and all of its children in preorder."""

    yield expr
    if isinstance(expr, BinaryExpr | CeilDivExpr):
        yield from walk_expr(expr.lhs)
        yield from walk_expr(expr.rhs)
    elif isinstance(expr, ScalarLoadExpr):
        yield from walk_expr(expr.index)


def evaluate_static(expr: Expr, env: Mapping[str, Any], *, label: str) -> int:
    """Evaluate and type-check a host-compile-time integer expression."""

    try:
        value = expr.evaluate(env)
    except (KeyError, TypeError, UnresolvedExpression) as err:
        raise ValueError(f"{label} is not statically evaluable: {expr.to_data()}") from err
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must evaluate to an integer; got {value!r}")
    return value
