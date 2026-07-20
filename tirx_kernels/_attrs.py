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

"""Private helpers for recursively inspecting nested attribute data."""

from collections.abc import Collection, Mapping
from typing import Any


def iter_nested_attr_keys(value: Any, path: str = ""):
    """Yield each mapping key and its dotted path from nested attribute data."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_path = f"{path}{key}"
            yield key, key_path
            yield from iter_nested_attr_keys(item, f"{key_path}.")
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from iter_nested_attr_keys(item, f"{path}{index}.")


def nested_attr_keys(value: Any) -> set[Any]:
    """Return all mapping keys contained in nested attribute data."""

    return {key for key, _ in iter_nested_attr_keys(value)}


def validate_no_nested_attr_keys(
    attrs: Mapping[str, Any], forbidden: Collection[str], *, owner: str
) -> None:
    """Reject forbidden keys at any depth while preserving their diagnostic path."""

    for key, path in iter_nested_attr_keys(attrs):
        if key in forbidden:
            raise ValueError(f"{owner} contains scheduler field {path!r}")


__all__ = ["iter_nested_attr_keys", "nested_attr_keys", "validate_no_nested_attr_keys"]
