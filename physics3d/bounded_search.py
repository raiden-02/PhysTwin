"""Deterministic bounded derivative-free search for the P5 fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    id: str
    lower_bound: float
    upper_bound: float
    initial: float
    unit: str


@dataclass(frozen=True)
class SearchResult:
    values: tuple[float, ...]
    objective: float
    initial_objective: float
    objective_evaluations: int
    generations: int
    coordinate_iterations: int


def bounded_differential_search(
    objective: Callable[[tuple[float, ...]], float],
    parameters: Sequence[ParameterSpec],
    *,
    seed: int,
    population_size: int = 12,
    generations: int = 12,
    coordinate_iterations: int = 24,
    differential_weight: float = 0.75,
    crossover_rate: float = 0.9,
) -> SearchResult:
    """Search a finite box with fixed-seed DE and coordinate refinement."""

    if not parameters:
        raise ValueError("parameters must not be empty")
    dimension = len(parameters)
    if population_size < max(4, dimension + 1):
        raise ValueError("population_size is too small")
    if generations < 0 or coordinate_iterations < 0:
        raise ValueError("iteration budgets must be non-negative")
    for parameter in parameters:
        if not parameter.lower_bound < parameter.upper_bound:
            raise ValueError(f"{parameter.id}: lower_bound must be < upper_bound")
        if not parameter.lower_bound <= parameter.initial <= parameter.upper_bound:
            raise ValueError(f"{parameter.id}: initial must be inside bounds")

    lower = np.asarray([item.lower_bound for item in parameters], dtype=np.float64)
    span = np.asarray(
        [item.upper_bound - item.lower_bound for item in parameters],
        dtype=np.float64,
    )

    def decode(unit: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in lower + np.clip(unit, 0.0, 1.0) * span)

    evaluations = 0

    def evaluate(unit: np.ndarray) -> float:
        nonlocal evaluations
        value = float(objective(decode(unit)))
        evaluations += 1
        if not np.isfinite(value):
            raise ValueError("objective must return a finite value")
        return value

    rng = np.random.default_rng(seed)
    population = rng.random((population_size, dimension), dtype=np.float64)
    population[0] = np.asarray(
        [
            (item.initial - item.lower_bound)
            / (item.upper_bound - item.lower_bound)
            for item in parameters
        ],
        dtype=np.float64,
    )
    scores = np.asarray([evaluate(candidate) for candidate in population])
    initial_objective = float(scores[0])

    for _ in range(generations):
        for index in range(population_size):
            candidates = np.delete(np.arange(population_size), index)
            a, b, c = rng.choice(candidates, size=3, replace=False)
            mutant = np.clip(
                population[a]
                + differential_weight * (population[b] - population[c]),
                0.0,
                1.0,
            )
            mask = rng.random(dimension) < crossover_rate
            mask[int(rng.integers(0, dimension))] = True
            trial = np.where(mask, mutant, population[index])
            score = evaluate(trial)
            if score < scores[index]:
                population[index] = trial
                scores[index] = score

    best_index = int(np.argmin(scores))
    best = population[best_index].copy()
    best_score = float(scores[best_index])
    step = 0.1
    completed_coordinate_iterations = 0
    for _ in range(coordinate_iterations):
        completed_coordinate_iterations += 1
        improved = False
        for axis in range(dimension):
            for direction in (-1.0, 1.0):
                trial = best.copy()
                trial[axis] = np.clip(trial[axis] + direction * step, 0.0, 1.0)
                if trial[axis] == best[axis]:
                    continue
                score = evaluate(trial)
                if score < best_score:
                    best = trial
                    best_score = score
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break

    return SearchResult(
        values=decode(best),
        objective=best_score,
        initial_objective=initial_objective,
        objective_evaluations=evaluations,
        generations=generations,
        coordinate_iterations=completed_coordinate_iterations,
    )
