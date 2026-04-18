from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np

from game.core import Action, SimpleFightingGame


class ScriptedOpponent:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def act(self, game: SimpleFightingGame) -> Action:
        me = game.enemy
        other = game.player
        distance = abs(me.x - other.x)
        move = 1
        jump = 0
        combat = 0

        if distance > game.attack_range * 1.25:
            move = 0 if me.x > other.x else 2
            if self.rng.random() < 0.02 and me.grounded:
                jump = 1
        else:
            roll = self.rng.random()
            if roll < 0.55:
                combat = 1
            elif roll < 0.78:
                combat = 2
            else:
                move = 0 if me.x > other.x else 2

            if distance < game.fighter_width * 1.2 and self.rng.random() < 0.35:
                move = 2 if me.x > other.x else 0

        return int(move), int(jump), int(combat)


class PPOOpponent:
    def __init__(self, model_path: str) -> None:
        try:
            from stable_baselines3 import PPO
        except Exception as exc:  # pragma: no cover - optional dependency runtime path
            raise RuntimeError(
                "stable-baselines3 chưa được cài. Hãy pip install -r requirements.txt"
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        self.model = PPO.load(str(path))

    def act(self, observation: np.ndarray) -> Action:
        action, _ = self.model.predict(observation, deterministic=True)
        move, jump, combat = [int(x) for x in action]
        return move, jump, combat
