from __future__ import annotations

import json
import random
from collections import deque
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


class AdaptiveScriptedOpponent(ScriptedOpponent):
    """
    Opponent có thể "nhớ" thói quen người chơi qua nhiều trận:
    - lưu thống kê hành vi người chơi vào file JSON
    - đọc lại thống kê ở lần chạy sau để điều chỉnh chiến thuật
    """

    def __init__(self, memory_path: str = "checkpoints/adaptive_memory.json", seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)
        self.memory_path = Path(memory_path)
        self.memory = self._load_memory()
        self._recent_player_actions: deque[str] = deque(maxlen=900)  # khoảng 15 giây @60fps

    def _default_memory(self) -> dict:
        return {
            "matches_played": 0,
            "player_wins": 0,
            "enemy_wins": 0,
            "draws": 0,
            "player_actions": {
                "attack": 0,
                "block": 0,
                "jump": 0,
                "idle": 0,
            },
        }

    def _load_memory(self) -> dict:
        if not self.memory_path.exists():
            return self._default_memory()
        try:
            loaded = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_memory()

        base = self._default_memory()
        for key in ("matches_played", "player_wins", "enemy_wins", "draws"):
            if isinstance(loaded.get(key), int):
                base[key] = loaded[key]
        if isinstance(loaded.get("player_actions"), dict):
            for action_key in base["player_actions"]:
                value = loaded["player_actions"].get(action_key)
                if isinstance(value, int):
                    base["player_actions"][action_key] = value
        return base

    def _save_memory(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_player_action(self, action: Action) -> None:
        _, jump, combat = action
        action_key = "idle"
        if combat == 1:
            action_key = "attack"
        elif combat == 2:
            action_key = "block"
        elif jump == 1:
            action_key = "jump"

        self.memory["player_actions"][action_key] += 1
        self._recent_player_actions.append(action_key)

    def on_match_end(self, winner: str | None) -> None:
        self.memory["matches_played"] += 1
        if winner == "player":
            self.memory["player_wins"] += 1
        elif winner == "enemy":
            self.memory["enemy_wins"] += 1
        else:
            self.memory["draws"] += 1
        self._save_memory()

    def _player_attack_ratio(self) -> float:
        if not self._recent_player_actions:
            total = sum(self.memory["player_actions"].values())
            if total <= 0:
                return 0.0
            return self.memory["player_actions"]["attack"] / total

        recent_attack = sum(1 for action in self._recent_player_actions if action == "attack")
        return recent_attack / len(self._recent_player_actions)

    def _player_block_ratio(self) -> float:
        if not self._recent_player_actions:
            total = sum(self.memory["player_actions"].values())
            if total <= 0:
                return 0.0
            return self.memory["player_actions"]["block"] / total

        recent_block = sum(1 for action in self._recent_player_actions if action == "block")
        return recent_block / len(self._recent_player_actions)

    def _player_win_ratio(self) -> float:
        matches = max(1, self.memory["matches_played"])
        return self.memory["player_wins"] / matches

    def act(self, game: SimpleFightingGame) -> Action:
        me = game.enemy
        other = game.player
        distance = abs(me.x - other.x)

        player_attack_ratio = self._player_attack_ratio()
        player_block_ratio = self._player_block_ratio()
        player_win_ratio = self._player_win_ratio()

        # Base policy
        attack_prob = 0.55
        block_prob = 0.23
        reposition_prob = 0.22

        # Người chơi tấn công nhiều -> bot thủ nhiều hơn để counter
        if player_attack_ratio > 0.32:
            block_prob += 0.18
            attack_prob -= 0.08
            reposition_prob -= 0.10

        # Người chơi thủ nhiều -> bot ép khoảng cách và đánh thường xuyên hơn
        if player_block_ratio > 0.28:
            attack_prob += 0.10
            reposition_prob += 0.08
            block_prob -= 0.18

        # Nếu người chơi đang thắng nhiều trận, bot sẽ chủ động hơn
        if player_win_ratio > 0.55:
            attack_prob += 0.08
            block_prob -= 0.04
            reposition_prob -= 0.04

        # Normalize về [0, 1]
        attack_prob = max(0.1, min(0.8, attack_prob))
        block_prob = max(0.05, min(0.75, block_prob))
        reposition_prob = max(0.05, min(0.7, reposition_prob))

        total = attack_prob + block_prob + reposition_prob
        attack_prob /= total
        block_prob /= total
        reposition_prob /= total

        move = 1
        jump = 0
        combat = 0

        if distance > game.attack_range * 1.25:
            move = 0 if me.x > other.x else 2
            if self.rng.random() < 0.03 and me.grounded:
                jump = 1
        else:
            roll = self.rng.random()
            if roll < attack_prob:
                combat = 1
            elif roll < attack_prob + block_prob:
                combat = 2
            else:
                move = 0 if me.x > other.x else 2

            if distance < game.fighter_width * 1.2 and self.rng.random() < reposition_prob:
                move = 2 if me.x > other.x else 0

        return int(move), int(jump), int(combat)
