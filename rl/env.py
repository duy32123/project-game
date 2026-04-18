from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, MultiDiscrete
except Exception as exc:  # pragma: no cover - import guidance
    raise RuntimeError(
        "Cần cài gymnasium để dùng FightingEnv. Hãy chạy: pip install -r requirements.txt"
    ) from exc

from game.core import Action, SimpleFightingGame
from rl.opponents import PPOOpponent, ScriptedOpponent


class FightingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        opponent: str = "scripted",
        opponent_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.game = SimpleFightingGame()
        self.action_space = MultiDiscrete([3, 2, 3])
        self.observation_space = Box(low=-5.0, high=5.0, shape=(18,), dtype=np.float32)

        self._renderer = None
        self._scripted_opponent = ScriptedOpponent()
        self._ppo_opponent = PPOOpponent(opponent_model_path) if opponent == "ppo" and opponent_model_path else None
        self.opponent = opponent

    def _get_obs(self) -> np.ndarray:
        return np.asarray(self.game.get_observation_vector(), dtype=np.float32)

    def _enemy_action(self) -> Action:
        if self.opponent == "ppo" and self._ppo_opponent is not None:
            obs = self._mirror_observation_for_enemy()
            return self._ppo_opponent.act(obs)
        return self._scripted_opponent.act(self.game)

    def _mirror_observation_for_enemy(self) -> np.ndarray:
        p = self.game.player
        e = self.game.enemy
        mirrored = np.asarray(
            [
                e.x / self.game.arena_width,
                e.y / self.game.ground_y,
                e.vx / self.game.move_speed,
                e.vy / self.game.jump_speed,
                e.hp / self.game.max_hp,
                1.0 if e.grounded else 0.0,
                e.attack_cooldown / max(1, self.game.attack_cooldown_frames),
                1.0 if e.blocking else 0.0,
                p.x / self.game.arena_width,
                p.y / self.game.ground_y,
                p.vx / self.game.move_speed,
                p.vy / self.game.jump_speed,
                p.hp / self.game.max_hp,
                1.0 if p.grounded else 0.0,
                p.attack_cooldown / max(1, self.game.attack_cooldown_frames),
                1.0 if p.blocking else 0.0,
                (p.x - e.x) / self.game.arena_width,
                abs(p.x - e.x) / self.game.arena_width,
            ],
            dtype=np.float32,
        )
        return mirrored

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self.game.reset()
        if seed is not None:
            self._scripted_opponent = ScriptedOpponent(seed=seed)
        obs = self._get_obs()
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action):
        move, jump, combat = [int(x) for x in action]
        enemy_action = self._enemy_action()
        _, info = self.game.step((move, jump, combat), enemy_action)

        obs = self._get_obs()
        reward = self._compute_reward(info)
        terminated = bool(self.game.done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def _compute_reward(self, info: dict[str, Any]) -> float:
        dealt = float(info["player_damage_dealt"])
        taken = float(info["player_damage_taken"])
        distance_before = float(info["distance_before"])
        distance_after = float(info["distance_after"])

        reward = 0.05 * dealt - 0.05 * taken
        reward -= 0.001  # chống câu giờ

        if distance_before > self.game.attack_range * 1.2 and distance_after < distance_before:
            reward += 0.01

        if self.game.done:
            if self.game.winner == "player":
                reward += 3.0
            elif self.game.winner == "enemy":
                reward -= 3.0
            else:
                reward += 0.2

        return reward

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from game.render import GameRenderer

            self._renderer = GameRenderer(self.game)

        overlay = None
        if self.game.done:
            overlay = f"Winner: {self.game.winner}"

        self._renderer.draw(overlay_text=overlay)
        if self.render_mode == "rgb_array":
            return self._renderer.get_rgb_array()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
