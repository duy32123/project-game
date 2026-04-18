from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

Action = Tuple[int, int, int]  # (move, jump, combat)
# move:   0 left, 1 idle, 2 right
# jump:   0 no, 1 yes
# combat: 0 none, 1 attack, 2 block


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class FighterState:
    name: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hp: float = 100.0
    facing: int = 1  # 1 right, -1 left
    grounded: bool = True
    attack_cooldown: int = 0
    blocking: bool = False
    attack_request: bool = False
    last_hit_frame: int = -9999
    color: Tuple[int, int, int] = (220, 220, 220)

    def reset_transients(self) -> None:
        self.blocking = False
        self.attack_request = False


@dataclass
class SimpleFightingGame:
    arena_width: int = 900
    arena_height: int = 420
    ground_y: int = 320
    fighter_width: int = 40
    fighter_height: int = 72
    move_speed: float = 260.0
    jump_speed: float = 560.0
    gravity: float = 1500.0
    max_hp: float = 100.0
    attack_range: float = 90.0
    attack_damage: float = 10.0
    attack_cooldown_frames: int = 24
    max_frames: int = 60 * 45
    dt: float = 1.0 / 60.0
    frame: int = 0
    winner: str | None = None
    done: bool = False
    player: FighterState = field(init=False)
    enemy: FighterState = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> Dict[str, float]:
        self.frame = 0
        self.done = False
        self.winner = None
        self.player = FighterState(
            name="player",
            x=220.0,
            y=float(self.ground_y),
            hp=self.max_hp,
            facing=1,
            color=(80, 180, 255),
        )
        self.enemy = FighterState(
            name="enemy",
            x=float(self.arena_width - 220),
            y=float(self.ground_y),
            hp=self.max_hp,
            facing=-1,
            color=(255, 120, 120),
        )
        return self.get_observation()

    def get_observation(self) -> Dict[str, float]:
        p = self.player
        e = self.enemy
        distance = e.x - p.x
        obs = {
            "player_x": p.x / self.arena_width,
            "player_y": p.y / self.ground_y,
            "player_vx": p.vx / self.move_speed,
            "player_vy": p.vy / self.jump_speed,
            "player_hp": p.hp / self.max_hp,
            "player_grounded": 1.0 if p.grounded else 0.0,
            "player_cooldown": p.attack_cooldown / max(1, self.attack_cooldown_frames),
            "player_blocking": 1.0 if p.blocking else 0.0,
            "enemy_x": e.x / self.arena_width,
            "enemy_y": e.y / self.ground_y,
            "enemy_vx": e.vx / self.move_speed,
            "enemy_vy": e.vy / self.jump_speed,
            "enemy_hp": e.hp / self.max_hp,
            "enemy_grounded": 1.0 if e.grounded else 0.0,
            "enemy_cooldown": e.attack_cooldown / max(1, self.attack_cooldown_frames),
            "enemy_blocking": 1.0 if e.blocking else 0.0,
            "relative_x": distance / self.arena_width,
            "distance_abs": abs(distance) / self.arena_width,
        }
        return obs

    def get_observation_vector(self) -> list[float]:
        obs = self.get_observation()
        return [float(v) for v in obs.values()]

    def step(self, player_action: Action, enemy_action: Action) -> tuple[Dict[str, float], Dict[str, float]]:
        if self.done:
            return self.get_observation(), {
                "done": True,
                "winner": self.winner,
                "player_damage_dealt": 0.0,
                "player_damage_taken": 0.0,
                "distance_before": 0.0,
                "distance_after": 0.0,
            }

        self.frame += 1
        p = self.player
        e = self.enemy

        distance_before = abs(e.x - p.x)
        p.reset_transients()
        e.reset_transients()

        self._update_facing()
        self._apply_action(p, e, player_action)
        self._apply_action(e, p, enemy_action)

        self._simulate_physics(p)
        self._simulate_physics(e)
        self._resolve_overlap()
        player_dealt, enemy_dealt = self._resolve_attacks()

        distance_after = abs(e.x - p.x)

        if p.attack_cooldown > 0:
            p.attack_cooldown -= 1
        if e.attack_cooldown > 0:
            e.attack_cooldown -= 1

        if p.hp <= 0 and e.hp <= 0:
            self.done = True
            self.winner = "draw"
        elif e.hp <= 0:
            self.done = True
            self.winner = "player"
        elif p.hp <= 0:
            self.done = True
            self.winner = "enemy"
        elif self.frame >= self.max_frames:
            self.done = True
            if p.hp > e.hp:
                self.winner = "player"
            elif e.hp > p.hp:
                self.winner = "enemy"
            else:
                self.winner = "draw"

        obs = self.get_observation()
        info = {
            "done": self.done,
            "winner": self.winner or "",
            "player_damage_dealt": float(player_dealt),
            "player_damage_taken": float(enemy_dealt),
            "distance_before": float(distance_before),
            "distance_after": float(distance_after),
        }
        return obs, info

    def _update_facing(self) -> None:
        if self.enemy.x >= self.player.x:
            self.player.facing = 1
            self.enemy.facing = -1
        else:
            self.player.facing = -1
            self.enemy.facing = 1

    def _apply_action(self, actor: FighterState, target: FighterState, action: Action) -> None:
        move, jump, combat = action

        speed_scale = 0.45 if combat == 2 else 1.0

        if move == 0:
            actor.vx = -self.move_speed * speed_scale
        elif move == 2:
            actor.vx = self.move_speed * speed_scale
        else:
            actor.vx = 0.0

        if jump == 1 and actor.grounded:
            actor.vy = -self.jump_speed
            actor.grounded = False

        if combat == 2:
            actor.blocking = True

        if combat == 1 and actor.attack_cooldown <= 0:
            actor.attack_request = True
            actor.attack_cooldown = self.attack_cooldown_frames

        if target.x > actor.x:
            actor.facing = 1
        else:
            actor.facing = -1

    def _simulate_physics(self, fighter: FighterState) -> None:
        fighter.vy += self.gravity * self.dt
        fighter.x += fighter.vx * self.dt
        fighter.y += fighter.vy * self.dt

        half_w = self.fighter_width / 2
        fighter.x = clamp(fighter.x, half_w, self.arena_width - half_w)

        if fighter.y >= self.ground_y:
            fighter.y = float(self.ground_y)
            fighter.vy = 0.0
            fighter.grounded = True
        else:
            fighter.grounded = False

    def _resolve_overlap(self) -> None:
        min_gap = float(self.fighter_width)
        gap = abs(self.enemy.x - self.player.x)
        if gap < min_gap:
            push = (min_gap - gap) / 2.0 + 0.01
            if self.player.x < self.enemy.x:
                self.player.x -= push
                self.enemy.x += push
            else:
                self.player.x += push
                self.enemy.x -= push
            half_w = self.fighter_width / 2
            self.player.x = clamp(self.player.x, half_w, self.arena_width - half_w)
            self.enemy.x = clamp(self.enemy.x, half_w, self.arena_width - half_w)

    def _resolve_attacks(self) -> tuple[float, float]:
        player_dealt = 0.0
        enemy_dealt = 0.0

        if self.player.attack_request:
            damage = self._compute_damage(self.player, self.enemy)
            self.enemy.hp = max(0.0, self.enemy.hp - damage)
            player_dealt += damage

        if self.enemy.attack_request:
            damage = self._compute_damage(self.enemy, self.player)
            self.player.hp = max(0.0, self.player.hp - damage)
            enemy_dealt += damage

        self.player.attack_request = False
        self.enemy.attack_request = False
        return player_dealt, enemy_dealt

    def _compute_damage(self, attacker: FighterState, defender: FighterState) -> float:
        dx = abs(attacker.x - defender.x)
        dy = abs(attacker.y - defender.y)
        in_range = dx <= self.attack_range and dy <= self.fighter_height * 0.6

        if not in_range:
            return 0.0

        blocked = defender.blocking and defender.facing == -attacker.facing
        if blocked:
            return self.attack_damage * 0.25
        return self.attack_damage

    @property
    def remaining_time(self) -> float:
        remaining_frames = max(0, self.max_frames - self.frame)
        return remaining_frames * self.dt
