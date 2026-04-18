from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.core import Action, SimpleFightingGame
from game.render import GameRenderer
from rl.opponents import AdaptiveScriptedOpponent, PPOOpponent, ScriptedOpponent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play vs scripted AI or PPO model.")
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["scripted", "adaptive", "ppo"],
        default="scripted",
        help="Opponent type: scripted | adaptive | ppo",
    )
    parser.add_argument("--model", type=str, default="", help="Path to a PPO model zip or prefix.")
    parser.add_argument(
        "--adaptive-memory",
        type=str,
        default="checkpoints/adaptive_memory.json",
        help="Path to JSON memory used by adaptive opponent.",
    )
    return parser.parse_args()


def keyboard_to_action() -> Action:
    keys = pygame.key.get_pressed()

    if keys[pygame.K_a] and not keys[pygame.K_d]:
        move = 0
    elif keys[pygame.K_d] and not keys[pygame.K_a]:
        move = 2
    else:
        move = 1

    jump = 1 if keys[pygame.K_w] else 0

    if keys[pygame.K_j]:
        combat = 1
    elif keys[pygame.K_k]:
        combat = 2
    else:
        combat = 0

    return int(move), int(jump), int(combat)


def enemy_observation(game: SimpleFightingGame) -> np.ndarray:
    p = game.player
    e = game.enemy
    return np.asarray(
        [
            e.x / game.arena_width,
            e.y / game.ground_y,
            e.vx / game.move_speed,
            e.vy / game.jump_speed,
            e.hp / game.max_hp,
            1.0 if e.grounded else 0.0,
            e.attack_cooldown / max(1, game.attack_cooldown_frames),
            1.0 if e.blocking else 0.0,
            p.x / game.arena_width,
            p.y / game.ground_y,
            p.vx / game.move_speed,
            p.vy / game.jump_speed,
            p.hp / game.max_hp,
            1.0 if p.grounded else 0.0,
            p.attack_cooldown / max(1, game.attack_cooldown_frames),
            1.0 if p.blocking else 0.0,
            (p.x - e.x) / game.arena_width,
            abs(p.x - e.x) / game.arena_width,
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()

    game = SimpleFightingGame()
    renderer = GameRenderer(game)

    scripted_ai = ScriptedOpponent()
    adaptive_ai = AdaptiveScriptedOpponent(memory_path=args.adaptive_memory)
    model_ai = PPOOpponent(args.model) if args.opponent == "ppo" and args.model else None

    if args.opponent == "ppo" and model_ai is None:
        raise ValueError("Bạn chọn --opponent ppo nhưng chưa truyền --model")

    running = True
    overlay = "Fight!"
    end_delay_frames = 120
    match_logged = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        player_action = keyboard_to_action()
        if args.opponent == "ppo" and model_ai is not None:
            ai_action = model_ai.act(enemy_observation(game))
        elif args.opponent == "adaptive":
            adaptive_ai.record_player_action(player_action)
            ai_action = adaptive_ai.act(game)
        else:
            ai_action = scripted_ai.act(game)

        game.step(player_action, ai_action)

        if game.done:
            overlay = f"Winner: {game.winner} | ESC to quit"
            if args.opponent == "adaptive" and not match_logged:
                adaptive_ai.on_match_end(game.winner)
                match_logged = True
            end_delay_frames -= 1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or end_delay_frames <= 0:
                running = False
        else:
            overlay = "Fight!"

        renderer.draw(overlay_text=overlay)

    renderer.close()


if __name__ == "__main__":
    main()
