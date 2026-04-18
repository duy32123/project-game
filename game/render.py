from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from game.core import SimpleFightingGame


class GameRenderer:
    def __init__(self, game: SimpleFightingGame, width: int = 900, height: int = 420) -> None:
        pygame.init()
        pygame.font.init()
        self.game = game
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Fight Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)

    def draw(self, overlay_text: Optional[str] = None) -> None:
        g = self.game
        self.screen.fill((20, 24, 35))

        pygame.draw.rect(self.screen, (50, 60, 85), (0, g.ground_y, self.width, self.height - g.ground_y))
        pygame.draw.line(self.screen, (180, 180, 180), (0, g.ground_y), (self.width, g.ground_y), 2)

        self._draw_fighter(g.player)
        self._draw_fighter(g.enemy)
        self._draw_hud()

        if overlay_text:
            text = self.font.render(overlay_text, True, (255, 230, 120))
            rect = text.get_rect(center=(self.width // 2, 36))
            self.screen.blit(text, rect)

        pygame.display.flip()
        self.clock.tick(60)

    def get_rgb_array(self) -> np.ndarray:
        pixels = pygame.surfarray.array3d(self.screen)
        return np.transpose(pixels, (1, 0, 2))

    def close(self) -> None:
        pygame.quit()

    def _draw_fighter(self, fighter) -> None:
        g = self.game
        rect = pygame.Rect(
            int(fighter.x - g.fighter_width / 2),
            int(fighter.y - g.fighter_height),
            g.fighter_width,
            g.fighter_height,
        )
        pygame.draw.rect(self.screen, fighter.color, rect, border_radius=6)

        eye_x = rect.centerx + 8 * fighter.facing
        eye_y = rect.y + 18
        pygame.draw.circle(self.screen, (10, 10, 10), (eye_x, eye_y), 3)

        if fighter.blocking:
            shield = pygame.Rect(rect.x - 8, rect.y + 10, rect.width + 16, rect.height - 20)
            pygame.draw.rect(self.screen, (245, 230, 120), shield, width=2, border_radius=8)

    def _draw_hud(self) -> None:
        g = self.game
        self._draw_hp_bar(40, 24, g.player.hp / g.max_hp, "YOU")
        self._draw_hp_bar(self.width - 280, 24, g.enemy.hp / g.max_hp, "AI")

        timer_text = self.font.render(f"{g.remaining_time:05.1f}s", True, (240, 240, 240))
        timer_rect = timer_text.get_rect(center=(self.width // 2, 24))
        self.screen.blit(timer_text, timer_rect)

        controls = "A/D move  W jump  J attack  K block"
        txt = self.small_font.render(controls, True, (200, 200, 200))
        self.screen.blit(txt, (20, self.height - 28))

    def _draw_hp_bar(self, x: int, y: int, ratio: float, label: str) -> None:
        ratio = max(0.0, min(1.0, ratio))
        pygame.draw.rect(self.screen, (70, 70, 70), (x, y, 240, 20), border_radius=6)
        pygame.draw.rect(self.screen, (90, 220, 120), (x, y, int(240 * ratio), 20), border_radius=6)
        txt = self.small_font.render(label, True, (255, 255, 255))
        self.screen.blit(txt, (x, y - 18))
