from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env import FightingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for the 2D fight game.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model.")
    parser.add_argument("--model-name", type=str, default="ppo_fighter", help="Output model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 chưa được cài. Hãy pip install -r requirements.txt"
        ) from exc

    env = Monitor(FightingEnv(render_mode=None, opponent="scripted"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log="runs",
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / args.model_name
    model.save(str(out_path))

    env.close()
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
