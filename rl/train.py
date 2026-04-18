from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env import FightingEnv


def build_vec_env(n_envs: int = 1):
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

    def _factory():
        env = FightingEnv(render_mode=None, opponent="scripted")
        return Monitor(env)

    env_fns = [_factory for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


class EpisodeStatsCallback:
    """Lazy wrapper so importing this file does not fail before SB3 is installed."""

    @staticmethod
    def build(verbose: int = 0):
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

        class _EpisodeStatsCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__(verbose=verbose)
                self.episode_count = 0
                self.last_reward = 0.0

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                rewards = self.locals.get("rewards", [])

                for idx, done in enumerate(dones):
                    if not done:
                        continue
                    self.episode_count += 1
                    info = infos[idx] if idx < len(infos) else {}
                    reward = float(rewards[idx]) if idx < len(rewards) else float("nan")
                    self.last_reward = reward
                    if self.verbose > 0:
                        winner = info.get("winner") if isinstance(info, dict) else None
                        print(f"[episode {self.episode_count}] winner={winner} reward={reward:.3f}")
                return True

        return _EpisodeStatsCallback(verbose=verbose)


def train_ppo(
    total_timesteps: int = 200_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "ppo_fighter",
):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(n_envs=n_envs)
    callback = EpisodeStatsCallback.build(verbose=1)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(checkpoint_path / "tb_logs"),
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    save_path = checkpoint_path / f"{model_name}.zip"
    model.save(str(save_path))
    print(f"Saved model to: {save_path}")
    return model, vec_env


def evaluate_policy_rollout(model_path: str, episodes: int = 3, render: bool = False) -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for training utilities.") from exc

    env = FightingEnv(render_mode="human" if render else None, opponent="scripted")
    model = PPO.load(model_path)

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        winner = info.get("winner", "") if isinstance(info, dict) else ""
        print(f"[eval episode {ep}] winner={winner} total_reward={total_reward:.3f}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for the 2D fight game.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps.")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save model.")
    parser.add_argument("--model-name", type=str, default="ppo_fighter", help="Output model name (without .zip).")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episodes for post-training evaluation.")
    parser.add_argument("--eval-render", action="store_true", help="Render during evaluation.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training rollout evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, vec_env = train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.save_dir,
        model_name=args.model_name,
    )
    vec_env.close()

    if not args.skip_eval:
        model_path = str(Path(args.save_dir) / f"{args.model_name}.zip")
        evaluate_policy_rollout(
            model_path=model_path,
            episodes=args.eval_episodes,
            render=args.eval_render,
        )


if __name__ == "__main__":
    main()
