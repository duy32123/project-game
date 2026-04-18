# RL Fight Game (Local 2D + Reinforcement Learning)

Một bộ khung tối giản để bạn:

- chạy game 2D local bằng **Pygame**
- bọc game thành **Gymnasium environment**
- train bot bằng **Stable-Baselines3 PPO**
- chơi **người vs AI** ngay trên máy

## Cấu trúc

```text
rl_fight_game/
├── README.md
├── requirements.txt
├── game/
│   ├── __init__.py
│   ├── core.py
│   └── render.py
├── rl/
│   ├── __init__.py
│   ├── env.py
│   ├── opponents.py
│   └── train.py
└── play/
    ├── __init__.py
    └── play_vs_ai.py
```

## 1) Cài môi trường

```bash
python -m venv .venv
```

### Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

### CMD
```bat
.\.venv\Scripts\activate.bat
```

Sau đó:

```bash
pip install -r requirements.txt
```

## 2) Chạy thử game với bot script

```bash
python -m play.play_vs_ai
```

### Phím điều khiển
- `A` / `D`: đi trái / phải
- `W`: nhảy
- `J`: đánh
- `K`: thủ

## 3) Train bot PPO

```bash
python -m rl.train --timesteps 200000 --n-envs 8 --learning-rate 3e-4
```

Model mặc định sẽ được lưu vào:

```text
checkpoints/ppo_fighter.zip
```

Một lệnh train "kỹ" hơn (nhiều steps + nhiều env + bỏ qua eval cuối nếu chỉ muốn train):

```bash
python -m rl.train --timesteps 1000000 --n-envs 8 --learning-rate 3e-4 --skip-eval
```

## 4) Chơi với bot đã train

```bash
python -m play.play_vs_ai --opponent ppo --model checkpoints/ppo_fighter.zip
```

## 4.1) Chơi với bot "thích nghi" có trí nhớ qua nhiều trận

Bot adaptive sẽ lưu thống kê thói quen của người chơi (đánh/thủ/nhảy/tỉ lệ thắng)
vào file JSON và dùng lại ở những lần chơi sau.

```bash
python -m play.play_vs_ai --opponent adaptive --adaptive-memory checkpoints/adaptive_memory.json
```

Bạn có thể xóa file memory để "reset kinh nghiệm" của bot:

```bash
rm -f checkpoints/adaptive_memory.json
```

## 5) Ghi chú quan trọng

- Đây là **bản khung để train RL nhanh**, không phải game đối kháng hoàn chỉnh.
- Quan sát hiện tại là **vector state**, không dùng ảnh, nên train nhanh hơn.
- Bản đầu dùng **đối thủ scripted** để PPO học nền tảng trước.
- Sau khi bot đánh ổn, bạn có thể mở rộng sang:
  - nhiều đòn đánh
  - projectile
  - combo / stun / knockback
  - self-play
  - nhiều map
  - quan sát từ ảnh (CNN)

## 6) Hướng nâng cấp tiếp theo

### Tăng độ khó
- thêm dash
- thêm light/heavy attack
- thêm block timing
- thêm hurtbox / hitbox riêng

### Tăng chất lượng train
- reward shaping tốt hơn
- random hóa vị trí spawn
- random hóa đối thủ script
- curriculum learning
- self-play với checkpoint cũ

### Ý tưởng reward hiện tại
- +damage gây ra
- -damage nhận vào
- +thắng round
- -thua round
- phạt nhỏ theo thời gian để tránh câu giờ

## 7) Troubleshooting

### `ModuleNotFoundError`
Hãy chạy bằng dạng module từ thư mục gốc:

```bash
python -m play.play_vs_ai
python -m rl.train
```

### Game mở lên nhưng lag khi train
Khi train, env mặc định không render. Đó là đúng. Render chỉ dùng khi chơi thử.

### PPO học chậm
Tăng timesteps lên 500k hoặc 1M, rồi chỉnh reward / hyperparameter.

---
Chúc bạn build tiếp lên bản boss-fight / PvP AI xịn hơn.
