# Vision

這個倉庫現在包含了一個簡單的終端殭屍生存遊戲 `zombie_game.py`。

## 遊戲玩法
- 透過 `python zombie_game.py` 進入遊戲。
- 使用 `w/a/s/d` 移動主角 (H)。
- 使用 `shoot up/down/left/right` 或 `su/sd/sl/sr` 往指定方向射擊殭屍 (Z)。
- 每個回合殭屍會朝主角靠近；被接觸即失敗。
- 撐過所有回合或消滅全部殭屍即可獲勝，擊殺殭屍會增加分數。

## 進階設定
可以透過參數客製化遊戲難度：
```bash
python zombie_game.py --size 12 --zombies 4 --spawn-rate 2 --turns 30 --seed 1
```

## 需求
- Python 3.11 以上，無需額外套件。
