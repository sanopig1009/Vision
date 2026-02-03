"""A simple terminal-based zombie survival game.

The player controls a hero on a square grid and must avoid zombies
while shooting them from a distance. The game is turn-based and uses
standard input for commands so it runs anywhere Python does.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence


DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class Position:
    row: int
    col: int

    def moved(self, delta: tuple[int, int]) -> "Position":
        dr, dc = delta
        return Position(self.row + dr, self.col + dc)


class ZombieGame:
    def __init__(
        self,
        size: int = 10,
        initial_zombies: int = 3,
        spawn_rate: int = 3,
        max_turns: int = 25,
        seed: int | None = None,
    ) -> None:
        if size < 5:
            raise ValueError("The grid must be at least 5x5.")

        if seed is not None:
            random.seed(seed)

        self.size = size
        self.spawn_rate = max(1, spawn_rate)
        self.max_turns = max_turns
        self.hero = Position(size // 2, size // 2)
        self.turn = 0
        self.score = 0
        self.zombies: List[Position] = []
        self._spawn_initial_zombies(initial_zombies)

    # --- Initialization helpers -------------------------------------------------
    def _spawn_initial_zombies(self, count: int) -> None:
        for _ in range(count):
            self._spawn_zombie()

    def _spawn_zombie(self) -> None:
        """Spawn a zombie along the border of the grid.

        Zombies prefer empty tiles but will not spawn on the hero.
        """

        choices: List[Position] = []
        last_row = self.size - 1
        last_col = self.size - 1

        for idx in range(self.size):
            choices.extend(
                [
                    Position(0, idx),
                    Position(last_row, idx),
                    Position(idx, 0),
                    Position(idx, last_col),
                ]
            )

        random.shuffle(choices)
        for pos in choices:
            if pos != self.hero and pos not in self.zombies:
                self.zombies.append(pos)
                return

    # --- Game loop --------------------------------------------------------------
    def run(self) -> None:
        self._print_intro()
        while self.turn < self.max_turns:
            self.turn += 1
            self._display_state()
            command = input("指令 (w/a/s/d 移動, shoot + 方向射擊, q 離開): ").strip().lower()
            if command in {"q", "quit", "exit"}:
                print("遊戲結束，感謝遊玩！")
                return

            performed = self._handle_command(command)
            if not performed:
                print("無效指令，請重新輸入。")
                self.turn -= 1
                continue

            if self.hero in self.zombies:
                print("你被殭屍抓到了！")
                break

            self._move_zombies()
            if self.hero in self.zombies:
                self._display_state()
                print("你被殭屍抓到了！")
                break

            if self.turn % self.spawn_rate == 0:
                self._spawn_zombie()

            if not self.zombies:
                print("恭喜消滅所有殭屍！")
                break

        else:
            print("你撐過了所有回合，英勇的倖存者！")
            return

        print(f"最終得分：{self.score}")

    # --- Command handling -------------------------------------------------------
    def _handle_command(self, command: str) -> bool:
        if not command:
            return False

        if command in {"w", "a", "s", "d"}:
            self._move_hero(command)
            return True

        if command.startswith("shoot"):
            _, *direction_parts = command.split()
            direction = direction_parts[0] if direction_parts else ""
            return self._shoot(direction)

        shortcuts = {
            "su": "up",
            "sd": "down",
            "sl": "left",
            "sr": "right",
        }
        if command in shortcuts:
            return self._shoot(shortcuts[command])

        return False

    def _move_hero(self, key: str) -> None:
        mapping = {
            "w": "up",
            "s": "down",
            "a": "left",
            "d": "right",
        }
        direction = mapping[key]
        next_pos = self._clamp_to_board(self.hero.moved(DIRECTIONS[direction]))
        self.hero = next_pos

    def _shoot(self, direction: str) -> bool:
        if direction not in DIRECTIONS:
            return False

        delta = DIRECTIONS[direction]
        target_line = self._line_from(self.hero, delta)
        for pos in target_line:
            if pos in self.zombies:
                self.zombies.remove(pos)
                self.score += 10
                print(f"你向{direction}射擊，消滅了一隻殭屍！")
                return True

        print("沒有殭屍在射擊路徑上。")
        return True

    # --- Movement helpers -------------------------------------------------------
    def _move_zombies(self) -> None:
        updated: List[Position] = []
        for zombie in self.zombies:
            dr = self._step_towards(zombie.row, self.hero.row)
            dc = self._step_towards(zombie.col, self.hero.col)
            updated.append(self._clamp_to_board(zombie.moved((dr, dc))))
        self.zombies = updated

    def _step_towards(self, current: int, target: int) -> int:
        if current < target:
            return 1
        if current > target:
            return -1
        return 0

    def _clamp_to_board(self, pos: Position) -> Position:
        return Position(
            max(0, min(self.size - 1, pos.row)),
            max(0, min(self.size - 1, pos.col)),
        )

    def _line_from(self, start: Position, delta: tuple[int, int]) -> Iterable[Position]:
        current = start.moved(delta)
        while self._in_bounds(current):
            yield current
            current = current.moved(delta)

    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.row < self.size and 0 <= pos.col < self.size

    # --- Rendering --------------------------------------------------------------
    def _display_state(self) -> None:
        grid = [["."] * self.size for _ in range(self.size)]
        for zombie in self.zombies:
            grid[zombie.row][zombie.col] = "Z"
        grid[self.hero.row][self.hero.col] = "H"

        header = f"回合 {self.turn}/{self.max_turns} | 殭屍數量：{len(self.zombies)} | 分數：{self.score}"
        print("\n" + header)
        print("".join(["─"] * (self.size + 2)))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("".join(["─"] * (self.size + 2)))

    def _print_intro(self) -> None:
        print(
            """
===========================
    打殭屍小遊戲
===========================
操控方法：
- w/a/s/d 移動主角 (H)
- shoot up/down/left/right 或 su/sd/sl/sr 射擊
- 每個回合殭屍 (Z) 會靠近你，若接觸即失敗
- 撐過所有回合或擊退全部殭屍即可獲勝
            """
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="命令列殭屍生存遊戲")
    parser.add_argument("--size", type=int, default=10, help="棋盤大小 (>=5)")
    parser.add_argument("--zombies", type=int, default=3, help="初始殭屍數")
    parser.add_argument("--spawn-rate", type=int, default=3, help="每隔幾回合生成殭屍")
    parser.add_argument("--turns", type=int, default=25, help="遊戲回合上限")
    parser.add_argument("--seed", type=int, default=None, help="隨機種子方便除錯或比賽")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    game = ZombieGame(
        size=args.size,
        initial_zombies=args.zombies,
        spawn_rate=args.spawn_rate,
        max_turns=args.turns,
        seed=args.seed,
    )
    game.run()


if __name__ == "__main__":
    main()
