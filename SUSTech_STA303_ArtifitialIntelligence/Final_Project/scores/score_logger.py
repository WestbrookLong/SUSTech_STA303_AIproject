"""
Score Logger for Reinforcement Learning
---------------------------------------
This utility logs episode scores, computes rolling averages,
and plots training progress over time.
"""

import os
import csv
from statistics import mean
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
SCORES_DIR = "./scores"

AVERAGE_SCORE_TO_SOLVE = 475
CONSECUTIVE_RUNS_TO_SOLVE = 100


class ScoreLogger:
    """Logs episode scores and generates plots."""

    def __init__(self, env_name: str, algorithm: str | None = None, hparams: str | None = None):
        self.env_name = env_name
        self.algorithm = algorithm or "default"
        # Optional short text describing hyperparameters for this run (printed on plots)
        self.hparams = hparams
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)

        os.makedirs(SCORES_DIR, exist_ok=True)

        # Build per-run file names with timestamp and algorithm
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"{self.env_name}_{self.algorithm}_{timestamp}"
        self.scores_csv_path = os.path.join(SCORES_DIR, f"{prefix}_scores.csv")
        self.scores_png_path = os.path.join(SCORES_DIR, f"{prefix}_scores.png")
        self.solved_csv_path = os.path.join(SCORES_DIR, f"{prefix}_solved.csv")
        self.solved_png_path = os.path.join(SCORES_DIR, f"{prefix}_solved.png")

    def add_score(self, score: float, run: int):
        """Add a score for a given episode and update logs/plots."""
        self._save_csv(self.scores_csv_path, score)
        self._save_png(
            input_path=self.scores_csv_path,
            output_path=self.scores_png_path,
            x_label="Episodes",
            y_label="Scores",
            average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
            show_goal=True,
            show_trend=True, 
            show_legend=True,
        )

        self.scores.append(score)
        mean_score = mean(self.scores)
        print(f"Scores: (min: {min(self.scores)}, avg: {mean_score:.2f}, max: {max(self.scores)})")

        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solved_in = run - CONSECUTIVE_RUNS_TO_SOLVE
            print(f"Solved in {solved_in} runs, {run} total runs.")
            self._save_csv(self.solved_csv_path, solved_in)
            self._save_png(
                input_path=self.solved_csv_path,
                output_path=self.solved_png_path,
                x_label="Trials",
                y_label="Steps before solved",
                average_of_n_last=None,
                show_goal=False,
                show_trend=False,
                show_legend=False,
            )
            # exit(0) 

    def _save_png(self, input_path, output_path,
                  x_label, y_label,
                  average_of_n_last,
                  show_goal, show_trend, show_legend):
        """Generate a PNG chart for score evolution."""
        if not os.path.exists(input_path):
            return

        x, y = [], []
        with open(input_path, "r") as scores_file:
            reader = csv.reader(scores_file)
            data = list(reader)
            for i, row in enumerate(data):
                x.append(i)
                y.append(float(row[0]))

        plt.subplots()
        plt.plot(x, y, label="Score per Episode")

        if average_of_n_last is not None and len(x) > 0:
            avg_range = min(average_of_n_last, len(x))
            plt.plot(
                x[-avg_range:],
                [np.mean(y[-avg_range:])] * avg_range,
                linestyle="--",
                label=f"Average of last {avg_range}",
            )

        if show_goal:
            # plot the target score line
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=f"Goal ({AVERAGE_SCORE_TO_SOLVE} Avg)")

        if show_trend and len(x) > 1:
            y_trend = []
            current_block_scores = [] 
            
            y_np = np.array(y)
            
            for i in range(len(y_np)):
                current_block_scores.append(y_np[i])
                
                current_avg = np.mean(current_block_scores)
                y_trend.append(current_avg)
                
                if (i + 1) % 100 == 0:
                    current_block_scores = []
            
            plt.plot(x, y_trend, linestyle="-.", label="Trend (100-ep Reset Avg)")

        plt.title(f"{self.env_name} - Training Progress")
        if self.hparams:
            # Put hyperparameters as a small text line above the plot
            plt.suptitle(self.hparams, fontsize=8, y=0.97)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if show_legend:
            plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score: float):
        """Append a score to the given CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([score])
