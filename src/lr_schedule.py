# SPDX-License-Identifier: MIT
"""Linear warmup followed by cosine annealing to zero."""

import math


class WarmUpCosine:
    """LR(step): linear warmup to learning_rate_base, then cosine decay to 0 at total_steps."""

    def __init__(self, learning_rate_base, total_steps, warmup_learning_rate,
                 warmup_steps):
        if total_steps < warmup_steps:
            raise ValueError("total_steps must be >= warmup_steps.")
        if learning_rate_base < warmup_learning_rate:
            raise ValueError("learning_rate_base must be >= warmup_learning_rate.")

        self.learning_rate_base = float(learning_rate_base)
        self.total_steps = int(total_steps)
        self.warmup_learning_rate = float(warmup_learning_rate)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        step = float(step)

        if step > self.total_steps:
            return 0.0

        cos_annealed = math.cos(
            math.pi * (step - self.warmup_steps) /
            float(self.total_steps - self.warmup_steps)
        )
        lr = 0.5 * self.learning_rate_base * (1.0 + cos_annealed)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            lr = slope * step + self.warmup_learning_rate

        return lr
