"""Reinforcement learning on arithmetic with Serverless RL: sample, score in your own code, train.

    export NEXTRON_API_KEY=nk_live_...
    python examples/live_loop.py rl_...         # a ready RL run: nextron-cloud rl start / wait

Stop the run when you are done: nextron-cloud rl stop rl_... --wait
"""
import random
import re
import sys

from nextron_cloud import NexTron

PROMPT = "What is {a} * {b} + {c}? Think briefly, then give the result as \\boxed{{number}}."


def reward(reply: str, answer: int) -> float:
    found = re.findall(r"\\boxed\{(-?\d+)\}", reply)
    return 1.0 if found and int(found[-1]) == answer else 0.0


def main() -> None:
    client = NexTron()
    with client.rl.connect(sys.argv[1]) as live:
        for step in range(5):
            problems = [(random.randint(11, 99), random.randint(11, 99), random.randint(1, 999)) for _ in range(64)]
            batch = live.sample([PROMPT.format(a=a, b=b, c=c) for a, b, c in problems], n=8, max_tokens=256)
            rewards = [[reward(reply.text or "", a * b + c) for reply in group] for (a, b, c), group in zip(problems, batch)]
            result = live.train(batch, rewards, learning_rate=2e-6)
            print(f"step {step}: reward {result.reward_mean:.3f}, {result.zero_variance_fraction:.0%} of the groups flat; "
                  f"sampling {result.generation_s:.0f} s, training {result.train_s:.0f} s "
                  f"({result.tokens_per_s_per_gpu or 0:.0f} tokens/s/GPU), waiting for this script {result.client_s:.0f} s")
        live.finish()   # saves the model and stops the run


if __name__ == "__main__":
    main()
