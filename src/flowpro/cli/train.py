import argparse
from flowpro.training.mixer import batch_counts


def main():
    p = argparse.ArgumentParser(description="Inspect FlowPRO offline RPRO round configuration")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    print(batch_counts(args.batch_size, args.round))

