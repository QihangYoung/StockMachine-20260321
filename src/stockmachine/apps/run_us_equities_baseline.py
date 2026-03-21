from __future__ import annotations

import argparse
import json

from stockmachine.research.us_equities_baseline import run_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first US equities baseline experiment.")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--predict-start", default="2024-01-01")
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/us_equities_baseline")
    args = parser.parse_args()

    result = run_baseline_experiment(
        start=args.start,
        end=args.end,
        predict_start=args.predict_start,
        validation_year=args.validation_year,
        top_k=args.top_k,
        horizon=args.horizon,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
