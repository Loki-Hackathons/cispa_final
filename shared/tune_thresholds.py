"""Optuna threshold tuning template — adapt objective for your task."""

import argparse

import optuna


def objective(trial: optuna.Trial) -> float:
    # Example: tune two thresholds (adapt to your task metric)
    tau_recon = trial.suggest_float("tau_recon", 0.01, 1.0, log=True)
    tau_conf = trial.suggest_float("tau_conf", 0.5, 0.99)

    # TODO: replace with real validation metric (e.g. TPR/(1+5*FPR))
    score = 1.0 - abs(tau_recon - 0.3) - abs(tau_conf - 0.92)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune thresholds with Optuna")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--study-name", default="threshold-tuning")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
