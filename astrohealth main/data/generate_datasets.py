"""Generate reproducible training, testing, and nominal-baseline CSV datasets for ASTROHEALTH."""
import csv
import os
import numpy as np

FEATS = [
    "hr", "sys", "dia", "spo2", "temp", "sleep", "stress", "fatigue", "mood",
    "bone_loss", "muscle_loss", "wbc", "crp", "rad",
]
LABELS = ["low", "moderate", "high"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def score(row):
    f = dict(zip(FEATS, row))
    p = 0
    p += 2 * (f["hr"] > 100 or f["hr"] < 50) + 2 * (f["sys"] > 140) + 3 * (f["spo2"] < 92) + 2 * (f["temp"] > 38)
    p += (f["sleep"] < 5.5) + (f["stress"] > 7) + (f["fatigue"] > 7) + (f["mood"] < 3.5)
    p += 2 * (f["bone_loss"] > 6) + (f["muscle_loss"] > 15) + (f["wbc"] > 11 or f["wbc"] < 3.5) + 2 * (f["crp"] > 5)
    return p


def synth(n, rng, k=1.0):
    g = lambda m, s: rng.normal(m, s * k, n)
    X = np.c_[
        g(68, 14), g(120, 14), g(76, 9), np.minimum(g(97, 2.2), 100), g(36.8, 0.5), g(6.8, 1.2),
        np.clip(g(4, 2), 0, 10), np.clip(g(4, 2), 0, 10), np.clip(g(6, 2), 0, 10),
        np.abs(g(2, 3)), np.abs(g(5, 5)), g(7, 2.5), np.abs(g(2, 2.5)), np.abs(g(0.6, 0.3)),
    ]
    return np.round(X, 2)


def is_extreme(row):
    f = dict(zip(FEATS, row))
    return int(
        f["hr"] > 105 or f["hr"] < 46 or f["sys"] > 148 or f["spo2"] < 91
        or f["temp"] > 38.3 or f["wbc"] > 12.5 or f["crp"] > 7.5 or f["rad"] >= 5.0
    )


def write_csv(filename, X, include_labels=True):
    path = os.path.join(OUT_DIR, filename)
    header = FEATS + (["risk_class", "risk_label", "is_anomaly"] if include_labels else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in X:
            vals = [f"{float(v):.2f}" for v in r]
            if include_labels:
                s = score(r)
                cls = 0 if s < 2 else 1 if s < 5 else 2
                vals.extend([cls, LABELS[cls], is_extreme(r)])
            w.writerow(vals)
    return path


def main():
    rng = np.random.default_rng(7)
    X_all = synth(8000, rng, k=1.0)
    # Inject 400 realistic high-strain / solar-event samples before shuffling train/test split
    X_stress = synth(400, rng, k=1.25)
    X_stress[:, 0] += rng.uniform(18, 36, 400)   # elevated HR
    X_stress[:, 1] += rng.uniform(15, 28, 400)   # elevated SBP
    X_stress[:, 3] = np.maximum(84.0, X_stress[:, 3] - rng.uniform(2, 6, 400))  # lower SpO2
    X_stress[:, 13] += rng.choice([0.0, 12.0], size=400, p=[0.85, 0.15])        # solar bursts
    X_stress = np.round(X_stress, 2)

    X_combined = np.vstack([X_all, X_stress])
    rng.shuffle(X_combined)
    split = int(0.8 * len(X_combined))  # 6720 train, 1680 test
    X_train, X_test = X_combined[:split], X_combined[split:]
    X_base = synth(3000, rng, k=0.6)

    write_csv("train_telemetry.csv", X_train, include_labels=True)
    write_csv("test_telemetry.csv", X_test, include_labels=True)
    write_csv("baseline_nominal.csv", X_base, include_labels=False)
    print(f"Generated train={len(X_train)}, test={len(X_test)}, baseline={len(X_base)} in {OUT_DIR}")


if __name__ == "__main__":
    main()
