"""
=============================================================================
EEG COGNITIVE STATE CLASSIFICATION — SEED DATASET
=============================================================================

"""



ROOT_DIR       = r"D:\IML_LAB_PROJECT\SEED_Dataset(preprocessed eeg)"
FEATURE_FOLDER = "ExtractedFeatures_1s"
OUTPUT_DIR     = r"D:\IML_LAB_PROJECT\results_lightweight"

N_FOLDS        = 5
WINDOW_SEC     = 10
STEP_SEC       = 5
N_FEATURES_SEL = 80
RANDOM_SEED    = 42


import os, glob, time, warnings, pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved to: {OUTPUT_DIR}")


TRIAL_LABELS_DEFAULT = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
CLASS_NAMES          = ["Negative", "Neutral", "Positive"]
BAND_NAMES           = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
N_CHANNELS = 62
N_BANDS    = 5

PALETTE = {
    "LinearSVC"        : "#e74c3c",
    "LDA"              : "#4a8fd4",
    "Random Forest"    : "#f0a050",
    "Hist Grad Boost"  : "#5bbfa8",
    "KNN"              : "#7ec8c8",
    "Logistic Reg"     : "#b07cc6",
}



def load_metadata(root_dir):
    meta = {}
    gender_file = os.path.join(root_dir, "subject-id-gender-seed.txt")
    if os.path.exists(gender_file):
        try:
            rows = []
            with open(gender_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        rows.append({"subject_id": int(parts[0]),
                                     "gender": parts[1]})
            if rows:
                meta["subject_gender"] = pd.DataFrame(rows)
                print(f"  Loaded gender info for {len(rows)} subjects.")
        except Exception as e:
            print(f"  Warning: {e}")
    return meta



def compute_band_ratio_features(de_window):
    """
    Band power ratio features from DE window (WINDOW_SEC, N_CH, N_BANDS).
    - Each band / total power  : 5 per channel
    - Theta/Alpha, Beta/Alpha, Gamma/Beta : 3 per channel
    """
    mean_de = de_window.mean(axis=0)
    total   = mean_de.sum(axis=1, keepdims=True) + 1e-9
    band_ratios  = mean_de / total
    theta_alpha  = mean_de[:, 1] / (mean_de[:, 2] + 1e-9)
    beta_alpha   = mean_de[:, 3] / (mean_de[:, 2] + 1e-9)
    gamma_beta   = mean_de[:, 4] / (mean_de[:, 3] + 1e-9)
    cross        = np.stack([theta_alpha, beta_alpha, gamma_beta], axis=1)
    return np.concatenate([band_ratios.flatten(), cross.flatten()])


def compute_stat_features(window):
    """
    Statistical features over time axis: mean, std, skewness, kurtosis.
    window: (WINDOW_SEC, n_features)  →  4 * n_features output
    """
    return np.concatenate([
        window.mean(axis=0),
        window.std(axis=0),
        stats.skew(window, axis=0),
        stats.kurtosis(window, axis=0),
    ])


def extract_subject_windows(mat, trial_labels):
    windows, labels = [], []
    for t in range(15):
        de   = mat.get(f"de_LDS{t+1}")
        psd  = mat.get(f"psd_LDS{t+1}")
        dasm = mat.get(f"dasm_LDS{t+1}")
        rasm = mat.get(f"rasm_LDS{t+1}")
        if de is None:
            continue

        T     = de.shape[1]
        de_t  = de.transpose(1, 0, 2)
        psd_t = psd.transpose(1, 0, 2) if psd is not None else None
        dm_t  = dasm.transpose(1, 0, 2) if dasm is not None else None
        rm_t  = rasm.transpose(1, 0, 2) if rasm is not None else None

        for start in range(0, T - WINDOW_SEC + 1, STEP_SEC):
            end  = start + WINDOW_SEC
            w_de = de_t[start:end]

            parts = []

            parts.append(compute_stat_features(w_de.reshape(WINDOW_SEC, -1)))

            if psd_t is not None:
                parts.append(compute_stat_features(
                    psd_t[start:end].reshape(WINDOW_SEC, -1)))

            parts.append(compute_band_ratio_features(w_de))

            for arr_t in [dm_t, rm_t]:
                if arr_t is not None:
                    w_a = arr_t[start:end].reshape(WINDOW_SEC, -1)
                    parts.append(w_a.mean(axis=0))
                    parts.append(w_a.std(axis=0))

            windows.append(np.concatenate(parts).astype(np.float32))
            labels.append(trial_labels[t])

    if not windows:
        return None, None
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)



def load_all_files(root_dir):
    feat_dir = os.path.join(root_dir, FEATURE_FOLDER)
    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(f"\n[ERROR] '{feat_dir}' not found.")

    trial_labels = list(TRIAL_LABELS_DEFAULT)
    label_file   = os.path.join(feat_dir, "label.mat")
    if os.path.exists(label_file):
        try:
            lmat         = sio.loadmat(label_file)
            lkey         = [k for k in lmat if not k.startswith("_")][0]
            loaded       = np.array(lmat[lkey]).flatten().astype(int)
            trial_labels = [int(v) + 1 for v in loaded]
            print(f"  Trial labels loaded: {trial_labels}")
        except Exception as e:
            print(f"  Warning: {e}")

    all_mats = sorted(glob.glob(os.path.join(feat_dir, "*.mat")))
    files = []
    for f in all_mats:
        try:
            int(os.path.basename(f).split("_")[0])
            files.append(f)
        except ValueError:
            print(f"  Skipping: {os.path.basename(f)}")

    if not files:
        raise FileNotFoundError(f"No subject .mat files in '{feat_dir}'")

    unique_subjects = sorted(set(
        int(os.path.basename(f).split("_")[0]) for f in files))
    print(f"  Found {len(files)} files | {len(unique_subjects)} subjects: {unique_subjects}")

    sub_session = {}
    all_X, all_y, subj_list, sess_list = [], [], [], []

    for filepath in files:
        basename   = os.path.basename(filepath)
        subj_id    = int(basename.split("_")[0])
        sub_session.setdefault(subj_id, 0)
        sess_idx   = sub_session[subj_id]
        sub_session[subj_id] += 1

        mat  = sio.loadmat(filepath)
        X, y = extract_subject_windows(mat, trial_labels)
        if X is None:
            print(f"  Warning: no data from {basename}")
            continue

        X = StandardScaler().fit_transform(X)
        all_X.append(X)
        all_y.append(y)
        subj_list.extend([subj_id]  * len(y))
        sess_list.extend([sess_idx] * len(y))

    X_all    = np.vstack(all_X).astype(np.float32)
    y_all    = np.concatenate(all_y).astype(np.int64)
    subjects = np.array(subj_list, dtype=np.int64)
    sessions = np.array(sess_list, dtype=np.int64)

    print(f"\n  Total windows : {len(y_all)}")
    print(f"  Feature dim   : {X_all.shape[1]}  (before MI selection)")
    print(f"  Class balance : "
          f"Neg={(y_all==0).sum()}  Neu={(y_all==1).sum()}  Pos={(y_all==2).sum()}")
    return X_all, y_all, subjects, sessions



def make_subject_folds(subjects, n_splits):
    unique = np.unique(subjects)
    rng    = np.random.RandomState(RANDOM_SEED)
    return np.array_split(rng.permutation(unique), n_splits)


def full_metrics(y_true, y_pred, name=""):
    return dict(
        name      = name,
        acc       = accuracy_score(y_true, y_pred),
        f1        = f1_score(y_true, y_pred, average="weighted"),
        precision = precision_score(y_true, y_pred, average="weighted",
                                    zero_division=0),
        recall    = recall_score(y_true, y_pred, average="weighted",
                                 zero_division=0),
        cm        = confusion_matrix(y_true, y_pred),
    )


def cv_model(clf, X, y, subjects, name="", n_folds=None):
    """Subject-wise K-fold CV with per-fold MI feature selection."""
    if n_folds is None:
        n_folds = N_FOLDS
    folds      = make_subject_folds(subjects, n_folds)
    pa, ta, fa = [], [], []

    for fold_idx, fold_subs in enumerate(folds):
        te  = np.isin(subjects, fold_subs)
        tr  = ~te
        sel = SelectKBest(mutual_info_classif, k=N_FEATURES_SEL)
        Xtr = sel.fit_transform(X[tr], y[tr])
        Xte = sel.transform(X[te])

        clf.fit(Xtr, y[tr])
        preds    = clf.predict(Xte)
        fold_acc = accuracy_score(y[te], preds)
        fa.append(fold_acc)
        pa.extend(preds)
        ta.extend(y[te])
        print(f"    Fold {fold_idx+1}/{n_folds}  acc = {fold_acc:.4f}")

    return full_metrics(ta, pa, name), fa



def get_feature_group_scores(X, y):
    """MI scores grouped by feature type."""
    selector = SelectKBest(mutual_info_classif, k="all")
    selector.fit(X, y)
    scores = selector.scores_

    F          = X.shape[1]
    n_de_stat  = 4 * N_CHANNELS * N_BANDS
    n_psd_stat = 4 * N_CHANNELS * N_BANDS
    n_ratio    = N_CHANNELS * (N_BANDS + 3)

    names = (["DE_stat"]   * n_de_stat +
             ["PSD_stat"]  * n_psd_stat +
             ["BandRatio"] * n_ratio +
             ["Asymmetry"] * max(0, F - n_de_stat - n_psd_stat - n_ratio))[:F]
    return scores, names



def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(results, path):
    names  = list(results.keys())
    accs   = [results[n]["acc"] for n in names]
    f1s    = [results[n]["f1"]  for n in names]
    colors = [PALETTE.get(n, "#9e9e9e") for n in names]
    idx = np.arange(len(names))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(idx - w/2, accs, w, label="Accuracy",
                color=colors, alpha=0.9, edgecolor="white")
    b2 = ax.bar(idx + w/2, f1s,  w, label="F1 Score",
                color=colors, alpha=0.55, edgecolor="white")
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, color="#555")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(
        "Lightweight ML — Accuracy & F1\n"
        f"(Subject-wise {N_FOLDS}-fold CV, Top-{N_FEATURES_SEL} MI Features, SEED)",
        fontsize=12)
    ax.legend(loc="upper left")
    ax.axhline(0.333, color="#bbb", lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)


def plot_confusion_matrices(results, path):
    n  = len(results)
    nc = 3
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    axes = np.array(axes).flatten()
    for ax, (name, res) in zip(axes, results.items()):
        cm   = res["cm"].astype(float)
        cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_n, annot=res["cm"], fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False, linewidths=0.5, annot_kws={"size": 11})
        ax.set_title(f"{name}\nAcc={res['acc']:.3f}  F1={res['f1']:.3f}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for ax in axes[len(results):]:
        ax.set_visible(False)
    plt.suptitle("Confusion Matrices — All Folds Combined", y=1.01, fontsize=13)
    plt.tight_layout()
    _save(fig, path)


def plot_per_class_f1(results, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    n_models = len(results)
    x = np.arange(len(CLASS_NAMES))
    w = 0.8 / n_models
    for i, (name, res) in enumerate(results.items()):
        cm = res["cm"]
        per_class = []
        for c in range(len(CLASS_NAMES)):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            d  = 2*tp + fp + fn
            per_class.append(2*tp/d if d > 0 else 0.0)
        off = (i - n_models/2 + 0.5) * w
        ax.bar(x + off, per_class, w, label=name,
               color=PALETTE.get(name, "#9e9e9e"), alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score by Model", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    _save(fig, path)


def plot_fold_variance(fold_dict, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    names  = list(fold_dict.keys())
    colors = [PALETTE.get(n, "#9e9e9e") for n in names]
    bp = ax.boxplot([fold_dict[n] for n in names], patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Fold Accuracy")
    ax.set_title(f"Accuracy Variance Across {N_FOLDS} Subject Folds", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    _save(fig, path)


def plot_band_power(X_all, y_all, path):
    n_mean_de = N_CHANNELS * N_BANDS
    X_band    = X_all[:, :n_mean_de].reshape(len(X_all), N_CHANNELS, N_BANDS).mean(axis=1)
    fig, ax   = plt.subplots(figsize=(8, 4))
    x = np.arange(N_BANDS); w = 0.25
    cls_colors = ["#d45a5a", "#888888", "#4a8fd4"]
    for c in range(3):
        ax.bar(x + (c-1)*w, X_band[y_all == c].mean(axis=0), w,
               label=CLASS_NAMES[c], color=cls_colors[c], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("Mean normalised DE power")
    ax.set_title("EEG Frequency Band Power by Cognitive State", fontsize=13)
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)


def plot_feature_group_importance(X_all, y_all, path):
    """MI importance by feature group + top-20 individual features."""
    scores, names = get_feature_group_scores(X_all, y_all)
    df            = pd.DataFrame({"score": scores, "group": names})
    group_means   = df.groupby("group")["score"].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ["#e74c3c", "#4a8fd4", "#f0a050", "#5bbfa8"]
    bars = axes[0].bar(group_means.index, group_means.values,
                       color=colors[:len(group_means)], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, group_means.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+0.0005,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Mean MI Score")
    axes[0].set_title("Feature Group Importance (MI)", fontsize=11)
    axes[0].spines[["top", "right"]].set_visible(False)

    top_idx    = np.argsort(scores)[::-1][:20]
    top_scores = scores[top_idx]
    top_lbls   = [f"F{i}({names[i][:6]})" for i in top_idx]
    axes[1].barh(range(20), top_scores[::-1], color="#4a8fd4", alpha=0.8, edgecolor="white")
    axes[1].set_yticks(range(20)); axes[1].set_yticklabels(top_lbls[::-1], fontsize=8)
    axes[1].set_xlabel("MI Score")
    axes[1].set_title("Top-20 Individual Features", fontsize=11)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Feature Importance — MI-based (Xu et al. approach)", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, path)


def plot_subject_loso(X_all, y_all, subjects, path):
    """Per-subject LOSO accuracy using LinearSVC (fast)."""
    unique_subs  = sorted(set(subjects.tolist()))
    accs_per_sub = []
    for s in unique_subs:
        mask = subjects == s
        sel  = SelectKBest(mutual_info_classif, k=N_FEATURES_SEL)
        Xtr  = sel.fit_transform(X_all[~mask], y_all[~mask])
        Xte  = sel.transform(X_all[mask])
        clf  = LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED)
        clf.fit(Xtr, y_all[~mask])
        accs_per_sub.append(accuracy_score(y_all[mask], clf.predict(Xte)))

    fig, ax = plt.subplots(figsize=(max(8, len(unique_subs)*0.7+2), 3.5))
    x = np.arange(len(unique_subs))
    ax.bar(x, accs_per_sub, 0.6,
           color=PALETTE.get("LinearSVC", "#e74c3c"), alpha=0.85, edgecolor="white")
    for xi, v in zip(x, accs_per_sub):
        ax.text(xi, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in unique_subs], fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("LOSO Accuracy")
    ax.set_title("Per-Subject Leave-One-Out Accuracy (LinearSVC + MI Selection)", fontsize=12)
    ax.axhline(0.333, color="#bbb", lw=0.8, ls="--", label="Chance")
    ax.axhline(np.mean(accs_per_sub), color="#e05050", lw=1.2, ls="-.",
               label=f"Mean = {np.mean(accs_per_sub):.3f}")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    _save(fig, path)


def plot_gender_analysis(gender_acc, path):
    if not gender_acc:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    genders = list(gender_acc.keys())
    accs    = [gender_acc[g] for g in genders]
    colors  = ["#4a8fd4", "#f0a050"]
    bars    = ax.bar(genders, accs, color=colors[:len(genders)],
                     alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy by Gender (LinearSVC, LOSO)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    _save(fig, path)


def plot_feature_selection_curve(X_all, y_all, subjects, path):
    """
    Accuracy vs K (number of MI features) — mirrors Xu et al. Fig 9.
    Uses LinearSVC + 3-fold CV for speed.
    """
    k_values = [10, 20, 40, 60, 80, 100, 150, 200, 300]
    k_values = [k for k in k_values if k <= X_all.shape[1]]
    accs_k   = []
    folds    = make_subject_folds(subjects, 3)

    print(f"  (Using 3-fold CV + LinearSVC for speed)")
    for k in k_values:
        fold_accs = []
        for fold_subs in folds:
            te  = np.isin(subjects, fold_subs); tr = ~te
            sel = SelectKBest(mutual_info_classif, k=k)
            Xtr = sel.fit_transform(X_all[tr], y_all[tr])
            Xte = sel.transform(X_all[te])
            clf = LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED)
            clf.fit(Xtr, y_all[tr])
            fold_accs.append(accuracy_score(y_all[te], clf.predict(Xte)))
        accs_k.append(np.mean(fold_accs))
        print(f"    K={k:4d}  acc={accs_k[-1]:.4f}")

    best_k   = k_values[np.argmax(accs_k)]
    best_acc = max(accs_k)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_values, accs_k, "o-", color="#e74c3c", lw=2, ms=7)
    ax.axvline(best_k, color="#4a8fd4", lw=1.2, ls="--",
               label=f"Best K={best_k} (acc={best_acc:.3f})")
    ax.axvline(N_FEATURES_SEL, color="#f0a050", lw=1.2, ls=":",
               label=f"Default K={N_FEATURES_SEL}")
    ax.set_xlabel("Number of Selected Features (K)")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title(
        "Feature Selection Curve — Accuracy vs Top-K MI Features\n"
        "(Inspired by Xu et al. 2023, Figure 9)",
        fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)
    return best_k, best_acc



def main():
    print("=" * 68)
    print(" EEG COGNITIVE STATE CLASSIFICATION — LIGHTWEIGHT ML (SEED)")
    print(" Inspired by: Xu et al. (2023) Multi-Level Attention Recognition")
    print("=" * 68)
    t_start = time.time()

    print("\nLoading metadata...")
    meta = load_metadata(ROOT_DIR)

    print(f"\nLoading + extracting features from '{FEATURE_FOLDER}'...")
    X_all, y_all, subjects, sessions = load_all_files(ROOT_DIR)
    print(f"  Done in {time.time()-t_start:.1f}s\n")

    print("─" * 68)
    print("FEATURE SELECTION CURVE  (LinearSVC, 3-fold, fast)")
    print("─" * 68)
    best_k, _ = plot_feature_selection_curve(
        X_all, y_all, subjects,
        os.path.join(OUTPUT_DIR, "1_feature_selection_curve.png"))

    models = {
        "LinearSVC"      : LinearSVC(C=0.5, max_iter=2000,
                                     random_state=RANDOM_SEED),
        "LDA"            : LinearDiscriminantAnalysis(solver="svd"),
        "Random Forest"  : RandomForestClassifier(n_estimators=200,
                                                   max_features="sqrt",
                                                   min_samples_leaf=5,
                                                   n_jobs=-1,
                                                   random_state=RANDOM_SEED),
        "Hist Grad Boost": HistGradientBoostingClassifier(
                                max_iter=200, max_depth=5,
                                learning_rate=0.1, min_samples_leaf=20,
                                random_state=RANDOM_SEED),
        "KNN"            : KNeighborsClassifier(n_neighbors=7,
                                                algorithm="ball_tree",
                                                n_jobs=-1),
        "Logistic Reg"   : LogisticRegression(C=1.0, max_iter=500,
                                               solver="saga",
                                               multi_class="multinomial",
                                               n_jobs=-1,
                                               random_state=RANDOM_SEED),
    }

    results   = {}
    fold_accs = {}

    print("\n" + "─" * 68)
    print(f"RUNNING {N_FOLDS}-FOLD SUBJECT-WISE CV  (Top-{N_FEATURES_SEL} MI features)")
    print("─" * 68)

    for name, clf in models.items():
        print(f"\n  [{name}]")
        t1 = time.time()
        res, fa = cv_model(clf, X_all, y_all, subjects, name)
        results[name]   = res
        fold_accs[name] = fa
        print(f"  → Acc={res['acc']:.4f}  F1={res['f1']:.4f}  "
              f"Prec={res['precision']:.4f}  Rec={res['recall']:.4f}  "
              f"({time.time()-t1:.1f}s)")

    print("\n" + "=" * 68)
    print(" FINAL RESULTS")
    print("=" * 68)
    print(f"{'Model':<20} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print("-" * 68)
    sorted_r = sorted(results.items(), key=lambda x: x[1]["acc"], reverse=True)
    for name, res in sorted_r:
        tag = "  ← BEST" if name == sorted_r[0][0] else ""
        print(f"{name:<20} {res['acc']:>9.4f} {res['f1']:>7.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>8.4f}{tag}")

    best_name, best_res = sorted_r[0]
    print(f"\nDetailed report — {best_name}:")
    cm = best_res["cm"]
    at, ap = [], []
    for i in range(3):
        for j in range(3):
            at.extend([i]*cm[i, j]); ap.extend([j]*cm[i, j])
    print(classification_report(at, ap, target_names=CLASS_NAMES, digits=4))

    gender_acc = {}
    if "subject_gender" in meta:
        print("Gender analysis (LOSO LinearSVC)...")
        gdf = meta["subject_gender"].set_index("subject_id")
        for gender in gdf["gender"].unique():
            subs_g  = set(gdf[gdf["gender"] == gender].index.tolist())
            mask_te = np.array([s in subs_g for s in subjects])
            if mask_te.sum() == 0:
                continue
            sel_g = SelectKBest(mutual_info_classif, k=N_FEATURES_SEL)
            Xtr_g = sel_g.fit_transform(X_all[~mask_te], y_all[~mask_te])
            Xte_g = sel_g.transform(X_all[mask_te])
            clf_g = LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED)
            clf_g.fit(Xtr_g, y_all[~mask_te])
            gender_acc[gender] = accuracy_score(y_all[mask_te], clf_g.predict(Xte_g))
            print(f"  Gender '{gender}': acc = {gender_acc[gender]:.4f}")

    rows = [{"model"    : name,
             "accuracy" : round(res["acc"],       6),
             "f1"       : round(res["f1"],        6),
             "precision": round(res["precision"], 6),
             "recall"   : round(res["recall"],    6)}
            for name, res in results.items()]
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    pkl_path = os.path.join(OUTPUT_DIR, "all_results.pkl")
    with open(pkl_path, "wb") as fp:
        pickle.dump(results, fp)
    print(f"PKL saved: {pkl_path}")

    print("\nGenerating plots...")
    plot_model_comparison(results,
        os.path.join(OUTPUT_DIR, "2_model_comparison.png"))
    plot_confusion_matrices(results,
        os.path.join(OUTPUT_DIR, "3_confusion_matrices.png"))
    plot_per_class_f1(results,
        os.path.join(OUTPUT_DIR, "4_per_class_f1.png"))
    plot_fold_variance(fold_accs,
        os.path.join(OUTPUT_DIR, "5_fold_variance.png"))
    plot_band_power(X_all, y_all,
        os.path.join(OUTPUT_DIR, "6_band_power.png"))
    plot_feature_group_importance(X_all, y_all,
        os.path.join(OUTPUT_DIR, "7_feature_importance.png"))
    plot_subject_loso(X_all, y_all, subjects,
        os.path.join(OUTPUT_DIR, "8_subject_loso.png"))
    if gender_acc:
        plot_gender_analysis(gender_acc,
            os.path.join(OUTPUT_DIR, "9_gender_analysis.png"))

    total = (time.time() - t_start) / 60
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Total runtime: {total:.1f} min")
    print("=" * 68)


if __name__ == "__main__":
    main()
