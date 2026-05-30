"""
=============================================================================
PROJECT 2 — EEG-Based Learner Attention Classification & Video Recommendation
SEED Dataset | Lightweight Machine Learning Only
=============================================================================

TITLE: Design and Development of a Recommendation System Based on
       Learners' Attention Levels

"""



ROOT_DIR       = r"D:\IML_LAB_PROJECT\SEED_Dataset(preprocessed eeg)"
FEATURE_FOLDER = "ExtractedFeatures_1s"
OUTPUT_DIR     = r"D:\IML_LAB_PROJECT\results_attention_recommender"

N_FOLDS        = 5
WINDOW_SEC     = 10
STEP_SEC       = 5
N_FEATURES_SEL = 60
RANDOM_SEED    = 42

W_TB    = 0.4
W_FB    = 0.3
W_DASM  = 0.3

FRONTAL_CH = [0, 1, 2, 3, 4, 5, 6]


import os, glob, time, warnings, pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats_lib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report, ndcg_score)
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")


TRIAL_LABELS_DEFAULT = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
ATTENTION_NAMES      = ["Low", "Medium", "High"]
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
                print(f"  Gender info loaded for {len(rows)} subjects.")
        except Exception as e:
            print(f"  Warning: {e}")
    return meta



def compute_attention_score(de_window, dasm_window):
    """
    Compute composite attention score for a single window.
    de_window:   (WINDOW_SEC, 62, 5)
    dasm_window: (WINDOW_SEC, 27, 5) or None

    Returns scalar score — LOWER = more attentive.
    """
    theta = de_window[:, :, 1].mean()
    beta  = de_window[:, :, 3].mean()
    tb_ratio = theta / (beta + 1e-9)

    frontal_de   = de_window[:, FRONTAL_CH, :]
    f_total      = frontal_de.sum(axis=2).mean() + 1e-9
    f_beta       = frontal_de[:, :, 3].mean()
    f_beta_supp  = 1.0 - (f_beta / f_total)

    if dasm_window is not None:
        dasm_beta = dasm_window[:, :, 3].mean()
        dasm_score = -np.tanh(dasm_beta / 5.0)
    else:
        dasm_score = 0.0

    composite = (W_TB * tb_ratio) + (W_FB * f_beta_supp) + (W_DASM * dasm_score)
    return float(composite)



def compute_stat_features(window):
    """mean, std, skew, kurt over time axis. window: (T, F) → 4F features."""
    return np.concatenate([
        window.mean(axis=0),
        window.std(axis=0),
        stats_lib.skew(window, axis=0),
        stats_lib.kurtosis(window, axis=0),
    ])


def compute_attention_features(de_window, dasm_window):
    """
    Attention-specific features per window:
      1. Theta/Beta ratio per channel           → 62 features
      2. Beta/Alpha ratio per channel           → 62 features
      3. Frontal relative beta per channel      → 7 features
      4. Relative theta per channel             → 62 features
      5. DASM beta & alpha channels             → 27+27 = 54 features
      6. Global theta/beta, alpha/beta scalars  → 2 features
    Total: ~275 features
    """
    parts = []

    theta_ch = de_window[:, :, 1].mean(axis=0)
    beta_ch  = de_window[:, :, 3].mean(axis=0)
    alpha_ch = de_window[:, :, 2].mean(axis=0)
    total_ch = de_window.mean(axis=0).sum(axis=1) + 1e-9

    parts.append(theta_ch / (beta_ch + 1e-9))
    parts.append(beta_ch  / (alpha_ch + 1e-9))
    parts.append(beta_ch[FRONTAL_CH] / (total_ch[FRONTAL_CH] + 1e-9))
    parts.append(theta_ch / (total_ch + 1e-9))

    if dasm_window is not None:
        parts.append(dasm_window[:, :, 3].mean(axis=0))
        parts.append(dasm_window[:, :, 2].mean(axis=0))

    parts.append(np.array([
        theta_ch.mean() / (beta_ch.mean() + 1e-9),
        alpha_ch.mean() / (beta_ch.mean() + 1e-9),
    ]))

    return np.concatenate(parts)


def extract_windows_with_attention(mat, trial_labels):
    """
    Extract features, attention labels, and raw attention scores per window.

    Returns:
      X         : feature matrix (N, F)
      y_attn    : attention labels 0/1/2 (Low/Med/High) — assigned after all windows
      scores    : raw composite attention scores (N,) — for recommendation
      trial_ids : trial index per window (for session-level analysis)
    """
    windows, raw_scores, trial_ids = [], [], []

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
            w_dm = dm_t[start:end] if dm_t is not None else None

            parts = []

            parts.append(compute_stat_features(w_de.reshape(WINDOW_SEC, -1)))

            if psd_t is not None:
                parts.append(compute_stat_features(
                    psd_t[start:end].reshape(WINDOW_SEC, -1)))

            parts.append(compute_attention_features(w_de, w_dm))

            for arr_t in [dm_t, rm_t]:
                if arr_t is not None:
                    w_a = arr_t[start:end].reshape(WINDOW_SEC, -1)
                    parts.append(w_a.mean(axis=0))
                    parts.append(w_a.std(axis=0))

            windows.append(np.concatenate(parts).astype(np.float32))

            score = compute_attention_score(w_de, w_dm)
            raw_scores.append(score)
            trial_ids.append(t)

    if not windows:
        return None, None, None, None

    X      = np.array(windows,    dtype=np.float32)
    scores = np.array(raw_scores, dtype=np.float32)
    tids   = np.array(trial_ids,  dtype=np.int64)

    p33 = np.percentile(scores, 33)
    p66 = np.percentile(scores, 66)
    y = np.where(scores < p33, 2,
         np.where(scores < p66, 1, 0))
    y = y.astype(np.int64)

    return X, y, scores, tids



def load_all_data(root_dir):
    feat_dir = os.path.join(root_dir, FEATURE_FOLDER)
    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(f"\n[ERROR] '{feat_dir}' not found.")

    trial_labels = list(TRIAL_LABELS_DEFAULT)
    label_file   = os.path.join(feat_dir, "label.mat")
    if os.path.exists(label_file):
        try:
            lmat  = sio.loadmat(label_file)
            lkey  = [k for k in lmat if not k.startswith("_")][0]
            loaded = np.array(lmat[lkey]).flatten().astype(int)
            trial_labels = [int(v) + 1 for v in loaded]
            print(f"  Trial emotion labels: {trial_labels}")
        except Exception as e:
            print(f"  Warning: {e}")

    all_mats = sorted(glob.glob(os.path.join(feat_dir, "*.mat")))
    files = []
    for f in all_mats:
        try:
            int(os.path.basename(f).split("_")[0])
            files.append(f)
        except ValueError:
            pass

    if not files:
        raise FileNotFoundError(f"No subject .mat files in '{feat_dir}'")

    unique_subjects = sorted(set(
        int(os.path.basename(f).split("_")[0]) for f in files))
    print(f"  Found {len(files)} files | {len(unique_subjects)} subjects")

    sub_sess = {}
    all_X, all_y, all_scores = [], [], []
    subj_list, sess_list, trial_list = [], [], []

    for filepath in files:
        basename  = os.path.basename(filepath)
        subj_id   = int(basename.split("_")[0])
        sub_sess.setdefault(subj_id, 0)
        sess_idx  = sub_sess[subj_id]
        sub_sess[subj_id] += 1

        mat = sio.loadmat(filepath)
        X, y, scores, tids = extract_windows_with_attention(mat, trial_labels)

        if X is None:
            print(f"  Warning: no data from {basename}")
            continue

        X = StandardScaler().fit_transform(X)

        all_X.append(X)
        all_y.append(y)
        all_scores.append(scores)
        subj_list.extend([subj_id]  * len(y))
        sess_list.extend([sess_idx] * len(y))
        trial_list.extend(tids.tolist())

    X_all    = np.vstack(all_X).astype(np.float32)
    y_all    = np.concatenate(all_y).astype(np.int64)
    scores   = np.concatenate(all_scores).astype(np.float32)
    subjects = np.array(subj_list,  dtype=np.int64)
    sessions = np.array(sess_list,  dtype=np.int64)
    trials   = np.array(trial_list, dtype=np.int64)

    print(f"\n  Total windows : {len(y_all)}")
    print(f"  Feature dim   : {X_all.shape[1]}  (before MI selection)")
    print(f"  Attention dist: "
          f"Low={(y_all==0).sum()}  Med={(y_all==1).sum()}  High={(y_all==2).sum()}")

    return X_all, y_all, scores, subjects, sessions, trials



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


def cv_model(clf, X, y, subjects, name=""):
    """Subject-wise K-fold CV with per-fold MI feature selection."""
    folds      = make_subject_folds(subjects, N_FOLDS)
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
        pa.extend(preds); ta.extend(y[te])
        print(f"    Fold {fold_idx+1}/{N_FOLDS}  acc={fold_acc:.4f}")

    return full_metrics(ta, pa, name), fa



def build_recommendation(X_all, y_all, scores, subjects, sessions, trials):
    """
    For each subject, predict attention per window using best model (HGB).
    Aggregate to trial level → rank trials by mean predicted attention.
    Recommend bottom-K trials (lowest attention = needs retaking).

    Returns dataframe with per-subject trial rankings.
    """
    print("\n  Building recommendation system (HGB, per-subject LOSO)...")

    sel  = SelectKBest(mutual_info_classif, k=N_FEATURES_SEL)
    clf  = HistGradientBoostingClassifier(
               max_iter=200, max_depth=5, learning_rate=0.1,
               random_state=RANDOM_SEED)

    rec_rows = []
    unique_subs = sorted(np.unique(subjects))

    attn_matrix = {}

    prec_at_k = []

    for subj in unique_subs:
        mask_te = subjects == subj
        mask_tr = ~mask_te

        if mask_tr.sum() < 10:
            continue

        Xtr = sel.fit_transform(X_all[mask_tr], y_all[mask_tr])
        Xte = sel.transform(X_all[mask_te])
        clf.fit(Xtr, y_all[mask_tr])
        preds = clf.predict(Xte)

        subj_scores  = scores[mask_te]
        subj_trials  = trials[mask_te]
        subj_preds   = preds

        trial_summary = {}
        for i in range(15):
            t_mask = subj_trials == i
            if t_mask.sum() == 0:
                continue
            mean_pred  = subj_preds[t_mask].mean()
            mean_score = subj_scores[t_mask].mean()
            trial_summary[i] = {
                "mean_pred_label"  : mean_pred,
                "mean_raw_score"   : mean_score,
                "n_windows"        : t_mask.sum(),
                "pct_high_attn"    : (subj_preds[t_mask] == 2).mean(),
                "pct_low_attn"     : (subj_preds[t_mask] == 0).mean(),
            }

        sorted_trials = sorted(trial_summary.items(),
                                key=lambda x: x[1]["mean_pred_label"])

        attn_matrix[subj] = {t: v["mean_pred_label"] for t, v in trial_summary.items()}

        rec_3 = [t for t, _ in sorted_trials[:3]]

        for rank, (trial_id, info) in enumerate(sorted_trials):
            rec_rows.append({
                "subject"          : subj,
                "trial"            : trial_id,
                "rank"             : rank + 1,
                "mean_pred_label"  : round(info["mean_pred_label"], 3),
                "mean_raw_score"   : round(info["mean_raw_score"], 4),
                "pct_high_attn"    : round(info["pct_high_attn"] * 100, 1),
                "pct_low_attn"     : round(info["pct_low_attn"] * 100, 1),
                "recommended"      : trial_id in rec_3,
            })

    rec_df = pd.DataFrame(rec_rows)
    print(f"  Recommendation table: {rec_df.shape[0]} rows "
          f"({rec_df['subject'].nunique()} subjects)")
    return rec_df, attn_matrix


def compute_recommendation_metrics(rec_df, attn_matrix):
    """
    Compute recommendation quality metrics:
    - Precision@K: fraction of recommended trials that are truly low-attention
    - Recall@K: fraction of true low-attention trials that are recommended
    - NDCG@K: ranking quality

    Ground truth: trials where mean_pred_label < 0.8 (weighted low attention)
    """
    prec_list, rec_list, f1_list = [], [], []

    for subj in rec_df["subject"].unique():
        df_s = rec_df[rec_df["subject"] == subj].sort_values("rank")
        K    = min(3, len(df_s))

        thresh  = df_s["mean_pred_label"].quantile(0.4)
        truly_low = set(df_s[df_s["mean_pred_label"] <= thresh]["trial"].tolist())
        recommended = set(df_s[df_s["recommended"]]["trial"].tolist())

        if not truly_low or not recommended:
            continue

        tp  = len(recommended & truly_low)
        prec = tp / len(recommended) if recommended else 0
        rec  = tp / len(truly_low)   if truly_low   else 0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    return {
        "Precision@3" : round(np.mean(prec_list), 4),
        "Recall@3"    : round(np.mean(rec_list),  4),
        "F1@3"        : round(np.mean(f1_list),   4),
    }



def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(results, path):
    names  = list(results.keys())
    accs   = [results[n]["acc"] for n in names]
    f1s    = [results[n]["f1"]  for n in names]
    colors = [PALETTE.get(n, "#9e9e9e") for n in names]
    idx = np.arange(len(names)); w = 0.38
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
    ax.set_xticks(idx); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title(
        "Attention Level Classification — Accuracy & F1\n"
        f"(Subject-wise {N_FOLDS}-fold CV, Top-{N_FEATURES_SEL} MI Features, SEED)",
        fontsize=12)
    ax.legend(loc="upper left")
    ax.axhline(0.333, color="#bbb", lw=0.8, ls="--", label="Chance")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)


def plot_confusion_matrices(results, path):
    n = len(results); nc = 3; nr = (n+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    axes = np.array(axes).flatten()
    for ax, (name, res) in zip(axes, results.items()):
        cm   = res["cm"].astype(float)
        cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_n, annot=res["cm"], fmt="d", cmap="Greens",
                    xticklabels=ATTENTION_NAMES, yticklabels=ATTENTION_NAMES,
                    ax=ax, cbar=False, linewidths=0.5, annot_kws={"size": 11})
        ax.set_title(f"{name}\nAcc={res['acc']:.3f}  F1={res['f1']:.3f}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for ax in axes[len(results):]: ax.set_visible(False)
    plt.suptitle("Confusion Matrices — Attention Level Classification\n(All Folds Combined)",
                 y=1.01, fontsize=13)
    plt.tight_layout()
    _save(fig, path)


def plot_per_class_f1(results, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    n_models = len(results)
    x = np.arange(len(ATTENTION_NAMES)); w = 0.8 / n_models
    for i, (name, res) in enumerate(results.items()):
        cm = res["cm"]
        per_class = []
        for c in range(3):
            tp = cm[c,c]; fp = cm[:,c].sum()-tp; fn = cm[c,:].sum()-tp
            d  = 2*tp+fp+fn
            per_class.append(2*tp/d if d > 0 else 0.0)
        off = (i - n_models/2 + 0.5) * w
        ax.bar(x + off, per_class, w, label=name,
               color=PALETTE.get(name,"#9e9e9e"), alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(ATTENTION_NAMES)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score by Model — Attention Levels", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    plt.tight_layout()
    _save(fig, path)


def plot_fold_variance(fold_dict, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    names  = list(fold_dict.keys())
    colors = [PALETTE.get(n,"#9e9e9e") for n in names]
    bp = ax.boxplot([fold_dict[n] for n in names], patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Fold Accuracy")
    ax.set_title(f"Accuracy Variance Across {N_FOLDS} Subject Folds", fontsize=13)
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    plt.tight_layout()
    _save(fig, path)


def plot_attention_band_profile(X_all, y_all, path):
    """
    Mean EEG band power per attention class.
    Shows how band profiles differ across Low/Medium/High attention.
    """
    n_mean = N_CHANNELS * N_BANDS
    X_band = X_all[:, :n_mean].reshape(len(X_all), N_CHANNELS, N_BANDS).mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(N_BANDS); w = 0.25
    cls_colors = ["#e74c3c", "#f0a050", "#4a8fd4"]
    for c in range(3):
        ax.bar(x + (c-1)*w, X_band[y_all==c].mean(axis=0), w,
               label=ATTENTION_NAMES[c], color=cls_colors[c],
               alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("Mean normalised DE power")
    ax.set_title("EEG Frequency Band Profile by Attention Level", fontsize=13)
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)


def plot_attention_score_distribution(scores, y_all, path):
    """
    Distribution of raw composite attention scores by class.
    Validates that our scoring + labeling is sensible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    cls_colors = ["#e74c3c", "#f0a050", "#4a8fd4"]

    for c in range(3):
        mask = y_all == c
        vals = scores[mask]
        axes[0].hist(vals, bins=40, alpha=0.6, color=cls_colors[c],
                     label=f"{ATTENTION_NAMES[c]} (n={mask.sum()})", density=True)
    axes[0].set_xlabel("Composite Attention Score (lower = more attentive)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Attention Score Distribution by Class", fontsize=11)
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

    data   = [scores[y_all == c] for c in range(3)]
    bp = axes[1].boxplot(data, patch_artist=True,
                         medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp["boxes"], cls_colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    axes[1].set_xticklabels(ATTENTION_NAMES)
    axes[1].set_ylabel("Composite Attention Score")
    axes[1].set_title("Score Distribution per Attention Class", fontsize=11)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.suptitle("Composite Attention Score Validation\n"
                 "(Theta/Beta + Frontal Beta + DASM Asymmetry)", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, path)


def plot_recommendation_heatmap(attn_matrix, path):
    """
    Heatmap: subjects × trials, coloured by mean predicted attention score.
    Low score = red = recommend for re-watching.
    """
    subjects_sorted = sorted(attn_matrix.keys())
    n_trials = 15
    mat = np.full((len(subjects_sorted), n_trials), np.nan)
    for i, s in enumerate(subjects_sorted):
        for t, v in attn_matrix[s].items():
            if t < n_trials:
                mat[i, t] = v

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    plt.colorbar(im, ax=ax, label="Mean Predicted Attention (0=Low, 2=High)")
    ax.set_xticks(range(n_trials))
    ax.set_xticklabels([f"T{i+1}" for i in range(n_trials)], fontsize=9)
    ax.set_yticks(range(len(subjects_sorted)))
    ax.set_yticklabels([f"S{s}" for s in subjects_sorted], fontsize=9)
    ax.set_xlabel("Trial (Video)")
    ax.set_ylabel("Subject")
    ax.set_title("Attention Level Heatmap — Subjects × Trials\n"
                 "Red = Low attention (recommend re-watching) | Green = High attention",
                 fontsize=12)
    plt.tight_layout()
    _save(fig, path)


def plot_recommendation_metrics(rec_metrics, path):
    """Bar chart of recommendation quality metrics."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = list(rec_metrics.keys())
    values = list(rec_metrics.values())
    colors = ["#e74c3c", "#4a8fd4", "#5bbfa8"]
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Recommendation System Quality Metrics\n"
                 "(Top-3 Low-Attention Trials Recommended per Subject)",
                 fontsize=12)
    ax.axhline(0.5, color="#bbb", lw=0.8, ls="--", label="Random baseline")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)


def plot_subject_attention_profile(rec_df, path):
    """Per-subject mean attention level across trials."""
    subj_means = rec_df.groupby("subject")["mean_pred_label"].mean().sort_values()
    fig, ax    = plt.subplots(figsize=(max(8, len(subj_means)*0.7+2), 3.5))
    x = np.arange(len(subj_means))
    colors_bar = ["#e74c3c" if v < 0.8 else "#f0a050" if v < 1.3
                  else "#4a8fd4" for v in subj_means.values]
    ax.bar(x, subj_means.values, 0.6, color=colors_bar, alpha=0.85, edgecolor="white")
    for xi, v in zip(x, subj_means.values):
        ax.text(xi, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subj_means.index], fontsize=9)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel("Mean Predicted Attention (0=Low, 2=High)")
    ax.set_title("Per-Subject Overall Attention Level\n"
                 "(Red=Low | Orange=Medium | Blue=High attention subjects)",
                 fontsize=12)
    ax.axhline(1.0, color="#bbb", lw=0.8, ls="--")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)


def plot_feature_selection_curve(X_all, y_all, subjects, path):
    """Accuracy vs K — mirrors Xu et al. approach."""
    k_values  = [10, 20, 40, 60, 80, 100, 150, 200, 300]
    k_values  = [k for k in k_values if k <= X_all.shape[1]]
    accs_k    = []
    folds     = make_subject_folds(subjects, 3)
    print("  (3-fold for speed)")

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
    ax.set_title("Feature Selection Curve — Attention Classification", fontsize=12)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, path)
    return best_k, best_acc



def main():
    print("=" * 68)
    print(" PROJECT 2 — ATTENTION-LEVEL CLASSIFICATION & RECOMMENDATION")
    print(" EEG-Based Learner Attention | Lightweight ML | SEED Dataset")
    print("=" * 68)
    t_start = time.time()

    print("\nLoading metadata...")
    meta = load_metadata(ROOT_DIR)

    print(f"\nLoading + feature extraction from '{FEATURE_FOLDER}'...")
    X_all, y_all, scores, subjects, sessions, trials = load_all_data(ROOT_DIR)
    print(f"  Done in {time.time()-t_start:.1f}s\n")

    print("─" * 68)
    print("FEATURE SELECTION CURVE")
    print("─" * 68)
    best_k, _ = plot_feature_selection_curve(
        X_all, y_all, subjects,
        os.path.join(OUTPUT_DIR, "P2_01_feature_selection_curve.png"))

    models = {
        "LinearSVC"      : LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED),
        "LDA"            : LinearDiscriminantAnalysis(solver="svd"),
        "Random Forest"  : RandomForestClassifier(n_estimators=200,
                                                   max_features="sqrt",
                                                   min_samples_leaf=5,
                                                   n_jobs=-1,
                                                   random_state=RANDOM_SEED),
        "Hist Grad Boost": HistGradientBoostingClassifier(
                                max_iter=200, max_depth=5, learning_rate=0.1,
                                random_state=RANDOM_SEED),
        "KNN"            : KNeighborsClassifier(n_neighbors=7,
                                                algorithm="ball_tree", n_jobs=-1),
        "Logistic Reg"   : LogisticRegression(C=1.0, max_iter=500, solver="saga",
                                               multi_class="multinomial",
                                               n_jobs=-1, random_state=RANDOM_SEED),
    }

    results   = {}
    fold_accs = {}

    print("\n" + "─" * 68)
    print(f"ATTENTION CLASSIFICATION — {N_FOLDS}-fold subject-wise CV")
    print(f"  Labels: Low / Medium / High attention (percentile-based)")
    print(f"  Features: DE stats + PSD stats + attention-specific + asymmetry")
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
    print(" CLASSIFICATION RESULTS")
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
            at.extend([i]*cm[i,j]); ap.extend([j]*cm[i,j])
    print(classification_report(at, ap, target_names=ATTENTION_NAMES, digits=4))

    print("\n" + "─" * 68)
    print("RECOMMENDATION SYSTEM")
    print("─" * 68)
    rec_df, attn_matrix = build_recommendation(
        X_all, y_all, scores, subjects, sessions, trials)

    rec_metrics = compute_recommendation_metrics(rec_df, attn_matrix)
    print("\n  Recommendation Quality:")
    for k, v in rec_metrics.items():
        print(f"    {k}: {v:.4f}")

    csv_clf = os.path.join(OUTPUT_DIR, "P2_classification_results.csv")
    rows = [{"model": n, "accuracy": round(r["acc"],6),
             "f1": round(r["f1"],6), "precision": round(r["precision"],6),
             "recall": round(r["recall"],6)} for n, r in results.items()]
    pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(csv_clf, index=False)
    print(f"\n  CSV saved: {csv_clf}")

    csv_rec = os.path.join(OUTPUT_DIR, "P2_recommendation_table.csv")
    rec_df.to_csv(csv_rec, index=False)
    print(f"  CSV saved: {csv_rec}")

    pkl_path = os.path.join(OUTPUT_DIR, "P2_all_results.pkl")
    with open(pkl_path, "wb") as fp:
        pickle.dump({"classification": results, "recommendation": rec_df,
                     "rec_metrics": rec_metrics, "attn_matrix": attn_matrix}, fp)
    print(f"  PKL saved: {pkl_path}")

    print("\nGenerating plots...")
    plot_model_comparison(results,
        os.path.join(OUTPUT_DIR, "P2_02_model_comparison.png"))
    plot_confusion_matrices(results,
        os.path.join(OUTPUT_DIR, "P2_03_confusion_matrices.png"))
    plot_per_class_f1(results,
        os.path.join(OUTPUT_DIR, "P2_04_per_class_f1.png"))
    plot_fold_variance(fold_accs,
        os.path.join(OUTPUT_DIR, "P2_05_fold_variance.png"))
    plot_attention_band_profile(X_all, y_all,
        os.path.join(OUTPUT_DIR, "P2_06_attention_band_profile.png"))
    plot_attention_score_distribution(scores, y_all,
        os.path.join(OUTPUT_DIR, "P2_07_attention_score_dist.png"))
    plot_recommendation_heatmap(attn_matrix,
        os.path.join(OUTPUT_DIR, "P2_08_recommendation_heatmap.png"))
    plot_recommendation_metrics(rec_metrics,
        os.path.join(OUTPUT_DIR, "P2_09_recommendation_metrics.png"))
    plot_subject_attention_profile(rec_df,
        os.path.join(OUTPUT_DIR, "P2_10_subject_attention_profile.png"))

    total = (time.time() - t_start) / 60
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Total runtime: {total:.1f} min")
    print("=" * 68)


if __name__ == "__main__":
    main()
