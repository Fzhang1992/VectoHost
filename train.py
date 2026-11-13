#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified virus host 3-class training and five-fold OOF evaluation.

Run one script for 1000bp / 3000bp / 5000bp / full_length.
A single CROSS_HOST_TAXIDS set is applied to every dataset:
records whose taxid matches the set are removed BEFORE cross-validation,
from both training and evaluation. No length-dependent list is used.

Features: precomputed 3-mer (64d), ESM2 (320d), LucaVirus (2560d).
Classifiers, scaling, seed and within-training-fold undersampling retain
settings of the original training scripts. This script does not save a final
model or extract embeddings.
"""

from pathlib import Path
import gzip
import re
import json
import warnings
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score
)

warnings.filterwarnings("ignore")

SEED = 20260914
N_SPLITS = 5
CLASS_NAMES = ["Insect", "Plant", "Other"]

# Four length-specific reference datasets; the estimator code below is shared.
BASE = Path("/public/home/test/test05/zhangzheng/lyo/三分类/重构数据集后")
DATA_ROOT = BASE / "数据保存"
LENGTH = None
DATA_DIR = None
FEATURE_DIR = None
OUTDIR = None

FEATURE_NAMES = ("3mer", "ESM2", "LucaVirus")
EXPECTED_BY_LENGTH = {
    "1000bp": {"Insect": 1040, "Plant": 2111, "Other": 30314},
    "3000bp": {"Insect": 927, "Plant": 1511, "Other": 29494},
    "5000bp": {"Insect": 838, "Plant": 1340, "Other": 28313},
    "full_length": {"Insect": 1040, "Plant": 2111, "Other": 30314},
}
EXPECTED = None
FEATURE_DIMS = {"3mer": 64, "ESM2": 320, "LucaVirus": 2560}
CROSS_HOST_TAXIDS = frozenset({
    "1277443", "228582", "2170100", "1985699", "12287", "223287", "1795442",
    "3052585", "1774200", "187978", "10991", "10986", "11307", "12851"
})


def norm_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_feature(feature_type):
    """读取features/{feature}/{class}目录；不静默删除任何TaxID。"""
    if feature_type not in FEATURE_DIMS:
        raise ValueError(f"未知特征类型：{feature_type}")
    dim = FEATURE_DIMS[feature_type]
    matrices, metas = [], []
    for cls in CLASS_NAMES:
        directory = FEATURE_DIR / feature_type / cls
        npy = directory / "features.npy"
        idsf = directory / "ids.tsv"
        if not npy.is_file() or not idsf.is_file():
            raise FileNotFoundError(f"特征文件未就绪：{npy}; {idsf}")
        X = np.load(npy, mmap_mode="r", allow_pickle=False)
        ids = pd.read_csv(idsf, sep="\t", dtype=str, encoding="utf-8-sig")
        if not {"taxid", "class"}.issubset(ids.columns):
            raise RuntimeError(f"ID表必须含taxid和class列：{idsf}; 实际列={ids.columns.tolist()}")
        ids["taxid"] = ids["taxid"].map(norm_id)
        ids["class"] = ids["class"].astype(str).str.strip()
        if X.ndim != 2 or X.shape != (len(ids), dim):
            raise RuntimeError(f"矩阵维度/行数错误：{npy} shape={X.shape}; IDs={len(ids)}; expected_dim={dim}")
        if len(ids) != EXPECTED[cls]:
            raise RuntimeError(f"类别样本数异常：{feature_type}/{cls}={len(ids)}，预期={EXPECTED[cls]}")
        if (ids["class"] != cls).any():
            bad = ids.loc[ids["class"] != cls, ["taxid", "class"]].head(10).to_dict("records")
            raise RuntimeError(f"类别标签错误：{idsf}; 示例={bad}")
        if ids["taxid"].isna().any() or (ids["taxid"].str.strip() == "").any():
            raise RuntimeError(f"ID表含空TaxID：{idsf}")
        if ids["taxid"].duplicated().any():
            dup = ids.loc[ids["taxid"].duplicated(keep=False), "taxid"].unique().tolist()
            raise RuntimeError(f"同类别内TaxID重复：{feature_type}/{cls}; 示例={dup[:20]}")
        if not np.isfinite(X).all():
            raise RuntimeError(f"特征矩阵包含NaN/Inf：{npy}")
        matrices.append(np.asarray(X, dtype=np.float32))
        metas.append(ids[["taxid", "class"]].copy())
        print(f"[INPUT] {feature_type}/{cls}: ids={len(ids)}, shape={X.shape}")

    X_all = np.vstack(matrices)
    meta_all = pd.concat(metas, ignore_index=True)
    if meta_all["taxid"].duplicated().any():
        dup = meta_all.loc[meta_all["taxid"].duplicated(keep=False), "taxid"].unique().tolist()
        raise RuntimeError(f"跨类别TaxID重复，拒绝静默排除：{dup[:20]}")
    return X_all, meta_all, set()



def taxid_from_fasta_header(header):
    token = header.strip().lstrip(">").split("|", 1)[0].split()[0]
    if token.isdigit():
        return token
    match = re.search(r"(?:^|\s|[|])taxid\s*=\s*(\d+)(?:$|\s|[|])", header, flags=re.I)
    if match:
        return match.group(1)
    raise RuntimeError(f"无法从FASTA标题提取TaxID: {header[:160]}")


def audit_source_fasta(length, cls):
    """核对5kb/全长源FASTA；5kb还需每条恰好5000bp。"""
    suffix = ".fasta.gz" if cls == "Other" else ".fasta"
    path = DATA_DIR / f"{cls}_{length}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少用于核对特征TaxID的源FASTA: {path}；不跳过此项检查"
        )

    opener = gzip.open if path.suffix == ".gz" else open
    ids = set()
    n_records = 0
    header = None
    seq_len = 0

    def finish_record(current_header, current_len):
        if current_header is None:
            return
        tid = taxid_from_fasta_header(current_header)
        if tid in ids:
            raise RuntimeError(f"{path} 存在重复TaxID: {tid}")
        if current_len <= 0:
            raise RuntimeError(f"{path} 存在空序列: TaxID={tid}")
        if length == "5000bp" and current_len != 5000:
            raise RuntimeError(
                f"{path} 序列长度异常: TaxID={tid}, length={current_len}, expected=5000"
            )
        ids.add(tid)

    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    finish_record(header, seq_len)
                    n_records += 1
                header = line[1:].strip()
                seq_len = 0
            else:
                if header is None:
                    raise RuntimeError(f"{path}:{line_no} 在FASTA标题前出现序列内容")
                seq_len += len(line)
        if header is not None:
            finish_record(header, seq_len)
            n_records += 1

    if n_records != EXPECTED[cls]:
        raise RuntimeError(
            f"{length}/{cls} 源FASTA数量={n_records}, 预期={EXPECTED[cls]}: {path}"
        )
    print(f"[FASTA PASS] {length}/{cls}: {n_records}条；TaxID唯一"
          + ("；每条恰好5000bp" if length == "5000bp" else "；序列非空"))
    return ids


def check_features_against_source(meta):
    """检查每类的3mer TaxID集合与原始建模FASTA完全一致。"""
    for cls in CLASS_NAMES:
        matrix_ids = set(meta.loc[meta["class"] == cls, "taxid"].astype(str))
        fasta_ids = audit_source_fasta(LENGTH, cls)
        missing = sorted(fasta_ids - matrix_ids)
        extra = sorted(matrix_ids - fasta_ids)
        if missing or extra:
            raise RuntimeError(
                f"{LENGTH}/{cls}: 特征TaxID与源FASTA不一致；"
                f"缺失={len(missing)} {missing[:15]}；"
                f"多出={len(extra)} {extra[:15]}"
            )
        print(f"[ID MATCH PASS] {LENGTH}/{cls}: features IDs与源FASTA完全一致")



def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=SEED
            ))
        ]),
        "GaussianNB": GaussianNB(),
    }


def exclude_cross_host(X3, Xe, Xl, meta):
    """Once for every length: exclude the same 14 TaxIDs from ALL CV subsets."""
    matched = meta["taxid"].astype(str).isin(CROSS_HOST_TAXIDS)
    excluded = meta.loc[matched, ["taxid", "class"]].copy()
    if (excluded["class"] != "Insect").any():
        raise RuntimeError("跨宿主TaxID出现在非Insect类别，请先核对原数据")
    if excluded["taxid"].duplicated().any():
        raise RuntimeError("跨宿主名单中存在重复TaxID，禁止静默忽略")
    keep = (~matched).to_numpy()
    cleaned = meta.loc[keep].reset_index(drop=True)
    if cleaned["taxid"].isin(CROSS_HOST_TAXIDS).any():
        raise AssertionError("14个TaxID排除不完整")
    print(f"[EXCLUDE] 检查同一份{len(CROSS_HOST_TAXIDS)}个跨宿主TaxID清单，"
          f"本长度实际排除{len(excluded)}条，不参与训练或验证")
    print(f"[EXCLUDED TAXIDS] {sorted(excluded['taxid'].tolist(), key=int)}")
    return X3[keep], Xe[keep], Xl[keep], cleaned, excluded.reset_index(drop=True)


def calc_metrics(y_true, y_pred, y_prob):
    """Standard, single-label macro OvR metrics on non-cross-host samples."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    return {
        "AUC": roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"),
        "ACC": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def calc_confusion(y_true, y_pred):
    """Strict 3-class confusion matrix, rows=true and columns=predicted."""
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    class_rows = []
    for c, class_name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        tn = int(cm.sum()) - tp - fn - fp
        class_rows.append({"Class": class_name, "TP": tp, "TN": tn,
                           "FP": fp, "FN": fn})
    return cm, class_rows


def create_folds(y):
    """Ordinary stratified 5-fold CV, after the unified TaxID filter."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    valid = np.concatenate([va for _, va in folds])
    if len(valid) != len(y) or not np.array_equal(np.sort(valid), np.arange(len(y))):
        raise AssertionError("验证样本未恰好出现一次")
    print(f"[CV] StratifiedKFold({N_SPLITS}), eligible samples={len(y)}")
    return folds


def align_to(meta_ref, X, meta):
    idx = {t: i for i, t in enumerate(meta["taxid"].astype(str))}
    miss = [t for t in meta_ref["taxid"].astype(str) if t not in idx]
    if miss:
        raise RuntimeError(f"ESM2缺少TaxID: {len(miss)}，示例: {miss[:20]}")
    order = [idx[t] for t in meta_ref["taxid"].astype(str)]
    return X[order, :]


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="三分类模型训练统一入口：1000bp/3000bp/5000bp/full_length"
    )
    ap.add_argument("--length", required=True,
                    choices=("1000bp", "3000bp", "5000bp", "full_length"),
                    help="选择需要训练的长度段；每次只处理一个长度段")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="训练数据根目录，默认原服务器 数据保存/ 目录")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="可指定新的输出目录；已有非空目录拒绝覆盖")
    ap.add_argument("--check-only", action="store_true",
                    help="只核验特征、TaxID、源FASTA及五折分配，不训练、不写结果")
    ap.add_argument("--run", action="store_true", help="核验通过后正式训练并输出五折结果")
    args = ap.parse_args()
    if not (args.check_only or args.run) or (args.check_only and args.run):
        ap.error("请选择且仅选择 --check-only 或 --run")

    global LENGTH, DATA_DIR, FEATURE_DIR, OUTDIR, EXPECTED
    LENGTH = args.length
    DATA_DIR = args.data_root / LENGTH
    FEATURE_DIR = DATA_DIR / "features"
    OUTDIR = args.output_dir if args.output_dir is not None else DATA_DIR / "三分类模型结果"
    EXPECTED = EXPECTED_BY_LENGTH[LENGTH]
    print(f"[LENGTH] {LENGTH}")
    print(f"[DATA_DIR] {DATA_DIR}")
    print(f"[OUTDIR] {OUTDIR}")
    print("[1] Load 3mer")
    X3, m3, ov3 = build_feature("3mer")

    print("[2] Load ESM2")
    Xe, me, ove = build_feature("ESM2")

    print("[3] Load LucaVirus")
    Xl, ml, ovl = build_feature("LucaVirus")

    if not (ov3 == ove == ovl):
        raise RuntimeError(
            "3mer、ESM2、LucaVirus 的跨类重复TaxID集合不一致: "
            f"3mer={len(ov3)}, ESM2={len(ove)}, LucaVirus={len(ovl)}"
        )

    # 核对 TaxID 集合
    s3 = set(m3["taxid"].astype(str))
    se = set(me["taxid"].astype(str))
    sl = set(ml["taxid"].astype(str))

    if not (s3 == se == sl):
        raise RuntimeError(
            "3mer、ESM2、LucaVirus TaxID集合不一致: "
            f"only3mer_vs_ESM2={len(s3-se)}, "
            f"onlyESM2_vs_3mer={len(se-s3)}, "
            f"only3mer_vs_LucaVirus={len(s3-sl)}, "
            f"onlyLucaVirus_vs_3mer={len(sl-s3)}"
        )

    # Additional mapping consistency check for 1000bp features
    if LENGTH == "1000bp":
        class_ref = dict(zip(m3["taxid"].astype(str), m3["class"].astype(str)))
        for feature_name, m in (("ESM2", me), ("LucaVirus", ml)):
            class_map = dict(zip(m["taxid"].astype(str), m["class"].astype(str)))
            if class_map != class_ref:
                raise RuntimeError(f"{feature_name}与3mer的TaxID-类别映射不一致")

    # 按 3mer 顺序重排 ESM2 / LucaVirus
    Xe = align_to(m3, Xe, me)
    Xl = align_to(m3, Xl, ml)
    meta = m3.copy()

    # Source FASTA mapping is available/required for 5000bp and full_length.
    # Keep this verification before filtering the 14 cross-host TaxIDs.
    if LENGTH in ("5000bp", "full_length"):
        check_features_against_source(meta)

    # One shared 14-TaxID rule for all lengths. Apply *after* checking
    # original matrix/FASTA correspondence, and *before* any CV split.
    X3, Xe, Xl, meta, excluded_cross = exclude_cross_host(X3, Xe, Xl, meta)

    if X3.shape[1] != 64:
        raise RuntimeError(f"3mer维度异常: {X3.shape}")
    if Xe.shape[1] != 320:
        raise RuntimeError(f"ESM2维度异常: {Xe.shape}")
    if Xl.shape[1] != 2560:
        raise RuntimeError(f"LucaVirus维度异常: {Xl.shape}")

    overlap_df = pd.DataFrame({"taxid": sorted(ov3)})

    counts = (
        meta["class"].value_counts()
        .reindex(["Insect", "Plant", "Other"])
        .fillna(0).astype(int)
    )

    expected_after_filter = dict(EXPECTED)
    expected_after_filter["Insect"] -= len(excluded_cross)
    if counts.to_dict() != expected_after_filter:
        raise RuntimeError(f"排除14个TaxID后类别数量异常：{counts.to_dict()} vs {expected_after_filter}")
    print("Class counts:")
    print(counts.to_string())
    print("Excluded cross-host TaxIDs:", len(excluded_cross))
    print("Excluded overlaps:", len(ov3))
    print("3mer shape:", X3.shape)
    print("ESM2 shape:", Xe.shape)
    print("LucaVirus shape:", Xl.shape)

    cmap = {"Insect": 0, "Plant": 1, "Other": 2}
    y = meta["class"].map(cmap).values.astype(int)

    folds = create_folds(y)

    if args.check_only:
        print(f"[CHECK PASS] {LENGTH} 特征、14个TaxID排除及五折划分正常；未训练、未写结果")
        return

    if OUTDIR.exists() and any(OUTDIR.iterdir()):
        raise FileExistsError(f"输出目录已有结果，拒绝覆盖：{OUTDIR}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    overlap_df.to_csv(OUTDIR / "excluded_overlap_taxids.tsv", sep="\t", index=False)
    excluded_cross.to_csv(OUTDIR / "excluded_cross_host_taxids.tsv", sep="\t", index=False)
    counts.rename_axis("class").reset_index(name="n").to_csv(
        OUTDIR / "class_counts.tsv", sep="\t", index=False
    )

    feature_sets = {"3mer": X3, "ESM2": Xe, "LucaVirus": Xl}

    fold_rows = []
    summary_rows = []
    pred_rows = []
    fold_class_rows = []
    oof_class_rows = []
    fold_cm_rows = []
    oof_cm_rows = []

    for feat_name, X in feature_sets.items():
        for model_name in get_models():

            print(f"\n[RUN] {feat_name} + {model_name}")

            metrics_this = []
            oof_pred = np.full(len(y), -1, dtype=int)
            oof_prob = np.full((len(y), 3), np.nan, dtype=float)
            oof_fold = np.full(len(y), -1, dtype=int)

            for fold, (tr, va) in enumerate(folds, start=1):
                Xtr, ytr = X[tr], y[tr]
                Xva, yva = X[va], y[va]

                before = np.bincount(ytr, minlength=3)

                rus = RandomUnderSampler(
                    sampling_strategy="all",
                    random_state=SEED + fold
                )
                Xbal, ybal = rus.fit_resample(Xtr, ytr)

                after = np.bincount(ybal, minlength=3)
                valc = np.bincount(yva, minlength=3)

                print(
                    f" Fold {fold}: "
                    f"train_before={before.tolist()} "
                    f"train_after={after.tolist()} "
                    f"val={valc.tolist()}"
                )

                model = get_models()[model_name]
                model.fit(Xbal, ybal)

                prob = model.predict_proba(Xva)

                # 统一概率列顺序到 0,1,2
                if hasattr(model, "classes_"):
                    cls = model.classes_
                else:
                    cls = model.named_steps["clf"].classes_

                p3 = np.zeros((len(va), 3), dtype=float)
                for j, c in enumerate(cls):
                    p3[:, int(c)] = prob[:, j]

                # 用对齐后的三类概率选择最终类别，与导出的概率一致
                pred = np.argmax(p3, axis=1)

                oof_pred[va] = pred
                oof_prob[va] = p3
                oof_fold[va] = fold

                mm = calc_metrics(yva, pred, p3)
                metrics_this.append(mm)

                cm, class_rows = calc_confusion(yva, pred)
                for row in class_rows:
                    fold_class_rows.append({
                        "Feature": feat_name,
                        "Model": model_name,
                        "Fold": fold,
                        **row,
                    })
                for c, class_name in enumerate(CLASS_NAMES):
                    fold_cm_rows.append({
                        "Feature": feat_name,
                        "Model": model_name,
                        "Fold": fold,
                        "TrueClass": class_name,
                        "Pred_Insect": int(cm[c, 0]),
                        "Pred_Plant": int(cm[c, 1]),
                        "Pred_Other": int(cm[c, 2]),
                    })

                fold_rows.append({
                    "Feature": feat_name,
                    "Model": model_name,
                    "Fold": fold,
                    **mm,
                    "train_Insect_before": int(before[0]),
                    "train_Plant_before": int(before[1]),
                    "train_Other_before": int(before[2]),
                    "train_Insect_after": int(after[0]),
                    "train_Plant_after": int(after[1]),
                    "train_Other_after": int(after[2]),
                    "val_Insect": int(valc[0]),
                    "val_Plant": int(valc[1]),
                    "val_Other": int(valc[2]),
                })

            summary_rows.append({
                "Feature": feat_name,
                "Model": model_name,
                "AUC": np.mean([m["AUC"] for m in metrics_this]),
                "ACC": np.mean([m["ACC"] for m in metrics_this]),
                "Recall": np.mean([m["Recall"] for m in metrics_this]),
                "Precision": np.mean([m["Precision"] for m in metrics_this]),
                "F1": np.mean([m["F1"] for m in metrics_this]),
            })

            # 每个样本只使用其验证折的预测，统计全部 OOF 样本的实际计数
            cm, class_rows = calc_confusion(y, oof_pred)
            for row in class_rows:
                oof_class_rows.append({
                    "Feature": feat_name,
                    "Model": model_name,
                    **row,
                })
            for c, class_name in enumerate(CLASS_NAMES):
                oof_cm_rows.append({
                    "Feature": feat_name,
                    "Model": model_name,
                    "TrueClass": class_name,
                    "Pred_Insect": int(cm[c, 0]),
                    "Pred_Plant": int(cm[c, 1]),
                    "Pred_Other": int(cm[c, 2]),
                })

            # Excluded cross-host TaxIDs have no OOF rows; strict single-label evaluation.
            y_eval = y
            ovr_outcomes = np.empty((len(y), 3), dtype="U2")
            for c in range(3):
                ovr_outcomes[:, c] = np.where(
                    y_eval == c,
                    np.where(oof_pred == c, "TP", "FN"),
                    np.where(oof_pred == c, "FP", "TN"),
                )

            for i in range(len(y)):
                pred_rows.append({
                    "Feature": feat_name,
                    "Model": model_name,
                    "Fold": int(oof_fold[i]),
                    "taxid": meta.iloc[i]["taxid"],
                    "true_class": meta.iloc[i]["class"],
                    "eval_true_class": CLASS_NAMES[y_eval[i]],
                    "allowed_classes": meta.iloc[i]["class"],
                    "pred_class": CLASS_NAMES[oof_pred[i]],
                    "is_correct": int(y_eval[i] == oof_pred[i]),
                    "Insect_OvR": ovr_outcomes[i, 0],
                    "Plant_OvR": ovr_outcomes[i, 1],
                    "Other_OvR": ovr_outcomes[i, 2],
                    "prob_Insect": oof_prob[i, 0],
                    "prob_Plant": oof_prob[i, 1],
                    "prob_Other": oof_prob[i, 2],
                })

    summary = pd.DataFrame(summary_rows)[
        ["Feature","Model","AUC","ACC","Recall","Precision","F1"]
    ]

    summary.to_csv(
        OUTDIR / "feature_model_summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f"
    )

    pd.DataFrame(fold_rows).to_csv(
        OUTDIR / "fivefold_results.tsv",
        sep="\t",
        index=False,
        float_format="%.6f"
    )

    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(
        OUTDIR / "oof_predictions.tsv",
        sep="\t",
        index=False,
        float_format="%.6f"
    )

    predictions.loc[predictions["is_correct"] == 0].to_csv(
        OUTDIR / "misclassified_samples.tsv",
        sep="\t", index=False, float_format="%.6f"
    )

    pd.DataFrame(fold_class_rows).to_csv(
        OUTDIR / "fivefold_class_confusion.tsv",
        sep="\t", index=False
    )
    pd.DataFrame(oof_class_rows).to_csv(
        OUTDIR / "oof_class_confusion.tsv",
        sep="\t", index=False
    )
    pd.DataFrame(fold_cm_rows).to_csv(
        OUTDIR / "fivefold_confusion_matrix.tsv",
        sep="\t", index=False
    )
    pd.DataFrame(oof_cm_rows).to_csv(
        OUTDIR / "oof_confusion_matrix.tsv",
        sep="\t", index=False
    )

    with open(OUTDIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "task": "Insect vs Plant vs Other",
            "length": LENGTH,
            "features": ["3mer","ESM2","LucaVirus"],
            "models": ["RandomForest","KNN","SVM","GaussianNB"],
            "cv": "StratifiedKFold(5)",
            "excluded_cross_host_taxids": sorted(CROSS_HOST_TAXIDS),
            "excluded_cross_host_in_this_length": sorted(excluded_cross["taxid"].tolist()),
            "cross_host_policy": "One 14-TaxID set; exclude before CV from training and validation",
            "AUC_cross_host": "Cross-host cases excluded, conventional single-label macro OvR",
            "confusion_cross_host": "Cross-host cases excluded; ordinary single-label confusion matrix",
            "balance": "RandomUnderSampler inside training fold only",
            "validation_balance": "none",
            "prediction": "argmax of aligned class probabilities",
            "confusion_matrix_axes": "rows=true class, columns=predicted class",
            "confusion_class_order": CLASS_NAMES,
            "class_confusion": "TP/TN/FP/FN computed per class vs rest",
            "oof_confusion": "counts over all out-of-fold samples, not fold averages",
            "is_correct": "1=correct, 0=incorrect",
            "AUC": "macro OvR",
            "Recall": "macro",
            "Precision": "macro",
            "F1": "macro",
            "feature_dimensions": {"3mer": 64, "ESM2": 320, "LucaVirus": 2560},
            "seed": SEED,
            "unified_source": "one train.py for 1000bp/3000bp/5000bp/full_length",
            "eligible_class_counts_after_filter": counts.to_dict(),
        }, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    ))
    print("\nOutput:", OUTDIR)


if __name__ == "__main__":
    main()
