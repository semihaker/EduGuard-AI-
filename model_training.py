"""
AI-Supported Student Focus & Risk Analysis — Model eğitimi

processed_data.csv okunur; LR, RF, KNN, SVM karşılaştırılır, Voting ensemble
üretilir. Canlı risk skoru için LR (risk_model.pkl), en iyi genel model (best_model.pkl).
"""

# --- Üç dosya — hangi iş nerede? (sunumda sık sorulan) ---
# data_preprocessing.py : Görüntü → MediaPipe → EAR/MAR/pitch/yaw → CSV (+ IQR, grafik). Sınıflandırıcı YOK.
# model_training.py     : BU DOSYA. CSV → train/test → LR/RF/KNN/SVM + VotingClassifier → .pkl + outputs/.
# main_app.py           : Webcam → aynı özellikler → risk_model.pkl (LogReg) ile canlı risk; best_model varsayılan değil.

# --- Dosya haritası (sunum / kod okuma) ---
# 1) Özellik sütunları + etiket → ikili y (infer_binary_labels)
# 2) Test metrikleri (metrics_row) ve çıktı grafikleri (CM, çubuk, heatmap, model_comparison)
# 3) Taban modeller (build_estimators): Pipeline + StandardScaler
# 4) main: CSV / otomatik ön işlem → train_test_split → 4 model döngüsü → VotingClassifier (soft) → best.pkl / risk.pkl
#    VotingClassifier sklearn.ensemble içindedir; burada yeni algoritma yazılmaz — en iyi 3 tabanın olasılık ortalaması kurulur.

# --- Sunum özeti ---
# Ne yapar: CSV’deki ham özelliklerle sınıflandırıcıları eğitir, karşılaştırır, .pkl kaydeder.
# Akış: Train/test ayrımı → Pipeline (ölçekleme sadece train’de) → metrik + karmaşıklık matrisi.
# Çıktı: risk_model.pkl (LogReg, canlı risk), best_model.pkl, outputs/*.png.
# Not: Ön işlemedeki z-score sütunları eğitimde kullanılmaz; ölçekleme model içinde split ile yapılır.

from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------------------------------------------------------
# Bölüm 1 — Girdi özellikleri (CSV ile data_preprocessing çıktısı uyumlu)
# -----------------------------------------------------------------------------
# Sunum: Sadece ham sütunlar (+ isteğe bağlı keyboard). data_preprocessing'teki *_z burada
# kullanılmaz; ölçekleme StandardScaler ile Pipeline içinde ve yalnızca train split üzerinde fit edilir.
# keyboard: görüntüden üretilen CSV'de genelde 0; canlı uygulamada tuş sinyali 1 olabilir.
FEATURE_COLS = ("ear", "mar", "pitch_deg", "yaw_deg", "keyboard")


# -----------------------------------------------------------------------------
# Bölüm 2 — Etiketler: klasör adlarından otomatik ikili sınıf (0 = düşük risk, 1 = risk)
# -----------------------------------------------------------------------------
def _normalize_label_token(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s).strip().lower())


def infer_binary_labels(series: pd.Series) -> Tuple[np.ndarray, str, str]:
    # Sunum: Klasör adlarından otomatik ikili sınıf (drowsy=1); yoksa alfabetik ikinci sınıf pozitif.
    """
    İki sınıf varsayımı: pozitif (risk) = 'drowsy' içerir ve 'non' içermez.
    Aksi halde alfabetik sıralamanın ikinci sınıfı pozitif kabul edilir.
    """
    raw = series.astype(str)
    tokens = raw.map(_normalize_label_token)
    uniq = sorted(tokens.unique())
    if len(uniq) < 2:
        raise SystemExit("En az iki sınıf gerekli (label).")
    pos_mask = tokens.str.contains("drowsy") & ~tokens.str.contains("non")
    if pos_mask.any() and (~pos_mask).any():
        y = pos_mask.astype(np.int32).to_numpy()
        neg_lab = raw[~pos_mask].iloc[0]
        pos_lab = raw[pos_mask].iloc[0]
        return y, str(neg_lab), str(pos_lab)
    if len(uniq) != 2:
        raise SystemExit("Otomatik ikili etiket çıkarılamadı; label sütununu kontrol edin.")
    neg_t, pos_t = uniq[0], uniq[1]
    y = (tokens == pos_t).astype(np.int32).to_numpy()
    neg_lab = raw[tokens == neg_t].iloc[0]
    pos_lab = raw[tokens == pos_t].iloc[0]
    return y, str(neg_lab), str(pos_lab)


# -----------------------------------------------------------------------------
# Bölüm 3 — Test kümesi metrikleri (ikili; pozitif sınıf = 1 = risk)
# -----------------------------------------------------------------------------
def metrics_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str,
    pos_label: int | None,
) -> Dict[str, float]:
    kwargs: Dict[str, Any] = {"zero_division": 0}
    if average == "binary":
        kwargs["pos_label"] = pos_label
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, average=average, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, average=average, **kwargs)),
    }


# -----------------------------------------------------------------------------
# Bölüm 4 — Rapor grafikleri: karmaşıklık matrisi, metrik çubukları, en iyi model heatmap
# -----------------------------------------------------------------------------
def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=list(display_labels),
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics_comparison_chart(summary: pd.DataFrame, out_path: Path) -> None:
    # Sunum: Taban modeller + voting_ensemble birlikte; F1 seçimde kullanılır ama bu grafikte yok.
    """Taban modeller + ensemble: accuracy, precision, recall (F1 grafikte yok)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    cols = [c for c in ("accuracy", "precision", "recall") if c in summary.columns]
    summary[cols].plot(kind="bar", ax=ax, width=0.85)
    ax.set_title("Model karsilastirmasi (test kumesi)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    ax.legend(loc="lower right", fontsize=8)
    ax.tick_params(axis="x", rotation=22)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_best_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    try:
        import seaborn as sns
    except ImportError as e:
        raise SystemExit(
            "Seaborn gerekli: pip install seaborn\n%s" % e
        ) from e

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# Görselleştirmede eksen etiketleri (FEATURE_COLS ile aynı sıra)
FEATURE_DISPLAY = ("EAR", "MAR", "Pitch", "Yaw", "Keyboard")


def _feature_importance_values(model: Any, model_key: str) -> Tuple[np.ndarray, str] | None:
    if model_key == "random_forest" and hasattr(model, "named_steps"):
        return np.asarray(model.named_steps["clf"].feature_importances_, dtype=float), "RF importance"
    if model_key == "logistic_regression" and hasattr(model, "named_steps"):
        coef = model.named_steps["clf"].coef_
        return np.abs(np.asarray(coef, dtype=float).ravel()), "|katsayi| (olcekli)"
    return None


def save_model_comparison_and_importance_png(
    summary: pd.DataFrame,
    best_model: Any,
    best_name: str,
    out_path: Path,
) -> None:
    """Tek PNG: ustte 4 model Accuracy, altta (destekleniyorsa) ozellik onemi."""
    keys = ["logistic_regression", "random_forest", "knn", "svm"]
    labels = ["Logistic Regression", "Random Forest", "KNN", "SVM"]
    acc = [float(summary.loc[k, "accuracy"]) for k in keys]
    x = np.arange(len(keys))
    w = 0.5

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[1.15, 1.0])
    ax0.bar(x, acc, w, label="Accuracy", color="steelblue")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=12, ha="right")
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("Skor")
    ax0.set_title("Model karsilastirmasi (test) — Accuracy")
    ax0.legend(loc="lower right")
    ax0.grid(True, axis="y", alpha=0.25)

    fi = _feature_importance_values(best_model, best_name)
    if fi is not None:
        vals, xlab = fi
        order = np.argsort(vals)
        y_pos = np.arange(len(vals))
        labels_fi = [FEATURE_DISPLAY[i] if i < len(FEATURE_DISPLAY) else "f%d" % i for i in order]
        ax1.barh(y_pos, vals[order], color="teal", alpha=0.88)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels_fi)
        ax1.set_xlabel("Onem (%s)" % xlab)
        ax1.set_title(
            "Ozellik onemi — en iyi model: %s"
            % best_name.replace("_", " ")
        )
        ax1.grid(True, axis="x", alpha=0.25)
    else:
        ax1.text(
            0.5,
            0.5,
            "Ozellik onemi: secilen model '%s'\n(RF veya LogReg degil — gosterilmez)."
            % best_name,
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=11,
        )
        ax1.set_axis_off()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Bölüm 5 — Dört taban sınıflandırıcı (hepsi Pipeline: ölçekleme + clf)
# -----------------------------------------------------------------------------
# Sunum: LR/RF/SVM'de class_weight='balanced' sınıf dengesizliğine karşı. SVM'de probability=True
# soft voting için predict_proba şart. KNN mesafe tabanlı; ağırlıklı komşu ile yerel uyum.
def build_estimators(random_state: int) -> Dict[str, Any]:
    """Tüm taban modeller Pipeline; soft voting için predict_proba gerekli."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


# -----------------------------------------------------------------------------
# Bölüm 6 — CSV yokken görüntü kökü; ensemble için F1'e göre ilk k modeller
# -----------------------------------------------------------------------------
def default_images_root(script_dir: Path) -> Path | None:
    """Önce data/train (sınıf alt klasörleri burada), yoksa data kökü."""
    train = script_dir / "data" / "train"
    if train.is_dir():
        return train
    data = script_dir / "data"
    if data.is_dir():
        return data
    return None


def pick_top_for_voting(
    f1_by_name: Dict[str, float],
    k: int = 3,
) -> List[str]:
    # Sunum: Test F1'e göre en iyi k taban model; VotingClassifier'a en az 2 estimator gerekir.
    ordered = sorted(f1_by_name.items(), key=lambda x: x[1], reverse=True)
    names = [n for n, _ in ordered[:k]]
    if len(names) < 2:
        names = [n for n, _ in ordered[:2]]
    return names


# =============================================================================
# Bölüm 7 — Program girişi: CSV → train/test → modeller → ensemble → .pkl + outputs/
# =============================================================================
def main() -> None:
    # Sunum: Windows konsolunda Türkçe karakter için UTF-8 (başarısız olursa sessiz geç).
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Öğrenci odak/risk — model eğitimi.")
    parser.add_argument(
        "--data",
        type=str,
        default="processed_data.csv",
        help="Ön işlenmiş CSV (ear, mar, pitch_deg, yaw_deg, label).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out test oranı.",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="outputs",
        help="Confusion matrix görselleri.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="CSV yoksa: görüntü kökü (alt klasörler = sınıf). Varsayılan otomatik: data/train veya data.",
    )
    parser.add_argument(
        "--landmarker-model",
        type=str,
        default=None,
        help="CSV yokken ön işlemde kullanılacak face_landmarker.task yolu.",
    )
    parser.add_argument(
        "--no-auto-preprocess",
        action="store_true",
        help="CSV yoksa otomatik ön işlem yapma.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # --- 7a) CSV yoksa: data_preprocessing.py subprocess (tek komutla eğitim akışı) ---
    csv_path = Path(args.data)
    # Sunum: Özellik çıkarma bu dosyada değil — data_preprocessing.py subprocess ile çağrılır.
    # Aynı Python ile çalıştırılır; --no-auto-preprocess ile sadece CSV zorunlu hale getirilir.
    if not csv_path.is_file():
        script_dir = Path(__file__).resolve().parent
        img_root: Path | None = None
        if args.images_root:
            img_root = Path(args.images_root)
        elif not args.no_auto_preprocess:
            img_root = default_images_root(script_dir)
        if img_root is not None and img_root.is_dir():
            prep_script = script_dir / "data_preprocessing.py"
            if not prep_script.is_file():
                raise SystemExit(f"data_preprocessing.py bulunamadı: {prep_script}")
            cmd = [
                sys.executable,
                str(prep_script),
                "--data-root",
                str(img_root),
                "--output",
                str(csv_path.resolve()),
            ]
            if args.landmarker_model:
                cmd.extend(["--model", str(Path(args.landmarker_model).resolve())])
            print(
                "CSV yok; önce özellik çıkarımı çalıştırılıyor:\n  %s" % " ".join(cmd),
                flush=True,
            )
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise SystemExit(
                    "Ön işlem başarısız (çıkış kodu %s). Yaygın neden: "
                    "models/face_landmarker.task eksik.\nİndirin ve şuna koyun:\n  %s\n"
                    "veya: python model_training.py --landmarker-model TAM_YOL.task"
                    % (e.returncode, script_dir / "models" / "face_landmarker.task")
                ) from e
        if not csv_path.is_file():
            raise SystemExit(
                "CSV bulunamadı: %s\n"
                "Görüntü yapısı: data/train/drowsy, data/train/nondrowsy (veya data/drowsy …).\n"
                "  python data_preprocessing.py --data-root data/train --output processed_data.csv\n"
                "  python model_training.py --images-root data/train\n"
                "Otomatik: data/train varsa öncelik ona verilir."
                % (csv_path.resolve(),)
            )

    # --- 7b) Veri çerçevesi: X (özellik matrisi), y (ikili etiket), sınıf adları grafikler için ---
    df = pd.read_csv(csv_path)
    if "keyboard" not in df.columns:
        df["keyboard"] = 0.0
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Eksik sütunlar: {missing}")
    if "label" not in df.columns:
        raise SystemExit("label sütunu gerekli.")

    X = df.loc[:, list(FEATURE_COLS)].to_numpy(dtype=np.float64)
    y, neg_name, pos_name = infer_binary_labels(df["label"])
    if np.unique(y).size != 2:
        raise SystemExit("İkili sınıflandırma için tam iki sınıf gerekli.")

    average = "binary"
    pos_label = 1

    # --- 7c) Hold-out test: stratify ile sınıf oranı korunur; Scaler sızıntısı olmaması için Pipeline ---
    # Sunum: stratify=y hem sınıfların train/test dağılımını dengeler hem raporu anlamlı tutar.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    out_dir = Path(args.outputs)
    os.makedirs(str(out_dir), exist_ok=True)

    estimators = build_estimators(args.random_state)
    fitted: Dict[str, Any] = {}
    f1_scores: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []

    display_labels = (neg_name, pos_name)

    # --- 7d) Dört taban model: clone ile bağımsız fit, test metrikleri, model_cm.png ---
    # Sunum: Her model sadece train üzerinde fit; Pipeline içindeki Scaler da yalnızca train istatistiği.
    for name, est in estimators.items():
        model = clone(est)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = metrics_row(y_test, y_pred, average=average, pos_label=pos_label)
        fitted[name] = model
        f1_scores[name] = m["f1"]
        m["model"] = name
        rows.append(m)

        fname = name.replace(" ", "_") + "_cm.png"
        save_confusion_matrix(
            y_test,
            y_pred,
            display_labels,
            out_dir / fname,
            title=f"{name} — Confusion Matrix",
        )

    # --- 7e) VotingClassifier (sklearn.ensemble): yeni algoritma yazılmaz; 3 taban modelin soft birleşimi ---
    # Sunum: Test F1 sıralamasıyla ilk 3 seçilir; predict_proba ortalaması ile sınıf seçilir.
    top_names = pick_top_for_voting(f1_scores, k=3)
    vote_estimators = [(n.replace(" ", "_"), clone(fitted[n])) for n in top_names]
    # Sunum: voting="soft" — LR/RF/KNN/SVM proba üretebilmeli (SVM: probability=True).
    ensemble = VotingClassifier(estimators=vote_estimators, voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred_v = ensemble.predict(X_test)
    m_v = metrics_row(y_test, y_pred_v, average=average, pos_label=pos_label)
    m_v["model"] = "voting_ensemble"
    rows.append(m_v)
    f1_scores["voting_ensemble"] = m_v["f1"]
    fitted["voting_ensemble"] = ensemble

    save_confusion_matrix(
        y_test,
        y_pred_v,
        display_labels,
        out_dir / "voting_ensemble_cm.png",
        title="VotingClassifier (soft) — Confusion Matrix",
    )

    # --- 7f) Özet tablo + accuracy/precision/recall çubuk grafik; konsola metrik yazdırma ---
    summary = pd.DataFrame(rows).set_index("model")[
        ["accuracy", "precision", "recall", "f1"]
    ]
    save_metrics_comparison_chart(summary, out_dir / "model_metrics_comparison.png")

    print("\n=== Test metrikleri (pozitif sınıf = risk: %r) ===" % (pos_name,))
    print(summary.drop(columns=["f1"], errors="ignore").round(4).to_string())
    print("\nEnsemble icin secilen modeller (ilk 3):", top_names)

    # --- 7g) En iyi model: taban + voting_ensemble arasında test F1 maksimum; confusion_matrix.png ---
    # Sunum: Seçim kriteri F1; model_metrics_comparison grafiğinde F1 çubuğu yok ama mantık aynı.
    best_name = max(f1_scores.items(), key=lambda kv: kv[1])[0]
    y_pred_best = fitted[best_name].predict(X_test)
    save_best_confusion_heatmap(
        y_test,
        y_pred_best,
        display_labels,
        out_dir / "confusion_matrix.png",
        "Hata matrisi — %s (test)" % best_name,
    )

    # --- 7h) best_model.pkl: kazanan mimari tüm (X,y) ile yeniden eğitilir; model_comparison.png ---
    # Sunum: Dağılım için maksimum veri; canlıda varsayılan risk_model.pkl farklı — bkz. 7i ve kayıt yorumları.
    best_model = clone(fitted[best_name])
    best_model.fit(X, y)
    save_model_comparison_and_importance_png(
        summary,
        best_model,
        best_name,
        out_dir / "model_comparison.png",
    )

    # --- 7i) risk_model.pkl: her zaman Logistic Regression + tüm veri (hızlı tahmin, predict_proba[:,1] risk) ---
    # Sunum: best_model testte ne kazanırsa kazansın, canlı risk skoru için ayrı politika: sabit LR.
    lr_full = clone(estimators["logistic_regression"])
    lr_full.fit(X, y)

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("risk_model.pkl", "wb") as f:
        pickle.dump(lr_full, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Sunum: best_model.pkl = test F1 şampiyonu (LR/RF/KNN/SVM veya voting_ensemble), tüm veriyle fit.
    # risk_model.pkl = daima Logistic Regression Pipeline; main_app.py varsayılan olarak bunu yükler.
    # İstenirse: python main_app.py --risk-model best_model.pkl ile başka mimari yüklenebilir.
    print("\nKaydedildi: best_model.pkl (%s, tüm veriyle yeniden eğitildi)" % best_name)
    print("Kaydedildi: risk_model.pkl (Logistic Regression — canlı risk skoru: predict_proba[:, 1])")


if __name__ == "__main__":
    # Doğrudan çalıştırma: python model_training.py [--data ...]
    main()
