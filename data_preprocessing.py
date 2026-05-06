#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Supported Student Focus & Risk Analysis — Veri ön işleme

MediaPipe Face Landmarker ile EAR, MAR, baş pozu (pitch, yaw) çıkarımı;
IQR ile aykırı değer temizliği; KNN/SVM için StandardScaler ile normalizasyon.
"""

# --- Sunum özeti ---
# Ne yapar: Klasördeki yüz görüntülerinden sayısal özellik çıkarır → CSV + grafik.
# Akış: Görüntü listesi → MediaPipe yüz → EAR/MAR/pitch/yaw → IQR → z-score sütunları.
# Çıktı: processed_data.csv (ham + _z), outputs/outlier_analysis.png (ve isteğe bağlı PNG).
# Konsol: --progress-every ile ilerleme; fence dışı satır sayıları özetlenir.

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Akademik notlar (Hoca sorarsa diye)
# -----------------------------------------------------------------------------
# MAR (Mouth Aspect Ratio) neden kullanılır?
#   Uykululuk ve yorgunlukta yüz kas tonusu düşer; alt çene gevşeyerek ağız
#   dikey olarak daha fazla açılır (esneme, uyku öncesi ağız açıklığı artışı).
#   EAR yalnızca göz kapak dinamiğini özetler; MAR ise oral-fasiyal bölgedeki
#   bu değişimi nicel olarak yakalar. Bu nedenle çok modlu (göz + ağız) öznitelik
#   vektörü, sınıflandırıcıların ayırıcılığını artırır (daha düşük yanlış alarm).
#
# EAR (Eye Aspect Ratio) — Soukupová & Čech (2016) çerçevesi:
#   Göz açıklığının dikey/ yatay oranı; göz kapanınca düşer — blink/perclos
#   ile ilişkilidir.
#
# Pitch / Yaw (baş pozu):
#   Dikkat dağılması veya uykuya eğilimde baş öne eğilir (pitch) veya yana
#   döner (yaw); bu yüzden mesafe tabanlı sınıflandırıcılara tamamlayıcı sinyal
#   sağlar. solvePnP + 3D yüz modeli yaygın bir yaklaşımdır.
#
# IQR ile aykırı değer:
#   Landmark gürültüsü, kısmi yüz, motion blur tek tek özelliklerde aşırı
#   değer üretebilir; Tukey fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR) sağlam bir
#   tek değişkenli filtreleme sağlar.
#
# StandardScaler (KNN, SVM):
#   Bu modeller öklidyen/çekirdek mesafesine dayanır; ölçek farkı (ör. yaw
#   derece, EAR 0–0.4) büyük özelliğin domine etmesine yol açar. Z-score
#   normalizasyonu (μ=0, σ=1) özellikleri karşılaştırılabilir hale getirir.
# -----------------------------------------------------------------------------

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:
    raise SystemExit(
        "mediapipe paketi gerekli: pip install mediapipe\n" + str(e)
    ) from e

# MediaPipe Face Mesh topolojisinde yaygın kullanılan indeksler (478 nokta ile uyumlu)
# Kaynak: Google MediaPipe Face Landmarker / Face Mesh şemaları ve drowsiness literatürü
LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
# MAR: ağız köşeleri (genişlik) + üst-alt dudak dikey çiftleri (478 topolojisi)
MAR_VERTICAL_PAIRS = ((78, 88), (81, 95), (13, 14))
MAR_CORNERS = (61, 291)

# solvePnP için 2D/3D eşleşen referans noktalar (yaklaşık 3D yüz modeli — OpenCV head pose örneği)
POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

# -----------------------------------------------------------------------------
# Geometri: EAR / MAR / baş pozu (landmark koordinatlarından sayısal özellik)
# -----------------------------------------------------------------------------


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# Görüntü I/O: cv2.imread bazen Unicode yollarda başarısız; np.fromfile + imdecode yedek yol.


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Windows + Türkçe yol: cv2.imread bazen None döner; np.fromfile + imdecode güvenilir."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def eye_aspect_ratio(coords: np.ndarray, indices: Tuple[int, ...]) -> float:
    """İki göz için ortalama EAR (6 nokta formülü)."""
    p = [coords[i] for i in indices]
    v1 = _euclid(p[1], p[5])
    v2 = _euclid(p[2], p[4])
    h = _euclid(p[0], p[3])
    if h < 1e-8:
        return float("nan")
    return (v1 + v2) / (2.0 * h)


def mouth_aspect_ratio(coords: np.ndarray) -> float:
    """
    MAR: üst/alt dudak dikey mesafelerinin ortalamasının, ağız köşeleri
    arası genişliğe oranı. Uykululukta dikey açılım artışına duyarlıdır
    (bkz. dosya başındaki MAR akademik notu).
    """
    lc, rc = MAR_CORNERS
    horiz = _euclid(coords[lc], coords[rc])
    if horiz < 1e-8:
        return float("nan")
    verts = [_euclid(coords[u], coords[lo]) for u, lo in MAR_VERTICAL_PAIRS]
    return float(np.mean(verts) / horiz)


def rotation_vector_to_euler_angles(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Derece cinsinden roll, pitch, yaw (OpenCV solvePnP eksen düzeni)."""
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
        pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def head_pose_pitch_yaw(
    landmarks_norm: Iterable,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float]:
    """
    Normalize [0,1] landmark'lardan piksel koordinatı; solvePnP ile pitch/yaw/roll.
    Kamera iç parametreleri bilinmediğinde görüntü merkezine yakın odak uzunluğu yaklaşımı kullanılır.
    """
    h, w = image_height, image_width
    idx_order = [
        POSE_LANDMARK_IDX["nose_tip"],
        POSE_LANDMARK_IDX["chin"],
        POSE_LANDMARK_IDX["left_eye_outer"],
        POSE_LANDMARK_IDX["right_eye_outer"],
        POSE_LANDMARK_IDX["mouth_left"],
        POSE_LANDMARK_IDX["mouth_right"],
    ]
    pts = []
    lm_list = list(landmarks_norm)
    for i in idx_order:
        lm = lm_list[i]
        pts.append([lm.x * w, lm.y * h])
    image_points = np.array(pts, dtype=np.float64)

    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return float("nan"), float("nan"), float("nan")
    roll, pitch, yaw = rotation_vector_to_euler_angles(rvec)
    return roll, pitch, yaw

# -----------------------------------------------------------------------------
# MediaPipe: .task yolu (Unicode/Windows) ve FaceLandmarker oluşturma
# -----------------------------------------------------------------------------


def resolve_mediapipe_model_path(model_path: str) -> str:
    """
    MediaPipe Tasks (Windows) Unicode yol adlarında (ör. Masaüstü) .task açamayabiliyor.
    Yol ASCII değilse veya Windows'ta güvenli yükleme için model TEMP altına kopyalanır.
    """
    src = Path(model_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(str(src))
    if src.stat().st_size < 100_000:
        raise ValueError(
            "Model dosyasi cok kucuk (%s bayt); bos veya hatali indirme olabilir."
            % src.stat().st_size
        )

    use_direct = str(src)
    if sys.platform == "win32":
        try:
            use_direct.encode("ascii")
        except UnicodeEncodeError:
            use_direct = ""
        if not use_direct:
            dest = Path(tempfile.gettempdir()) / "mediapipe_face_landmarker.task"
            if (
                not dest.is_file()
                or dest.stat().st_size != src.stat().st_size
            ):
                shutil.copy2(src, dest)
            return str(dest.resolve())

    return str(src)


def build_face_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    resolved = resolve_mediapipe_model_path(model_path)
    base = mp_python.BaseOptions(model_asset_path=resolved)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

# -----------------------------------------------------------------------------
# Veri taraması: kök altındaki görüntüler ve sınıf etiketi (ilk alt klasör adı)
# -----------------------------------------------------------------------------


def iter_images(root: Path) -> Iterable[Tuple[Path, str]]:
    """(yol, etiket): köke göre ilk alt klasör = sınıf (örn. train/drowsy/a.jpg -> drowsy)."""
    root = root.resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff", ".gif"}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() not in exts:
                continue
            try:
                rel = p.resolve().relative_to(root)
            except ValueError:
                yield p, Path(dirpath).name
                continue
            if len(rel.parts) >= 2:
                yield p, rel.parts[0]
            else:
                yield p, Path(dirpath).name


# -----------------------------------------------------------------------------
# Özellik çıkarımı (feature extraction): tek görüntü → ear, mar, pitch_deg, yaw_deg
# MediaPipe yüz landmark → EAR/MAR formülleri + solvePnP ile baş açıları (derece).
# -----------------------------------------------------------------------------
def extract_row(
    landmarker: mp_vision.FaceLandmarker,
    bgr: np.ndarray,
    image_path: str,
    label: str,
) -> Optional[dict]:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    coords = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float64)

    need = max(MAR_CORNERS[1], max(RIGHT_EYE_IDX), max(p for pr in MAR_VERTICAL_PAIRS for p in pr))
    if coords.shape[0] < need + 1:
        return None

    ear_l = eye_aspect_ratio(coords, LEFT_EYE_IDX)
    ear_r = eye_aspect_ratio(coords, RIGHT_EYE_IDX)
    ear = float(np.nanmean([ear_l, ear_r]))
    mar = mouth_aspect_ratio(coords)
    _roll, pitch, yaw = head_pose_pitch_yaw(lm, w, h)

    return {
        "image_path": str(image_path),
        "label": label,
        "ear": ear,
        "mar": mar,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
    }

# -----------------------------------------------------------------------------
# Aykırı değer (outlier): Tukey IQR — her özellikte fence içinde olmayan satır elenir (AND).
# count_outliers_*: sütun bazlı istatistik; remove_outliers_iqr ile aynı kurallar.
# -----------------------------------------------------------------------------


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Tuple[str, ...],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Her sütun için Tukey IQR; herhangi birinde aykırıysa satır çıkar."""
    # Sunum: mask &= ... = tüm sütunlarda birden içeride kalmalı (AND); biri bile dışarıdaysa satır gider.
    mask = pd.Series(True, index=df.index)
    for col in columns:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        mask &= s.between(low, high, inclusive="both")
    return df.loc[mask].copy()


def count_outliers_per_column(
    df: pd.DataFrame, columns: Tuple[str, ...], factor: float = 1.5
) -> Tuple[dict, int]:
    """
    Her sütun için Tukey fence dışı satır sayısı (bir satır birden fazla sütunda aykırı olabilir).
    Dönüş: (sütun_adı -> sayı), en az bir sütunda aykırı satır sayısı (remove_outliers_iqr ile uyumlu).
    """
    per: dict = {}
    mask = pd.Series(True, index=df.index)
    for col in columns:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        ok = s.between(low, high, inclusive="both")
        per[col] = int((~ok).sum())
        mask &= ok
    return per, int((~mask).sum())

# -----------------------------------------------------------------------------
# Outlier görselleştirme: IQR öncesi/sonrası boxplot (rapor ve outputs/)
# save_outlier_analysis → outlier_analysis.png (özellik başına iki panel)
# save_outlier_plots → --plots-dir verilirse ek kutu grafikleri ve örnek sayısı çubukları
# -----------------------------------------------------------------------------


def save_outlier_analysis(df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: Path) -> None:
    """
    IQR'nin uygulandigi tum sayisal ozellikler: oncesi / sonrasi yan yana boxplot.
    Kayit: outlier_analysis.png
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("ear", "EAR"),
        ("mar", "MAR"),
        ("pitch_deg", "Pitch (deg)"),
        ("yaw_deg", "Yaw (deg)"),
    ]
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n))

    for row, (col, title) in enumerate(pairs):
        b = df_before[col].dropna().to_numpy()
        a = df_after[col].dropna().to_numpy()
        if b.size:
            axes[row, 0].boxplot(
                [b],
                labels=[title],
                patch_artist=True,
                boxprops=dict(facecolor="lightsteelblue", alpha=0.85),
            )
        else:
            axes[row, 0].text(0.5, 0.5, "Veri yok", ha="center", va="center", transform=axes[row, 0].transAxes)
        axes[row, 0].set_title("IQR oncesi (n=%d)" % len(b))
        axes[row, 0].set_ylabel(title)
        axes[row, 0].grid(True, axis="y", alpha=0.3)

        if a.size:
            axes[row, 1].boxplot(
                [a],
                labels=[title],
                patch_artist=True,
                boxprops=dict(facecolor="mediumseagreen", alpha=0.85),
            )
        else:
            axes[row, 1].text(0.5, 0.5, "Veri yok", ha="center", va="center", transform=axes[row, 1].transAxes)
        axes[row, 1].set_title("IQR sonrasi (n=%d)" % len(a))
        axes[row, 1].set_ylabel(title)
        axes[row, 1].grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        "Outlier analizi (Tukey IQR) — tum ozellikler, temizlik oncesi / sonrasi",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    out_path = out_dir / "outlier_analysis.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_outlier_plots(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    plots_dir: Path,
    iqr_factor: float,
) -> None:
    """IQR öncesi/sonrası kutu grafikleri ve örnek sayıları — outputs/ raporu için."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    data_b = [df_before[c].dropna() for c in feature_cols]
    data_a = [df_after[c].dropna() for c in feature_cols]
    axes[0].boxplot(data_b, labels=list(feature_cols))
    axes[0].set_title("IQR oncesi (aykiri degerler dahil)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].boxplot(data_a, labels=list(feature_cols))
    axes[1].set_title("IQR sonrasi (Tukey %.1f)" % iqr_factor)
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "outliers_boxplot_before_after.png"), dpi=150)
    plt.close(fig)

    removed = len(df_before) - len(df_after)
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["IQR oncesi", "IQR sonrasi"],
        [len(df_before), len(df_after)],
        color=["steelblue", "seagreen"],
    )
    ax.set_ylabel("Ornek sayisi")
    ax.set_title("IQR ile elenen satir: %d" % removed)
    plt.tight_layout()
    plt.savefig(str(plots_dir / "outliers_sample_counts.png"), dpi=150)
    plt.close(fig2)


# --- Program girişi: parametreler → görüntü taraması → IQR → grafik ve CSV ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drowsy/NonDrowsy görüntülerden özellik çıkarımı ve CSV üretimi."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/train",
        help="Alt klasörlerde sınıf klasörleri (ör. drowsy, nondrowsy). Düz data/ için: --data-root data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "FACE_LANDMARKER_MODEL",
            str(Path(__file__).resolve().parent / "models" / "face_landmarker.task"),
        ),
        help="face_landmarker.task yolu veya FACE_LANDMARKER_MODEL ortam değişkeni.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed_data.csv",
        help="Ham + z-score sütunlarını içeren çıktı CSV.",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR çarpanı (klasik Tukey: 1.5).",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Aykırı değer / IQR görsellerini kaydet (ör. outputs).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        metavar="N",
        help="Her N görüntüde bir ilerleme satırı yaz (0=sadece başlangıç/bitiş). Varsayılan: 25.",
    )
    args = parser.parse_args()

    # 1) Landmarker modeli ve veri kökü kontrolü
    model_path = Path(args.model)
    if not model_path.is_file():
        print(
            "Model bulunamadı: {}\n"
            "İndirin: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task\n"
            "veya --model ile tam yolu verin.".format(model_path),
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path(args.data_root)
    if not root.is_dir():
        print("Veri kökü bulunamadı: {}".format(root), file=sys.stderr)
        sys.exit(1)

    # 2) MediaPipe FaceLandmarker (tek yüz, görüntü modu)
    landmarker = build_face_landmarker(str(model_path))
    pe = max(0, args.progress_every)

    # 2b) Klasör tarama: her görüntü için extract_row → satır listesi
    def scan(r: Path) -> Tuple[List[dict], int]:
        # Sunum: Önce listelenir ki toplam N ve yüzde ilerleme anlamlı olsun (bellek ↔ UX takası).
        items = list(iter_images(r))
        total = len(items)
        if total == 0:
            return [], 0
        print(
            "Toplam %d goruntu; MediaPipe ile isleniyor..." % total,
            flush=True,
        )
        out: List[dict] = []
        sk = 0
        for idx, (img_path, label) in enumerate(items, start=1):
            bgr = imread_unicode(img_path)
            if bgr is None:
                sk += 1
            else:
                rec = extract_row(landmarker, bgr, str(img_path), label)
                if rec is None or any(
                    np.isnan(rec[k]) for k in ("ear", "mar", "pitch_deg", "yaw_deg")
                ):
                    sk += 1
                else:
                    out.append(rec)
            if pe > 0 and (
                idx == 1 or idx == total or (idx % pe == 0)
            ):
                print(
                    "  [%d/%d] %.1f%%  gecerli: %d  atlanan: %d"
                    % (idx, total, 100.0 * idx / total, len(out), sk),
                    flush=True,
                )
        print(
            "Tarama bitti: gecerli %d satir, atlanan %d goruntu."
            % (len(out), sk),
            flush=True,
        )
        return out, sk

    rows, skipped = scan(root)
    # Sunum: data/train boşsa bir üst data/ ile tekrar dene (klasör yapısı esnekliği).
    if not rows and root.name.lower() == "train" and root.parent.is_dir():
        alt = root.parent.resolve()
        if alt != root.resolve():
            print(
                "Uyari: %s icinde ornek yok; ust klasor deneniyor: %s" % (root, alt),
                file=sys.stderr,
            )
            rows, skipped = scan(alt)

    landmarker.close()

    if not rows:
        print(
            "Hic gecerli ornek uretilemedi. Kontrol: gorsel uzantisi, yuz gorunurlugu, "
            "klasor yapisı (or. data/train/drowsy/*.jpg).",
            file=sys.stderr,
        )
        sys.exit(2)

    # 3) Ham DataFrame ve IQR temizliği (çok az satır kalırsa IQR iptal)
    df = pd.DataFrame(rows)
    feature_cols = ("ear", "mar", "pitch_deg", "yaw_deg")
    df_clean = remove_outliers_iqr(df, feature_cols, factor=args.iqr_factor)
    iqr_skipped = False
    # Sunum: Çok az örnek kaldıysa IQR veriyi yok etmesin diye tamamen iptal edilir.
    if len(df_clean) < 2:
        print(
            "Uyari: IQR sonrasi yetersiz ornek; IQR atlaniyor (tum satirlar korunuyor).",
            file=sys.stderr,
        )
        df_clean = df.copy()
        iqr_skipped = True

    per_col, any_out = count_outliers_per_column(df, feature_cols, factor=args.iqr_factor)
    removed_actual = len(df) - len(df_clean)
    print("Outlier analizi (Tukey IQR, faktor=%.2f):" % args.iqr_factor)
    for c, k in per_col.items():
        print("  %s: fence disi %d satir" % (c, k))
    print(
        "  Elenen satir (uygulanan IQR): %d (%d -> %d)."
        % (removed_actual, len(df), len(df_clean))
    )
    if iqr_skipped and any_out > 0:
        print(
            "  Not: Fence disi satir sayilari bilgi amacli; bu calistirmada IQR uygulanmadi."
        )
    print("  Grafik: %s" % (Path(args.plots_dir or "outputs") / "outlier_analysis.png"))

    # 4) Outlier grafikleri (outlier_analysis.png; klasör: --plots-dir veya varsayılan outputs/)
    _out = Path(args.plots_dir or "outputs")
    os.makedirs(str(_out), exist_ok=True)
    save_outlier_analysis(df, df_clean, _out)

    if args.plots_dir:
        save_outlier_plots(
            df,
            df_clean,
            feature_cols,
            Path(args.plots_dir),
            args.iqr_factor,
        )

    # 5) Klavye sinyali görüntüden çıkmaz; CSV'de 0 (canlı uygulama 0/1 gönderir, yeniden eğitimle anlamlı olur).
    df_clean["keyboard"] = 0.0

    # 6) Z-score sütunları (*_z): CSV’de mesafe tabanlı modeller için referans; model_training eğitimde ham sütun + kendi Scaler.
    # StandardScaler: mesafe tabanlı modeller için z-score sütunları
    scaler = StandardScaler()
    Z = scaler.fit_transform(df_clean.loc[:, list(feature_cols)].values)
    z_names = ["ear_z", "mar_z", "pitch_deg_z", "yaw_deg_z"]
    for i, name in enumerate(z_names):
        df_clean[name] = Z[:, i]

    # 7) CSV kaydı (ham + keyboard + *_z)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    print(
        "Kaydedildi: {} (satır: {}, IQR sonrası; atlanan görüntü: {})".format(
            out_path, len(df_clean), skipped
        )
    )


if __name__ == "__main__":
    main()
