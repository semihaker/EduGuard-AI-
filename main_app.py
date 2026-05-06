"""
AI-Supported Student Focus & Risk Analysis — Canlı uygulama

Neden LogReg seçildi?
  Düşük gecikme ve olasılık tabanlı risk analizi için — Logistic Regression
  hafif çıkarım sunar; predict_proba ile sürekli 0–100 risk skoru üretmek
  ensemble’a göre daha uygundur (gerçek zamanlı döngü, düşük CPU).

OpenCV + MediaPipe Face Landmarker, risk_model.pkl (predict_proba),
10 karelik risk yumuşatma, 5 sn EAR kalibrasyonu, köşe grafikleri,
sol altta risk %% cizgi grafigi, proje klasorundeki sinan_alert.mp3 ile %%80 ustu ses uyari.
"""

# --- Sunum özeti ---
# Ne yapar: Webcam’den kare alır, ön işlemle aynı özellikleri çıkarır, eğitilmiş modele verir.
# Akış: Yüz → EAR/MAR/pitch/yaw → predict_proba → %% risk (yumuşatılmış) → panel + uyarı sesi.
# Girdi: risk_model.pkl (model_training çıktısı). Gerçek zamanlı için hafif LogReg tercihi.

# --- Dosya haritası (sunum / kod okuma) ---
# 1) Sabitler + Face Mesh kenar listeleri (OpenCV çizim)
# 2) Canlı özellik: extract_features (data_preprocessing ile aynı EAR/MAR/pitch/yaw)
# 3) Arayüz: yüz overlay, risk HUD, köşe sparkline’lar, risk zaman grafiği
# 4) Klavye yumuşatma + ses uyarısı (pygame / Windows yedek)
# 5) Model yardımcıları + main: kalibrasyon → döngü → pickle Pipeline (predict_proba)

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import data_preprocessing as prep
import mediapipe as mp

# Face Landmarker görüntü sarmalayıcısı (tasks API)
MPImage = mp.Image
MPImageFormat = mp.ImageFormat

# -----------------------------------------------------------------------------
# Bölüm 1 — Zaman pencereleri, panel boyutları, varsayılan uyarı sesi
# -----------------------------------------------------------------------------
# Sunum: SMOOTH_FRAMES son kare ortalaması titremeyi azaltır; CALIBRATION_SEC ile kişiye özel EAR tabanı.
CALIBRATION_SEC = 5.0
SMOOTH_FRAMES = 10
ALERT_RISK_PCT = 80.0
ALERT_COOLDOWN_SEC = 4.0
HISTORY_LEN = 120
RISK_HIST_LEN = 180
PANEL_W, PANEL_H = 220, 200
RISK_CHART_W, RISK_CHART_H = 320, 120
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ALERT_MP3 = SCRIPT_DIR / "sinan_alert.mp3"

# -----------------------------------------------------------------------------
# Bölüm 2 — Face Mesh bağlantı kümeleri (sadece görsel; özellik çıkarımı prep indeksleri kullanır)
# -----------------------------------------------------------------------------
# Sunum: MediaPipe solutions olmadan yüz ovali, göz, iris, dudak konturlarını hafif çizmek için.
# MediaPipe Face Mesh topolojisi (478 nokta; göz/iris/dudak kenar çizgileri — solutions paketi olmadan yerel kopya)
# Kaynak: mediapipe/python/solutions/face_mesh_connections.py
_FACEMESH_LIPS = frozenset(
    [
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    ]
)
_FACEMESH_LEFT_EYE = frozenset(
    [
        (263, 249),
        (249, 390),
        (390, 373),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
        (263, 466),
        (466, 388),
        (388, 387),
        (387, 386),
        (386, 385),
        (385, 384),
        (384, 398),
        (398, 362),
    ]
)
_FACEMESH_RIGHT_EYE = frozenset(
    [
        (33, 7),
        (7, 163),
        (163, 144),
        (144, 145),
        (145, 153),
        (153, 154),
        (154, 155),
        (155, 133),
        (33, 246),
        (246, 161),
        (161, 160),
        (160, 159),
        (159, 158),
        (158, 157),
        (157, 173),
        (173, 133),
    ]
)
_FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477), (477, 474)])
_FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472), (472, 469)])
_FACEMESH_FACE_OVAL = frozenset(
    [
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 323),
        (323, 361),
        (361, 288),
        (288, 397),
        (397, 365),
        (365, 379),
        (379, 378),
        (378, 400),
        (400, 377),
        (377, 152),
        (152, 148),
        (148, 176),
        (176, 149),
        (149, 150),
        (150, 136),
        (136, 172),
        (172, 58),
        (58, 132),
        (132, 93),
        (93, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10),
    ]
)

# -----------------------------------------------------------------------------
# Bölüm 3 — Klavye yazımı: risk çarpanı ve HUD yerleşimi
# -----------------------------------------------------------------------------
# Sunum: Yazarken baş eğik → yanlış yüksek risk; grace süresi içinde çarpan veya 5. özellik (keyboard=1).
KEYBOARD_TYPING_GRACE_SEC = 1.75
KEYBOARD_RISK_MULTIPLIER = 0.52

RISK_BAR_X, RISK_BAR_Y = 8, 8
RISK_BAR_W, RISK_BAR_H = 460, 32
RISK_HUD_TOTAL_H = 98  # başlık + çubuk + klavye satırı (aşağıdaki EAR satırları buna göre)


# -----------------------------------------------------------------------------
# Bölüm 4 — Canlı özellik vektörü (eğitim CSV ile uyum: prep fonksiyonları)
# -----------------------------------------------------------------------------
# Sunum: Bu fonksiyon data_preprocessing ile aynı EAR/MAR/baş açısı çıkarımını canlıda yapar.
def extract_features(
    bgr: np.ndarray,
    landmarker: prep.mp_vision.FaceLandmarker,
) -> Optional[Tuple[float, float, float, float, Any]]:
    """(ear, mar, pitch_deg, yaw_deg, landmarks) veya yüz yoksa None. landmarks = çizim için MediaPipe listesi."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=MPImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    coords = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float64)
    need = max(
        prep.MAR_CORNERS[1],
        max(prep.RIGHT_EYE_IDX),
        max(p for pr in prep.MAR_VERTICAL_PAIRS for p in pr),
    )
    if coords.shape[0] < need + 1:
        return None
    ear_l = prep.eye_aspect_ratio(coords, prep.LEFT_EYE_IDX)
    ear_r = prep.eye_aspect_ratio(coords, prep.RIGHT_EYE_IDX)
    ear = float(np.nanmean([ear_l, ear_r]))
    mar = prep.mouth_aspect_ratio(coords)
    _roll, pitch, yaw = prep.head_pose_pitch_yaw(lm, w, h)
    if any(np.isnan(v) for v in (ear, mar, pitch, yaw)):
        return None
    return ear, mar, pitch, yaw, lm


# -----------------------------------------------------------------------------
# Bölüm 5 — Video üstü çizim: yüz ovali, göz/ağız vurgusu (landmark nokta bulutu yok)
# -----------------------------------------------------------------------------
def draw_eye_mouth_overlay(
    frame: np.ndarray,
    landmarks: Any,
) -> None:
    """Yüz: çok hafif oval hat; göz ve ağız bölgelerinde daha belirgin çizgiler (nokta kalabalığı yok)."""
    n_lm = len(landmarks)

    def px(i: int) -> Tuple[int, int]:
        p = landmarks[i]
        return int(p.x * frame.shape[1]), int(p.y * frame.shape[0])

    def draw_edges_only(
        connections: frozenset,
        line_bgr: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        for a, b in connections:
            if a >= n_lm or b >= n_lm:
                continue
            cv2.line(frame, px(a), px(b), line_bgr, thickness, cv2.LINE_AA)

    # Yüz çevresi: soluk ince hat (tüm yüzü nazikçe çerçeveler)
    draw_edges_only(_FACEMESH_FACE_OVAL, (14, 38, 20), 1)

    # Göz kapakları — daha koyu / kalın
    draw_edges_only(_FACEMESH_LEFT_EYE, (42, 210, 72), 2)
    draw_edges_only(_FACEMESH_RIGHT_EYE, (42, 210, 72), 2)
    if n_lm > 477:
        draw_edges_only(_FACEMESH_LEFT_IRIS, (70, 230, 120), 1)
        draw_edges_only(_FACEMESH_RIGHT_IRIS, (70, 230, 120), 1)

    # Ağız konturu — en belirgin bölge
    draw_edges_only(_FACEMESH_LIPS, (38, 195, 62), 2)


# -----------------------------------------------------------------------------
# Bölüm 6 — Üst HUD: yüzde çubuğu, eşik çizgisi, klavye yumuşatma bildirimi
# -----------------------------------------------------------------------------
def draw_risk_hud_bar(
    frame: np.ndarray,
    risk_pct: float,
    threshold: float = ALERT_RISK_PCT,
    *,
    typing_soft: bool = False,
) -> None:
    """Üstte okunaklı yüzde çubuğu + sayı; eşik çizgisi (nötr metinler)."""
    x, y = RISK_BAR_X, RISK_BAR_Y
    w_bar, h_bar = RISK_BAR_W, RISK_BAR_H
    fh, fw = frame.shape[:2]
    x2 = min(x + w_bar, fw - 4)
    inner_w = x2 - x - 20
    if inner_w < 40:
        return
    pct = float(np.clip(risk_pct, 0.0, 100.0))

    total_h = RISK_HUD_TOTAL_H
    y2 = min(y + total_h, fh - 4)
    sub = frame[y:y2, x:x2]
    if sub.size == 0:
        return
    overlay = sub.copy()
    overlay[:] = (18, 20, 26)
    cv2.addWeighted(overlay, 0.72, sub, 0.28, 0, sub)
    cv2.rectangle(frame, (x, y), (x2 - 1, y2 - 1), (75, 78, 88), 1)

    title = "Akademik odak riski"
    cv2.putText(
        frame,
        title,
        (x + 10, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 250),
        2,
        cv2.LINE_AA,
    )

    bx0, bx1 = x + 10, x + 10 + inner_w
    by0, by1 = y + 34, y + 34 + h_bar
    cv2.rectangle(frame, (bx0, by0), (bx1 - 1, by1 - 1), (40, 42, 50), -1)
    fill_w = max(0, int(inner_w * (pct / 100.0)))
    bar_color = (55, 55, 240) if pct >= threshold else (70, 200, 95)
    if fill_w > 0:
        cv2.rectangle(frame, (bx0, by0), (bx0 + fill_w - 1, by1 - 1), bar_color, -1)
    thr_x = bx0 + int(inner_w * (threshold / 100.0))
    cv2.line(frame, (thr_x, by0), (thr_x, by1), (220, 220, 230), 2, cv2.LINE_AA)

    pct_str = "%.0f %%" % pct
    (tw, th), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = bx0 + (inner_w - tw) // 2
    ty = by0 + (h_bar + th) // 2 - 2
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(
            frame,
            pct_str,
            (tx + dx, ty + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        pct_str,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Esik %.0f%%" % threshold,
        (bx1 - 118, by0 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 185, 195),
        1,
        cv2.LINE_AA,
    )
    if typing_soft:
        cv2.putText(
            frame,
            "Klavye: risk tahmini dusuruldu",
            (x + 10, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 200, 255),
            1,
            cv2.LINE_AA,
        )


# -----------------------------------------------------------------------------
# Bölüm 7 — Uyarı sesi: risk düşünce durdur; klavyede zorla kes
# -----------------------------------------------------------------------------
def stop_alert_music_if_safe(mixer_ready: bool, risk_pct: float) -> None:
    """Risk eşiğin altına inince pygame müziği durdur (odak geri gelince şarkı susar)."""
    if not mixer_ready or risk_pct >= ALERT_RISK_PCT:
        return
    try:
        import pygame

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


def stop_alert_music_force(mixer_ready: bool) -> None:
    """Klavye yazımı gibi durumlarda uyarı sesini kes."""
    if not mixer_ready:
        return
    try:
        import pygame

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


# Sunum: pynput ile son tuş zamanı; yoksa dummy (keyboard sinyali 0).
def start_keyboard_tracker() -> Tuple[Optional[Any], Any]:
    """
    Sistem genelinde klavye (pynput). Yoksa (None, lambda: 0.0).
    Yazarken baş eğik olduğu için risk şişmesini azaltmak için kullanılır.
    """
    try:
        from pynput import keyboard as pkb
    except ImportError:
        return None, lambda: 0.0

    state: dict = {"t": 0.0}

    def on_press(_k: Any) -> None:
        state["t"] = time.perf_counter()

    listener = pkb.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener, lambda: float(state["t"])


# -----------------------------------------------------------------------------
# Bölüm 8 — Köşe panelleri ve risk zaman serisi (sparkline / mini grafik)
# -----------------------------------------------------------------------------
def _scale_series(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(float(v) - lo) / (hi - lo) for v in values]


def draw_sparkline(
    roi: np.ndarray,
    values: Sequence[float],
    color: Tuple[int, int, int],
    ref_norm: Optional[float] = None,
) -> None:
    """roi üzerine (w x h) yatay zaman serisi çiz."""
    h, w = roi.shape[:2]
    if len(values) < 2:
        return
    scaled = _scale_series(values)
    pts: List[Tuple[int, int]] = []
    for i, sn in enumerate(scaled):
        x = int(i * (w - 1) / max(len(scaled) - 1, 1))
        y = int(h - 4 - sn * (h - 10))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(roi, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)
    if ref_norm is not None:
        rn = float(np.clip(ref_norm, 0.0, 1.0))
        yref = int(h - 4 - rn * (h - 10))
        cv2.line(roi, (0, yref), (w - 1, yref), (180, 180, 180), 1, cv2.LINE_AA)


def draw_corner_panel(
    frame: np.ndarray,
    ear_hist: Sequence[float],
    mar_hist: Sequence[float],
    pitch_hist: Sequence[float],
    yaw_hist: Sequence[float],
    ear_baseline: Optional[float],
) -> None:
    """Sağ üst köşede EAR, MAR, baş pozu mini grafikleri."""
    fh, fw = frame.shape[:2]
    x0 = max(0, fw - PANEL_W - 8)
    y0 = 8
    x1 = min(fw, x0 + PANEL_W)
    y1 = min(fh, y0 + PANEL_H)
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return
    overlay = sub.copy()
    overlay[:] = (30, 30, 30)
    cv2.addWeighted(overlay, 0.45, sub, 0.55, 0, sub)

    row_h = (y1 - y0) // 3
    # EAR
    row = sub[0:row_h, :]
    ref_n = None
    if ear_baseline is not None and ear_hist:
        lo = min(min(ear_hist), ear_baseline * 0.5)
        hi = max(max(ear_hist), ear_baseline * 1.1)
        if hi > lo:
            ref_n = (ear_baseline - lo) / (hi - lo)
    draw_sparkline(row, list(ear_hist)[-HISTORY_LEN:], (100, 200, 255), ref_n)
    cv2.putText(
        sub,
        "EAR",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # MAR
    row = sub[row_h : 2 * row_h, :]
    draw_sparkline(row, list(mar_hist)[-HISTORY_LEN:], (120, 255, 160), None)
    cv2.putText(
        sub,
        "MAR",
        (4, row_h + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # Pitch / Yaw (aynı eksende normalize iki seri)
    row = sub[2 * row_h :, :]
    ph = list(pitch_hist)[-HISTORY_LEN:]
    yh = list(yaw_hist)[-HISTORY_LEN:]
    if len(ph) >= 2 and len(yh) >= 2:
        lo = min(min(ph), min(yh))
        hi = max(max(ph), max(yh))
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        sp = [(float(v) - lo) / (hi - lo) for v in ph]
        sy = [(float(v) - lo) / (hi - lo) for v in yh]
        h2, w2 = row.shape[:2]
        for _, series, col in (("P", sp, (255, 180, 100)), ("Y", sy, (255, 100, 180))):
            pts = []
            for i, sn in enumerate(series):
                x = int(i * (w2 - 1) / max(len(series) - 1, 1))
                y = int(h2 - 4 - sn * (h2 - 10))
                pts.append((x, y))
            for i in range(len(pts) - 1):
                cv2.line(row, pts[i], pts[i + 1], col, 1, cv2.LINE_AA)
    cv2.putText(
        sub,
        "Pitch/Yaw",
        (4, 2 * row_h + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_risk_percentage_line_chart(
    frame: np.ndarray,
    risk_pct_history: Sequence[float],
    threshold: float = ALERT_RISK_PCT,
) -> None:
    """Sol altta 0-100%% risk yuzdesi cizgi grafigi ve uyari esik cizgisi."""
    if len(risk_pct_history) < 2:
        return
    fh, fw = frame.shape[:2]
    pw = min(RISK_CHART_W, fw - 16)
    ph = RISK_CHART_H
    x0 = 8
    y0 = max(8, fh - ph - 10)
    x1 = x0 + pw
    y1 = y0 + ph
    if y1 > fh or x1 > fw:
        return

    roi = frame[y0:y1, x0:x1]
    bg = roi.copy()
    bg[:] = (25, 28, 32)
    cv2.addWeighted(bg, 0.55, roi, 0.45, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (80, 80, 90), 1)

    margin_l, margin_r, margin_top, margin_b = 36, 10, 22, 10
    px0 = x0 + margin_l
    py0 = y0 + margin_top
    plot_w = pw - margin_l - margin_r
    plot_h = ph - margin_top - margin_b
    px1 = px0 + plot_w
    py1 = py0 + plot_h

    cv2.putText(
        frame,
        "Risk %% (zaman)",
        (x0 + 4, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "100", (x0 + 2, py0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "50",
        (x0 + 6, py0 + plot_h // 2 + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "0", (x0 + 10, py1), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    y_thr = int(py0 + plot_h - (threshold / 100.0) * plot_h)
    cv2.line(frame, (px0, y_thr), (px1, y_thr), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Esik %.0f%%" % threshold,
        (min(px1 - 2, fw - 120), max(y_thr - 4, py0 + 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (200, 200, 220),
        1,
        cv2.LINE_AA,
    )

    vals = [float(np.clip(v, 0.0, 100.0)) for v in risk_pct_history][-RISK_HIST_LEN:]
    n = len(vals)
    col_line = (0, 200, 255) if vals[-1] < threshold else (0, 80, 255)
    for i in range(n - 1):
        xa = int(px0 + i * plot_w / max(n - 1, 1))
        ya = int(py0 + plot_h - (vals[i] / 100.0) * plot_h)
        xb = int(px0 + (i + 1) * plot_w / max(n - 1, 1))
        yb = int(py0 + plot_h - (vals[i + 1] / 100.0) * plot_h)
        cv2.line(frame, (xa, ya), (xb, yb), col_line, 2, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Bölüm 9 — Kayıtlı sklearn Pipeline ile uyumluluk (4 vs 5 özellik, pozitif sınıf sütunu)
# -----------------------------------------------------------------------------
def risk_model_n_features(model: Any) -> int:
    """Kayıtlı Pipeline'ın beklediği ham özellik sayısı (4 veya 5)."""
    n = getattr(model, "n_features_in_", None)
    if n is not None:
        return int(n)
    try:
        return int(model.named_steps["scaler"].n_features_in_)  # type: ignore[index]
    except Exception:
        return 4


def positive_risk_column(model) -> int:
    """Pipeline veya tek sınıflandırıcıda risk (pozitif) sütun indeksi."""
    # Sunum: classes_ sırası veri/etiketlere göre değişebilir; risk olasılığı doğru sütundan alınır.
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    classes = getattr(est, "classes_", None)
    if classes is None:
        return 1
    classes = np.asarray(classes)
    hits = np.where(classes == 1)[0]
    if len(hits):
        return int(hits[0])
    return int(np.argmax(classes))


# Sunum: Webcam ters/ayna; landmark ve model girdisi düzeltilmiş kare üzerinde.
def apply_camera_orientation(frame: np.ndarray, mode: str) -> np.ndarray:
    """Kamera yönü: none=değiştirme; rotate_180=tam ters; flip_h/flip_v yatay veya dikey ayna."""
    if mode == "none":
        return frame
    if mode == "rotate_180":
        # cv2.flip(-1) = 180°; bazi Windows kameralarda ROTATE_180 ile ayni ama daha tutarli
        return cv2.flip(frame, -1)
    if mode == "flip_h":
        return cv2.flip(frame, 1)
    if mode == "flip_v":
        return cv2.flip(frame, 0)
    return frame


# Sunum: --alert-sound yolu; proje kökü ve varsayılan MP3 ile geri dönüş.
def resolve_alert_sound_path(alert_arg: str) -> Path:
    """--alert-sound dosyasi, yoksa SCRIPT_DIR altinda ayni ad, yoksa sinan_alert.mp3."""
    p = Path(alert_arg)
    if p.is_file():
        return p.resolve()
    cand = SCRIPT_DIR / Path(alert_arg).name
    if cand.is_file():
        return cand
    if DEFAULT_ALERT_MP3.is_file():
        return DEFAULT_ALERT_MP3
    return cand


# Sunum: pygame.mixer; başarısızsa try_play_alert Windows’ta startfile ile yedekler.
def init_alert_mixer(sound_path: Path) -> bool:
    if not sound_path.is_file():
        return False
    try:
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        return True
    except Exception:
        return False


def try_play_alert(
    sound_path: Path,
    last_play: float,
    now: float,
    mixer_ready: bool,
) -> float:
    """Risk uyarısı (sinan_alert.mp3); cooldown; pygame veya Windows yedek."""
    if now - last_play < ALERT_COOLDOWN_SEC:
        return last_play
    if not sound_path.is_file():
        return last_play
    if mixer_ready:
        try:
            import pygame

            pygame.mixer.music.load(str(sound_path))
            pygame.mixer.music.play()
            return now
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            os.startfile(str(sound_path))  # type: ignore[attr-defined]
        except OSError:
            pass
    return now


# =============================================================================
# Bölüm 10 — Program girişi: argparse → risk_model.pkl → kamera döngüsü
# =============================================================================
def main() -> None:
    # --- 10a) Argümanlar: kamera, landmarker .task, risk_model.pkl, ses, kamera yönü, klavye yumuşatma ---
    parser = argparse.ArgumentParser(description="Canlı akademik odak riski (LogReg + MediaPipe).")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--landmarker-model",
        type=str,
        default=os.environ.get(
            "FACE_LANDMARKER_MODEL",
            str(Path(__file__).resolve().parent / "models" / "face_landmarker.task"),
        ),
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default="risk_model.pkl",
        help="pickle içinde Pipeline (StandardScaler + LogisticRegression).",
    )
    parser.add_argument(
        "--alert-sound",
        type=str,
        default="sinan_alert.mp3",
    )
    parser.add_argument(
        "--camera-fix",
        type=str,
        default="none",
        choices=["none", "rotate_180", "flip_h", "flip_v"],
        help="Kamera yönü: none=ham (çoğu webcam). Ters görüntü: rotate_180 veya flip_v. Ayna: flip_h.",
    )
    parser.add_argument(
        "--no-keyboard-soften",
        action="store_true",
        help="Klavye yazarken risk carpimini kapat (varsayilan: acik; pynput kuruluysa).",
    )
    args = parser.parse_args()
    print("Kamera yonelimi (--camera-fix): %s" % args.camera_fix, flush=True)

    # --- 10b) Zorunlu dosyalar: face_landmarker.task + risk_model.pkl (pickle Pipeline) ---
    lm_path = Path(args.landmarker_model)
    if not lm_path.is_file():
        print("Face Landmarker modeli yok: %s" % lm_path, file=sys.stderr)
        sys.exit(1)
    risk_path = Path(args.risk_model)
    if not risk_path.is_file():
        print("risk_model.pkl bulunamadı: %s" % risk_path, file=sys.stderr)
        sys.exit(1)

    # --- 10c) Model: predict_proba için doğru sütun; özellik boyutu (keyboard ile 5) ---
    with open(risk_path, "rb") as f:
        risk_model = pickle.load(f)
    # Sunum: İkili sınıfta risk sütunu 0 veya 1 olabilir — pozitif sınıfın indeksini bul.
    risk_proba_col = positive_risk_column(risk_model)
    risk_n_features = risk_model_n_features(risk_model)
    print(
        "risk_model.pkl ozellik sayisi: %d (4=eski; 5=keyboard dahil — yeniden egitim onerilir)"
        % risk_n_features,
        flush=True,
    )

    # --- 10d) Uyarı sesi: pygame veya (yoksa) Windows varsayılan oynatıcı ---
    sound_path = resolve_alert_sound_path(args.alert_sound)
    mixer_ok = init_alert_mixer(sound_path)
    if sound_path.is_file():
        print("Uyari sesi:", sound_path, flush=True)
    else:
        print(
            "Uyari: ses dosyasi bulunamadi (beklenen: %s)" % DEFAULT_ALERT_MP3,
            file=sys.stderr,
        )
    if sound_path.is_file() and not mixer_ok:
        print(
            "Not: pygame.mixer acilamadi; uyari icin varsayilan oynatici kullanilabilir.",
            file=sys.stderr,
        )

    # --- 10e) MediaPipe landmarker + webcam aç; isteğe bağlı pynput klavye dinleyicisi ---
    landmarker = prep.build_face_landmarker(str(lm_path))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Kamera açılamadı.", file=sys.stderr)
        landmarker.close()
        sys.exit(1)

    kb_listener: Optional[Any] = None
    get_last_key_t: Any = lambda: 0.0
    if not args.no_keyboard_soften:
        kb_listener, get_last_key_t = start_keyboard_tracker()
        if kb_listener is None:
            print(
                "Ipucu: klavyede yazarken yanlis yuksek risk icin: pip install pynput",
                flush=True,
            )

    # --- 10f) Durum: EAR kalibrasyonu, özellik/geçmiş kuyrukları, yumuşatılmış risk ---
    calib_ears: List[float] = []
    t_start = time.perf_counter()
    calibrating = True
    ear_baseline: Optional[float] = None
    # Sunum: İlk birkaç saniye “normal bakış” EAR medyanı — kişiye göre referans çizgisi.

    risk_window: Deque[float] = deque(maxlen=SMOOTH_FRAMES)
    risk_pct_hist: Deque[float] = deque(maxlen=RISK_HIST_LEN)
    ear_hist: Deque[float] = deque(maxlen=HISTORY_LEN)
    mar_hist: Deque[float] = deque(maxlen=HISTORY_LEN)
    pitch_hist: Deque[float] = deque(maxlen=HISTORY_LEN)
    yaw_hist: Deque[float] = deque(maxlen=HISTORY_LEN)

    last_alert_t = -1e9
    risk_smooth = 0.0

    try:
        # --- 10g) Ana döngü: kare → extract_features → kalibrasyon veya predict_proba → HUD / ses / grafikler ---
        # Sunum: Pipeline ile aynı özellik sırası; EAR tabanına göre ek risk düzeltmesi; SMOOTH_FRAMES ortalaması.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_camera_orientation(frame, args.camera_fix)
            now = time.perf_counter()
            feats = extract_features(frame, landmarker)

            if feats is not None:
                ear, mar, pitch, yaw, lm_overlay = feats
                draw_eye_mouth_overlay(frame, lm_overlay)

            if calibrating:
                elapsed = now - t_start
                if feats is not None:
                    ear_hist.append(ear)
                    mar_hist.append(mar)
                    pitch_hist.append(pitch)
                    yaw_hist.append(yaw)
                    if elapsed < CALIBRATION_SEC:
                        calib_ears.append(ear)
                if elapsed >= CALIBRATION_SEC:
                    if calib_ears:
                        ear_baseline = float(np.median(calib_ears))
                    calibrating = False
                msg = "Kalibrasyon: normal bakis (%.1f sn)" % max(
                    0.0, CALIBRATION_SEC - min(elapsed, CALIBRATION_SEC)
                )
                cv2.putText(
                    frame,
                    msg,
                    (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                if feats is not None:
                    ear_hist.append(ear)
                    mar_hist.append(mar)
                    pitch_hist.append(pitch)
                    yaw_hist.append(yaw)

                    last_kt = float(get_last_key_t())
                    typing_soft = (
                        not args.no_keyboard_soften
                        and kb_listener is not None
                        and last_kt > 0.0
                        and (now - last_kt) < KEYBOARD_TYPING_GRACE_SEC
                    )
                    k_feat = 1.0 if typing_soft else 0.0

                    if risk_n_features >= 5:
                        X = np.array(
                            [[ear, mar, pitch, yaw, k_feat]], dtype=np.float64
                        )
                    else:
                        X = np.array([[ear, mar, pitch, yaw]], dtype=np.float64)
                    # Sunum: Pipeline — özellik sayısı CSV / egitim ile aynı olmali (keyboard 5. boyut).
                    proba = risk_model.predict_proba(X)[0]
                    risk_raw = float(proba[risk_proba_col] * 100.0)

                    # Kişiselleştirme: kalibre EAR'a göre göreceli düşüşte risk artışı (eşik mantığı)
                    if ear_baseline is not None and ear_baseline > 1e-6:
                        rel = ear / ear_baseline
                        if rel < 0.88:
                            risk_raw = min(100.0, risk_raw + (0.88 - rel) * 35.0)

                    # Eski 4 ozellikli model: klavye carpani (5 ozellikte model klavyeyi zaten gorur)
                    if risk_n_features < 5 and typing_soft:
                        risk_raw = risk_raw * KEYBOARD_RISK_MULTIPLIER

                    # Sunum: Son SMOOTH_FRAMES karenin ortalaması — tek kare gürültüsünü azaltır.
                    risk_window.append(risk_raw)
                    risk_smooth = float(np.mean(risk_window))
                    risk_pct_hist.append(risk_smooth)

                    if typing_soft:
                        stop_alert_music_force(mixer_ok)
                    else:
                        stop_alert_music_if_safe(mixer_ok, risk_smooth)

                    if risk_smooth >= ALERT_RISK_PCT and not typing_soft:
                        last_alert_t = try_play_alert(
                            sound_path, last_alert_t, now, mixer_ok
                        )

                    draw_risk_hud_bar(
                        frame, risk_smooth, ALERT_RISK_PCT, typing_soft=typing_soft
                    )
                    # EAR/MAR/Pitch/Yaw sayisal tekrar yok — sag ust panelde grafik var; klavye: HUD satiri (typing_soft).
                else:
                    cv2.putText(
                        frame,
                        "Yuz tespit edilemedi",
                        (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

            draw_corner_panel(
                frame,
                ear_hist,
                mar_hist,
                pitch_hist,
                yaw_hist,
                ear_baseline if not calibrating else None,
            )
            if not calibrating and len(risk_pct_hist) >= 2:
                draw_risk_percentage_line_chart(frame, list(risk_pct_hist))

            cv2.putText(
                frame,
                "q: cikis",
                (12, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Akademik Odak Riski", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        # Sunum: Kaynak temizliği — klavye listener, kamera, OpenCV pencereleri, landmarker.
        if kb_listener is not None:
            try:
                kb_listener.stop()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    # Doğrudan çalıştırma: python main_app.py [--camera ...] [--risk-model ...]
    main()

