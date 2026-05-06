#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tek komut: Face Landmarker indir → ön işlem (data/train) → model eğitimi.
Çıktılar: outputs/ (IQR/outlier grafikleri, model karşılaştırması, confusion matrix PNG).
"""

# --- Sunum özeti ---
# Ne yapar: Pipeline’ı tek komutta çalıştırır (model yoksa indirir).
# Sıra: models/face_landmarker.task → data_preprocessing.py → model_training.py.
# Ne zaman: İlk kurulum veya “her şeyi baştan üret” demek için.

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

# Gercek .task dosyasi genelde ~2–8 MB; cok kucukse HTML hata sayfasi veya bos indirme.
MIN_MODEL_BYTES = 300_000

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# --- Program girişi: tek komutla indir → ön işlem → eğitim ---
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    task_path = models_dir / "face_landmarker.task"
    outputs_dir = script_dir / "outputs"
    csv_path = script_dir / "processed_data.csv"
    data_train = script_dir / "data" / "train"
    py = sys.executable

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    def model_ok(p: Path) -> bool:
        return p.is_file() and p.stat().st_size >= MIN_MODEL_BYTES

    root_task = script_dir / "face_landmarker.task"
    if not model_ok(task_path):
        if model_ok(root_task):
            shutil.copy2(root_task, task_path)
            print("Model proje kokunden models/ altina kopyalandi:", task_path, flush=True)
        else:
            if task_path.is_file():
                print("Gecersiz/kucuk model siliniyor, yeniden indiriliyor...", flush=True)
                task_path.unlink(missing_ok=True)
            print("Indiriliyor:", LANDMARKER_URL, flush=True)
            try:
                urlretrieve(LANDMARKER_URL, task_path)
            except URLError as e:
                raise SystemExit("Model indirilemedi: %s\nBaglantiyi kontrol edin." % e) from e
            print("Kaydedildi:", task_path, flush=True)
    if not model_ok(task_path):
        raise SystemExit(
            "Model dosyasi gecersiz veya cok kucuk: %s\n"
            "Manuel indirip models/face_landmarker.task olarak kaydedin."
            % task_path
        )
    print("Model hazir:", task_path, "(%d bayt)" % task_path.stat().st_size, flush=True)

    if not data_train.is_dir():
        raise SystemExit(
            "Veri klasoru yok: %s\n"
            "Beklenen: data/train/<sinif_klasorleri>/goruntuler" % data_train
        )

    prep = script_dir / "data_preprocessing.py"
    train = script_dir / "model_training.py"
    if not prep.is_file() or not train.is_file():
        raise SystemExit("data_preprocessing.py veya model_training.py bulunamadi.")

    # Sunum: [1/2] CSV + outlier PNG; --plots-dir ile ek kutu grafikleri de üretilir.
    cmd_prep = [
        py,
        str(prep),
        "--data-root",
        str(data_train),
        "--output",
        str(csv_path),
        "--model",
        str(task_path),
        "--plots-dir",
        str(outputs_dir),
    ]
    print("\n[1/2] On islem:", " ".join(cmd_prep), flush=True)
    subprocess.run(cmd_prep, check=True, cwd=str(script_dir))

    # Sunum: [2/2] CSV zaten var; --no-auto-preprocess ile tekrar ön işlem çağrılmaz.
    cmd_train = [
        py,
        str(train),
        "--data",
        str(csv_path),
        "--outputs",
        str(outputs_dir),
        "--no-auto-preprocess",
    ]
    print("\n[2/2] Model egitimi:", " ".join(cmd_train), flush=True)
    subprocess.run(cmd_train, check=True, cwd=str(script_dir))

    print("\nGrafikler:", outputs_dir.resolve())
    print("  - outlier_analysis.png (EAR/MAR boxplot oncesi/sonrasi)")
    print("  - outliers_boxplot_before_after.png, outliers_sample_counts.png (opsiyonel)")
    print("  - model_comparison.png (Accuracy/F1 + ozellik onemi, tek PNG)")
    print("  - model_metrics_comparison.png, confusion_matrix.png, *_cm.png")
    print("\nSİSTEM HAZIR, HOCAYA SUNABİLİRSİN")


if __name__ == "__main__":
    main()
