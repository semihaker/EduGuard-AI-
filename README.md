# AI-Supported Student Focus & Risk Analysis

Webcam görüntüsünden yüz landmark'ları çıkararak öğrencinin odak/uyuklama riskini gerçek zamanlı tahmin eden bir proje.

Proje; veri ön işleme, model eğitimi ve canlı uygulama adımlarını içerir.

## Ne Yapar?
Aşağıdaki görselde sistemin canlı çalışma anını görebilirsiniz. Sistem; EAR, MAR ve Pose verilerini anlık olarak grafiğe dökerken, sağlanan veriler üzerinden akademik risk skorunu hesaplamaktadır.

![Akademik Odak Riski Analizi](akademik%20risk%20uyari%20gorsel.png)

- Görüntülerden `EAR`, `MAR`, `pitch`, `yaw` özelliklerini çıkarır.
- IQR ile aykırı değer temizliği uygular.
- `Logistic Regression`, `Random Forest`, `KNN`, `SVM` ve `Soft Voting Ensemble` modellerini karşılaştırır.
- Canlı kamerada `predict_proba` tabanlı `0-100` risk yüzdesi gösterir.
- Risk belirli eşik üstüne çıkınca sesli uyarı verir.

## Proje Yapısı

| Dosya | Açıklama |
| --- | --- |
| `data_preprocessing.py` | Görüntülerden özellik çıkarır, IQR temizliği yapar, `processed_data.csv` üretir. |
| `model_training.py` | Modelleri eğitir, metrik ve görselleri üretir, `best_model.pkl` ve `risk_model.pkl` kaydeder. |
| `main_app.py` | Canlı kamera uygulaması; anlık risk hesaplar, grafik/HUD gösterir, sesli uyarı verir. |
| `fix_and_run.py` | Tek komutla model dosyasını indirir, ön işleme + eğitim zincirini çalıştırır. |

## Gereksinimler

Python 3.10+ önerilir.

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn matplotlib seaborn pygame
```

Face Landmarker dosyası:

- `models/face_landmarker.task` konumunda olmalı
- ya da `FACE_LANDMARKER_MODEL` ortam değişkeni ile yolu verilmelidir
- indirme linki: [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

## Veri Klasörü Düzeni

Örnek yapı:

```text
data/
  train/
    drowsy/
      img1.jpg
      ...
    nondrowsy/
      img2.jpg
      ...
```

Sınıf etiketi klasör adından alınır (`drowsy`, `nondrowsy` gibi).

## Hızlı Başlangıç

### 1) Tek komutla tüm pipeline

```bash
python fix_and_run.py
```

Bu komut sırasıyla:
1. `face_landmarker.task` dosyasını kontrol eder/indirir
2. `data/train` üzerinden ön işleme yapar
3. model eğitimini çalıştırır

### 2) Adım adım çalıştırmak istersen

Ön işleme:

```bash
python data_preprocessing.py --data-root data/train --output processed_data.csv --model models/face_landmarker.task --plots-dir outputs
```

Model eğitimi:

```bash
python model_training.py --data processed_data.csv --outputs outputs
```

Canlı uygulama:

```bash
python main_app.py --camera 0 --landmarker-model models/face_landmarker.task --risk-model risk_model.pkl --camera-fix none
```

## Canlı Uygulama Notları

- Çıkış tuşu: `q`
- Risk skoru: `predict_proba` ile `0-100`
- Görüntüde:
  - sağ üstte EAR/MAR/Pose mini grafikler
  - sol altta risk çizgi grafiği
- Uyarı eşiği: `%80` (yumuşatılmış risk)
- Ses dosyası: varsayılan `sinan_alert.mp3` (isteğe bağlı `--alert-sound`)
- Kamera yön seçenekleri: `none`, `rotate_180`, `flip_h`, `flip_v`

## Üretilen Çıktılar

Eğitim sonrası proje kökünde:

- `processed_data.csv`
- `risk_model.pkl`
- `best_model.pkl`

`outputs/` klasöründe örnek görseller:

- `outlier_analysis.png`
- `model_comparison.png`
- `model_metrics_comparison.png`
- `confusion_matrix.png`
- `*_cm.png` (model bazlı confusion matrix)

## Neden İki Farklı Model Dosyası Var?

- `risk_model.pkl`: canlı uygulamada düşük gecikmeli risk tahmini için (Logistic Regression pipeline)
- `best_model.pkl`: test F1'e göre en iyi modelin kaydı (taban model veya ensemble olabilir)

## Sık Karşılaşılan Sorunlar

- `face_landmarker.task bulunamadı`
  - dosyayı `models/` altına koyun veya `--landmarker-model` verin
- Kamera ters görünüyorsa
  - `--camera-fix rotate_180` veya `flip_h/flip_v` deneyin
- `risk_model.pkl bulunamadı`
  - önce `model_training.py` veya `fix_and_run.py` çalıştırın

## Lisans / Kullanım

Bu proje ders çalışması amacıyla geliştirilmiştir. Akademik kullanımda MediaPipe ve ilgili literatüre atıf yapılması önerilir.
