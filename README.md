# 🧠 BrainTumorAI
## Klasifikasi Tumor Otak Menggunakan Deep Learning Berbasis Citra MRI

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![License](https://img.shields.io/badge/License-Educational-green)

---

## Deskripsi Proyek

BrainTumorAI merupakan aplikasi berbasis **Deep Learning** yang dikembangkan sebagai tugas akhir mata kuliah **Kecerdasan Buatan**.

Sistem ini mampu melakukan klasifikasi citra **MRI Otak** ke dalam empat kategori yaitu:

- 🧠 Glioma
- 🧠 Meningioma
- 🧠 Pituitary
- ✅ Tidak Ada Tumor

Model utama yang digunakan adalah **Custom Convolutional Neural Network (CNN)** dan dibandingkan dengan **EfficientNetB0** sebagai model pembanding.

Website dikembangkan menggunakan **Flask Framework** sehingga pengguna dapat melakukan prediksi secara langsung melalui browser.

---

# 👨‍💻 Anggota Kelompok

| Nama | NIM |
|------|-----|
| Rezha Achmad Muharam | 2406081 |
| Faujan Alamsyah | 2406121 |

Institut Teknologi Garut

Program Studi Teknik Informatika

---

# 📂 Struktur Repository

```text
UAS-KecerdasanBuatan/
│
├── README.md
├── Laporan_UAS.md
├── uas_model.ipynb
├── requirements.txt
├── app.py
├── brain_tumor_model.keras
├── class_names.json
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── uploads/
│
└── data/
    ├── dataset/
    └── Jurnal/
```

---

# 🛠 Framework

- Python 3.12
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-Learn

---

# 📊 Dataset

Dataset yang digunakan merupakan **Brain MRI Images Dataset** yang terdiri dari **4 kelas**, yaitu:

- Glioma
- Meningioma
- Pituitary
- No Tumor

Dataset digunakan untuk proses training, validasi, dan pengujian model Deep Learning.

---

# 🧠 Model Deep Learning

Model yang digunakan:

- Custom CNN ✅
- EfficientNetB0

Model terbaik kemudian diimplementasikan ke dalam aplikasi web menggunakan Flask.

---

# 📈 Hasil Evaluasi

| Model | Accuracy | Precision | Recall | F1-Score | Keterangan |
|--------|---------:|----------:|--------:|---------:|------------|
| Custom CNN | 76.06% | 77.39% | 76.06% | 74.38% | Model utama yang diimplementasikan pada aplikasi Flask |
| EfficientNetB0 | 25.00% | 6.25% | 25.00% | 10.00% | Model pembanding (belum optimal) |

---

# 🚀 Cara Menjalankan Project

## 1. Clone Repository

```bash
git clone https://github.com/USERNAME/UAS-KecerdasanBuatan.git
```

Masuk ke folder project

```bash
cd UAS-KecerdasanBuatan
```

---

## 2. Buat Virtual Environment (Opsional)

Windows

```bash
py -m venv .venv
```

Aktifkan Virtual Environment

Command Prompt

```bash
.venv\Scripts\activate
```

PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Seluruh Library

```bash
py -m pip install -r requirements.txt
```

Tunggu hingga proses instalasi selesai.

---

## 4. Pastikan File Model Tersedia

Pastikan file berikut berada pada folder utama project.

```
brain_tumor_model.keras

class_names.json
```

---

## 5. Menjalankan Flask

Jalankan perintah berikut

```bash
py app.py
```

Apabila berhasil maka akan muncul

```text
* Running on http://127.0.0.1:5000
```

---

## 6. Membuka Website

Buka browser kemudian akses

```
http://127.0.0.1:5000
```

Website BrainTumorAI akan tampil.

---

## 7. Melakukan Prediksi

1. Masuk ke menu Prediksi
2. Drag & Drop gambar MRI
3. Klik **Prediksi Sekarang**
4. Tunggu proses inferensi selesai
5. Sistem akan menampilkan:

- Hasil Prediksi
- Confidence Score
- Probability setiap kelas
- Deskripsi penyakit

---

# 📚 Referensi

1. TensorFlow Documentation
2. Flask Documentation
3. Brain MRI Images Dataset
4. Scikit-Learn Documentation

---

# 📄 Lisensi

Repository ini dibuat untuk keperluan akademik pada Mata Kuliah Kecerdasan Buatan.

© 2026 Rezha Achmad Muharam & Faujan Alamsyah
