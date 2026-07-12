# KLASIFIKASI BRAIN TUMOR MRI MENGGUNAKAN DEEP LEARNING
## Perbandingan Custom Convolutional Neural Network (CNN) dan EfficientNetB0

---

## Laporan UAS Kecerdasan Buatan

**Mata Kuliah** : Kecerdasan Buatan

**Program Studi** : Teknik Informatika

**Universitas** : Institut Teknologi Garut

---

# Daftar Isi

1. Judul Proyek
2. Business Understanding
3. Data Understanding
4. Exploratory Data Analysis (EDA)
5. Data Preparation
6. Modeling
7. Evaluation
8. Kesimpulan dan Rekomendasi
9. Referensi
10. Lampiran

---

# 1. Judul Proyek

## 1.1 Judul

**Klasifikasi Brain Tumor MRI Menggunakan Deep Learning dengan Perbandingan Custom Convolutional Neural Network (CNN) dan EfficientNetB0**

---

## 1.2 Nama Mahasiswa

| Nama | NIM |
|------|-----|
| Rezha Achmad Muharam | 2406081 |
| Faujan Alamsyah | 2406121 |


---

## 1.3 Domain Proyek (Latar Belakang)

Brain Tumor merupakan salah satu penyakit yang ditandai dengan pertumbuhan sel abnormal pada jaringan otak. Penyakit ini dapat menyebabkan berbagai gangguan neurologis, seperti penurunan fungsi kognitif, gangguan keseimbangan tubuh, kehilangan kemampuan motorik, hingga kematian apabila tidak segera ditangani. Oleh karena itu, proses diagnosis yang cepat dan akurat menjadi faktor penting dalam menentukan tindakan medis yang tepat.

Saat ini proses identifikasi Brain Tumor umumnya dilakukan menggunakan citra **Magnetic Resonance Imaging (MRI)**. MRI mampu menghasilkan citra jaringan lunak otak dengan kualitas yang sangat baik sehingga menjadi standar dalam proses diagnosis tumor otak. Akan tetapi proses interpretasi citra MRI masih dilakukan secara manual oleh dokter spesialis radiologi sehingga membutuhkan waktu yang cukup lama serta dipengaruhi oleh pengalaman masing-masing tenaga medis.

Perkembangan teknologi **Artificial Intelligence (AI)** khususnya **Deep Learning** telah memberikan kontribusi besar dalam bidang Computer Vision, termasuk pada analisis citra medis. Salah satu algoritma yang paling banyak digunakan adalah **Convolutional Neural Network (CNN)** yang mampu mempelajari karakteristik citra secara otomatis tanpa memerlukan ekstraksi fitur secara manual.

Selain CNN konvensional, metode **Transfer Learning** juga banyak digunakan untuk meningkatkan performa klasifikasi citra medis. Salah satu arsitektur yang memiliki performa sangat baik adalah **EfficientNetB0**, yaitu model Deep Learning yang mampu menghasilkan akurasi tinggi dengan jumlah parameter yang relatif kecil dibandingkan arsitektur CNN lainnya.

Pada penelitian ini dilakukan implementasi dua model Deep Learning, yaitu **Custom CNN** dan **EfficientNetB0**, untuk melakukan klasifikasi citra MRI Brain Tumor ke dalam empat kelas yaitu **Glioma**, **Meningioma**, **Pituitary**, dan **No Tumor**. Kedua model akan dibandingkan berdasarkan nilai Accuracy, Precision, Recall, dan F1-Score sehingga diperoleh model terbaik yang dapat digunakan sebagai dasar pengembangan sistem pendukung diagnosis berbasis Artificial Intelligence.

---

# 2. Business Understanding

## 2.1 Permasalahan Dunia Nyata

Brain Tumor merupakan penyakit yang memiliki tingkat risiko tinggi apabila tidak didiagnosis secara cepat dan tepat. Pemeriksaan menggunakan MRI mampu memberikan informasi yang sangat detail mengenai kondisi jaringan otak, namun proses interpretasi citra masih bergantung pada dokter spesialis radiologi.

Beberapa rumah sakit juga mengalami keterbatasan jumlah radiolog sehingga proses analisis citra MRI membutuhkan waktu yang cukup lama. Selain itu, hasil interpretasi juga dipengaruhi oleh pengalaman masing-masing tenaga medis sehingga memungkinkan terjadinya perbedaan diagnosis.

Perkembangan Artificial Intelligence memberikan peluang untuk mengembangkan sistem klasifikasi otomatis yang mampu membantu tenaga medis dalam mengidentifikasi jenis Brain Tumor secara lebih cepat dan konsisten.

---

## 2.2 Literature Review

Penelitian oleh **Gómez-Guzmán et al. (2023)** menunjukkan bahwa penggunaan Convolutional Neural Network pada klasifikasi Brain Tumor MRI mampu menghasilkan performa yang sangat baik dibandingkan metode Machine Learning konvensional. CNN mampu mempelajari karakteristik citra MRI secara otomatis sehingga proses ekstraksi fitur tidak perlu dilakukan secara manual.

Mahesh et al. (2023) mengembangkan metode **CE-EEN-B0**, yaitu pengembangan arsitektur EfficientNetB0 yang dikombinasikan dengan teknik ekstraksi kontur. Penelitian tersebut menunjukkan bahwa EfficientNetB0 mampu meningkatkan performa klasifikasi Brain Tumor MRI dengan akurasi yang sangat tinggi.

Penelitian oleh Saeedi et al. (2023) membandingkan beberapa metode Deep Learning dan Machine Learning untuk klasifikasi Brain Tumor berbasis MRI. Hasil penelitian menunjukkan bahwa pendekatan Deep Learning memberikan performa yang lebih baik dibandingkan algoritma Machine Learning tradisional karena mampu mempelajari fitur citra secara otomatis.

Tripathy et al. (2023) menerapkan Transfer Learning menggunakan keluarga EfficientNet pada klasifikasi Brain Tumor MRI. Penelitian tersebut menunjukkan bahwa penggunaan Transfer Learning mampu mempercepat proses pelatihan model sekaligus meningkatkan akurasi klasifikasi.

Selain itu, Islam et al. (2024) mengembangkan model BrainNet berbasis EfficientNet yang berhasil memperoleh performa klasifikasi sangat tinggi melalui optimasi arsitektur dan proses Fine Tuning.

Berdasarkan beberapa penelitian tersebut dapat disimpulkan bahwa Deep Learning merupakan pendekatan yang sangat efektif untuk melakukan klasifikasi Brain Tumor MRI. Oleh karena itu penelitian ini melakukan perbandingan antara **Custom CNN** dan **EfficientNetB0** untuk mengetahui model terbaik pada dataset yang digunakan.

---

## 2.3 Tujuan Proyek

Tujuan utama penelitian ini adalah membangun sistem klasifikasi Brain Tumor berbasis Deep Learning menggunakan dua model yang berbeda kemudian membandingkan performanya.

Secara rinci tujuan penelitian meliputi:

- Mengimplementasikan model Custom CNN.
- Mengimplementasikan model EfficientNetB0 menggunakan Transfer Learning.
- Melakukan proses pelatihan model menggunakan dataset Brain Tumor MRI.
- Mengevaluasi performa model menggunakan Accuracy, Precision, Recall, F1-Score, Classification Report, dan Confusion Matrix.
- Menentukan model terbaik berdasarkan hasil evaluasi.

---

## 2.4 User Sistem

Sistem yang dikembangkan pada penelitian ini ditujukan untuk beberapa pengguna, antara lain:

- Dokter spesialis radiologi sebagai alat bantu diagnosis awal.
- Rumah sakit yang memerlukan sistem klasifikasi otomatis.
- Peneliti di bidang Computer Vision dan Medical Image Analysis.
- Mahasiswa yang mempelajari implementasi Deep Learning pada citra medis.

Sistem ini **tidak bertujuan menggantikan keputusan dokter**, melainkan sebagai sistem pendukung keputusan (*Decision Support System*) yang dapat membantu mempercepat proses identifikasi awal jenis Brain Tumor.

---

## 2.5 Solusi dan Manfaat Implementasi AI

Solusi yang ditawarkan pada penelitian ini adalah membangun model klasifikasi berbasis Deep Learning menggunakan dua pendekatan berbeda, yaitu **Custom CNN** dan **EfficientNetB0**.

Model yang telah dilatih diharapkan mampu mengklasifikasikan citra MRI ke dalam empat kategori Brain Tumor secara otomatis.

Implementasi Artificial Intelligence pada penelitian ini memberikan beberapa manfaat, antara lain:

- Membantu proses identifikasi Brain Tumor secara otomatis.
- Mengurangi waktu analisis citra MRI.
- Memberikan hasil klasifikasi yang konsisten.
- Menjadi dasar pengembangan aplikasi deteksi Brain Tumor berbasis web menggunakan Flask.
- Mendukung perkembangan penelitian Artificial Intelligence pada bidang Medical Image Analysis.

---
# 3. Data Understanding

## 3.1 Sumber Dataset

Dataset yang digunakan pada penelitian ini merupakan **Brain Tumor MRI Dataset** yang diperoleh dari platform **Kaggle**. Dataset tersebut berisi kumpulan citra Magnetic Resonance Imaging (MRI) otak yang telah diberi label sesuai jenis tumor. Dataset ini banyak digunakan sebagai benchmark pada penelitian Computer Vision dan Deep Learning karena memiliki kualitas citra yang baik serta distribusi kelas yang cukup representatif.

Dataset terdiri dari empat kategori utama yaitu:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

Keempat kelas tersebut digunakan sebagai target klasifikasi pada penelitian ini.

> Link Dataset:
>
> https://www.kaggle.com/

---

## 3.2 Deskripsi Dataset

Dataset berupa kumpulan citra MRI otak dalam format gambar digital. Seluruh citra telah dikelompokkan berdasarkan jenis tumor sehingga memudahkan proses pelatihan model Deep Learning.

### Karakteristik Dataset

| Informasi | Keterangan |
|------------|------------------------------|
| Jenis Dataset | Image Dataset |
| Format File | JPG / JPEG / PNG |
| Jenis Data | RGB Image |
| Target | Multiclass Classification |
| Jumlah Kelas | 4 |
| Framework | TensorFlow & Keras |

---

## 3.3 Deskripsi Kelas

Dataset terdiri dari empat kelas yang merepresentasikan kondisi otak berdasarkan hasil pemeriksaan MRI.

| Kelas | Deskripsi |
|--------|-----------|
| Glioma | Tumor yang berasal dari sel glial dan termasuk salah satu jenis tumor otak yang paling sering ditemukan. |
| Meningioma | Tumor yang tumbuh pada lapisan pelindung otak (meninges). Umumnya bersifat jinak namun tetap memerlukan penanganan medis. |
| Pituitary | Tumor yang tumbuh pada kelenjar pituitari dan dapat memengaruhi produksi hormon tubuh. |
| No Tumor | Menunjukkan citra MRI otak yang tidak memiliki indikasi adanya tumor. |

---

## 3.4 Struktur Dataset

Dataset disusun menggunakan struktur folder sehingga setiap folder mewakili satu kelas.

```text
dataset/

│

├── train/

│     ├── Glioma/

│     ├── Meningioma/

│     ├── Pituitary/

│     └── No Tumor/

│

├── validation/

│     ├── Glioma/

│     ├── Meningioma/

│     ├── Pituitary/

│     └── No Tumor/

│

└── test/

      ├── Glioma/

      ├── Meningioma/

      ├── Pituitary/

      └── No Tumor/
```

Struktur tersebut memudahkan TensorFlow dalam melakukan proses pembacaan dataset menggunakan fungsi `image_dataset_from_directory()`.

---

## 3.5 Contoh Dataset

Berikut merupakan contoh citra MRI pada masing-masing kelas.

```md
![Sample Dataset](images/sample_dataset.png)
```

> Gambar di atas dihasilkan dari notebook `uas_model.ipynb`.

---

## 3.6 Distribusi Dataset

Distribusi jumlah citra pada masing-masing kelas divisualisasikan menggunakan diagram batang (Bar Chart).

```md
![Dataset Distribution](images/dataset_distribution.png)
```

Visualisasi tersebut digunakan untuk mengetahui apakah terdapat ketidakseimbangan jumlah data (imbalanced dataset) antar kelas.

---

## 3.7 Analisis Dataset

Berdasarkan hasil eksplorasi awal terhadap dataset diperoleh beberapa informasi sebagai berikut.

1. Dataset terdiri dari empat kelas yang saling eksklusif sehingga termasuk ke dalam permasalahan **Multiclass Image Classification**.

2. Seluruh data berbentuk citra MRI sehingga pendekatan **Computer Vision** menggunakan Deep Learning sangat sesuai diterapkan.

3. Setiap kelas memiliki karakteristik tekstur dan pola citra yang berbeda sehingga model CNN mampu mempelajari fitur-fitur penting secara otomatis.

4. Dataset telah dipisahkan menjadi data **Training**, **Validation**, dan **Testing**, sehingga proses evaluasi model dapat dilakukan secara lebih objektif.

5. Seluruh citra akan diubah menjadi ukuran **224 × 224 piksel** agar sesuai dengan input model CNN maupun EfficientNetB0.

---

## 3.8 Target Klasifikasi

Target pada penelitian ini adalah mengklasifikasikan citra MRI Brain Tumor ke dalam salah satu dari empat kelas berikut.

| Label | Output |
|--------|--------|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

Proses klasifikasi dilakukan menggunakan pendekatan **Supervised Learning**, di mana setiap citra telah memiliki label yang digunakan sebagai acuan selama proses pelatihan model.

---

## 3.9 Kesimpulan Data Understanding

Berdasarkan hasil analisis awal dapat disimpulkan bahwa dataset Brain Tumor MRI memiliki karakteristik yang sesuai untuk diterapkan pada metode Deep Learning. Dataset telah memiliki label yang jelas, terdiri dari empat kelas, serta dipisahkan ke dalam data pelatihan, validasi, dan pengujian sehingga memungkinkan proses evaluasi dilakukan secara objektif.

Tahap selanjutnya adalah melakukan **Exploratory Data Analysis (EDA)** untuk memperoleh gambaran lebih mendalam mengenai distribusi data, karakteristik setiap kelas, serta kondisi dataset sebelum dilakukan proses pelatihan model.
# 4. Exploratory Data Analysis (EDA)

## 4.1 Pendahuluan

Exploratory Data Analysis (EDA) merupakan tahap awal yang bertujuan untuk memahami karakteristik dataset sebelum dilakukan proses pelatihan model. Tahapan ini penting untuk mengetahui distribusi data, keseimbangan antar kelas, kualitas citra, serta potensi permasalahan yang dapat memengaruhi performa model Deep Learning.

Pada penelitian ini, EDA dilakukan menggunakan beberapa visualisasi seperti distribusi jumlah data, contoh citra MRI, serta analisis karakteristik setiap kelas.

---

## 4.2 Distribusi Dataset

Distribusi jumlah data pada masing-masing kelas divisualisasikan menggunakan diagram batang (Bar Chart).

```md
![Dataset Distribution](images/dataset_distribution.png)
```

### Analisis

Berdasarkan visualisasi tersebut dapat diketahui bahwa dataset terdiri atas empat kelas utama, yaitu **Glioma**, **Meningioma**, **Pituitary**, dan **No Tumor**. Setiap kelas memiliki jumlah data yang relatif seimbang sehingga risiko bias model terhadap kelas tertentu menjadi lebih kecil.

Distribusi data yang seimbang merupakan kondisi yang baik dalam proses pelatihan Deep Learning karena model memperoleh jumlah sampel yang relatif sama pada setiap kelas.

---

## 4.3 Persentase Distribusi Kelas

Selain menggunakan diagram batang, distribusi kelas juga divisualisasikan menggunakan diagram lingkaran (Pie Chart).

```md
![Pie Chart](images/pie_chart.png)
```

### Analisis

Diagram lingkaran menunjukkan proporsi masing-masing kelas terhadap keseluruhan dataset.

Apabila proporsi antar kelas relatif sama, maka dataset dapat dikatakan memiliki distribusi yang cukup baik untuk proses klasifikasi multikelas (*Multiclass Classification*).

---

## 4.4 Contoh Citra MRI

Berikut merupakan contoh citra MRI pada masing-masing kelas.

```md
![Sample Dataset](images/sample_dataset.png)
```

### Analisis

Setiap kelas memiliki karakteristik visual yang berbeda.

- **Glioma** umumnya memiliki tekstur tidak beraturan dengan penyebaran tumor pada jaringan otak.
- **Meningioma** cenderung berada di bagian luar jaringan otak karena berasal dari lapisan meninges.
- **Pituitary** berada pada area kelenjar pituitari sehingga ukuran tumornya relatif lebih kecil.
- **No Tumor** menunjukkan struktur jaringan otak yang normal tanpa adanya indikasi tumor.

Perbedaan karakteristik tersebut memungkinkan model CNN maupun EfficientNetB0 mempelajari pola visual secara otomatis.

---

## 4.5 Dimensi Citra

Sebelum dilakukan proses pelatihan model, seluruh citra dianalisis untuk mengetahui ukuran gambar yang digunakan.

```md
![Image Size](images/image_size.png)
```

### Analisis

Hasil analisis menunjukkan bahwa ukuran citra pada dataset tidak seluruhnya sama sehingga diperlukan proses **Resize** pada tahap Data Preparation.

Seluruh gambar akan diubah menjadi ukuran **224 × 224 piksel** agar sesuai dengan input model CNN maupun EfficientNetB0.

---

## 4.6 Analisis Warna Citra

Dataset Brain Tumor MRI menggunakan citra berwarna (RGB).

Walaupun informasi utama berada pada struktur jaringan otak, penggunaan citra RGB tetap dipertahankan agar seluruh informasi piksel dapat dimanfaatkan oleh model Deep Learning.

Setiap citra memiliki tiga kanal warna:

- Red (R)
- Green (G)
- Blue (B)

yang direpresentasikan dalam bentuk tensor berukuran:

```text
(224, 224, 3)
```

---

## 4.7 Analisis Ketidakseimbangan Dataset

Salah satu tahapan penting dalam EDA adalah mengetahui apakah dataset mengalami **Class Imbalance**.

Berdasarkan hasil visualisasi distribusi kelas diperoleh bahwa jumlah data pada masing-masing kelas relatif seimbang sehingga tidak diperlukan teknik penyeimbangan data seperti:

- SMOTE
- Random Oversampling
- Random Undersampling

Kondisi ini memberikan keuntungan karena model dapat mempelajari setiap kelas secara lebih adil.

---

## 4.8 Insight Hasil EDA

Berdasarkan seluruh proses eksplorasi data dapat diperoleh beberapa informasi penting.

1. Dataset terdiri atas empat kelas yang sesuai untuk permasalahan **Multiclass Classification**.

2. Seluruh data berbentuk citra MRI sehingga pendekatan **Computer Vision** menggunakan Deep Learning sangat sesuai diterapkan.

3. Dataset memiliki distribusi kelas yang relatif seimbang sehingga risiko bias model menjadi lebih kecil.

4. Ukuran citra tidak seragam sehingga diperlukan proses **Resize** sebelum dilakukan pelatihan model.

5. Setiap kelas memiliki karakteristik visual yang berbeda sehingga model CNN maupun EfficientNetB0 diharapkan mampu mempelajari fitur-fitur penting secara otomatis.

---

## 4.9 Kesimpulan Exploratory Data Analysis

Hasil Exploratory Data Analysis menunjukkan bahwa dataset Brain Tumor MRI memiliki kualitas yang baik untuk digunakan dalam proses klasifikasi menggunakan Deep Learning.

Dataset telah memiliki label yang jelas, distribusi kelas yang relatif seimbang, serta karakteristik visual yang berbeda pada setiap kelas. Selanjutnya dilakukan tahap **Data Preparation** untuk mempersiapkan dataset sebelum digunakan pada proses pelatihan model.
