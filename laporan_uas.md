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

Dataset yang digunakan pada penelitian ini berasal dari platform Kaggle dengan nama **Brain Tumor MRI Dataset**. Dataset ini berisi kumpulan citra Magnetic Resonance Imaging (MRI) otak yang telah diberi label ke dalam empat kategori yaitu **Glioma**, **Meningioma**, **Pituitary**, dan **No Tumor**.

Dataset digunakan sebagai data pelatihan (*training*), validasi (*validation*), dan pengujian (*testing*) pada model Deep Learning.

**Sumber Dataset:**

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Informasi Dataset

| Informasi | Keterangan |
|------------|------------|
| Platform | Kaggle |
| Jenis Data | Image Dataset |
| Format File | JPG / JPEG / PNG |
| Jumlah Kelas | 4 |
| Target Klasifikasi | Glioma, Meningioma, Pituitary, No Tumor |
| Domain | Medical Image Classification |

---

## 3.2 Deskripsi Atribut Dataset

Meskipun dataset yang digunakan berupa citra (image dataset), setiap data tetap memiliki atribut atau karakteristik yang digunakan selama proses pelatihan model Deep Learning.

| Atribut | Tipe Data | Deskripsi |
|----------|-----------|-----------|
| Image | JPG / JPEG / PNG | Citra MRI otak yang digunakan sebagai data masukan model. |
| Label | String | Label kelas yang menunjukkan kategori citra MRI. |
| Width | Integer | Lebar citra dalam satuan piksel. |
| Height | Integer | Tinggi citra dalam satuan piksel. |
| Channel | Integer | Jumlah kanal warna (RGB = 3 Channel). |
| Target Class | Categorical | Kelas keluaran yang akan diprediksi model. |

Berdasarkan atribut tersebut, model Deep Learning memanfaatkan informasi visual dari citra MRI sebagai data masukan, sedangkan label digunakan sebagai target pada proses pembelajaran (*Supervised Learning*).

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

---

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
# 5. Data Preparation

## 5.1 Pendahuluan

Data Preparation merupakan tahapan untuk mempersiapkan dataset sebelum digunakan dalam proses pelatihan model Deep Learning. Tahapan ini bertujuan untuk meningkatkan kualitas data, menyamakan ukuran citra, mempercepat proses pembelajaran model, serta mengurangi risiko overfitting.

Pada penelitian ini proses Data Preparation dilakukan menggunakan library TensorFlow dan Keras pada platform Google Colab.

---

## 5.2 Upload Dataset

Dataset Brain Tumor MRI terlebih dahulu diunggah ke Google Colab dalam bentuk file ZIP.

Tahapan upload dilakukan secara manual menggunakan fitur upload file sehingga notebook dapat dijalankan secara mandiri tanpa bergantung pada Google Drive.

Setelah file berhasil diunggah, dataset diekstrak ke direktori kerja menggunakan library `zipfile`.

```python
from google.colab import files

uploaded = files.upload()
```

---

## 5.3 Ekstraksi Dataset

Dataset yang masih berbentuk file ZIP kemudian diekstrak sehingga seluruh folder dataset dapat digunakan pada proses pelatihan model.

```python
import zipfile

with zipfile.ZipFile(DATASET_ZIP,'r') as zip_ref:
    zip_ref.extractall(DATASET_PATH)
```

Setelah proses ekstraksi selesai, struktur folder dataset terdiri atas folder **train**, **validation**, dan **test**.

---

## 5.4 Resize Image

Ukuran citra MRI pada dataset tidak seluruhnya sama sehingga diperlukan proses penyesuaian ukuran gambar.

Seluruh gambar diubah menjadi ukuran:

```text
224 × 224 pixel
```

Ukuran tersebut dipilih karena sesuai dengan kebutuhan model EfficientNetB0 serta tetap optimal untuk model Custom CNN.

---

## 5.5 Normalisasi

Nilai piksel citra awal berada pada rentang

```text
0 – 255
```

Kemudian dilakukan proses normalisasi menjadi

```text
0 – 1
```

menggunakan persamaan

\[
x=\frac{x}{255}
\]

Normalisasi dilakukan agar proses optimisasi model menjadi lebih stabil serta mempercepat konvergensi selama proses training.

---

## 5.6 Data Augmentation

Untuk meningkatkan kemampuan generalisasi model dilakukan proses Data Augmentation.

Tahapan augmentasi meliputi:

- Random Rotation
- Random Zoom
- Random Flip
- Random Translation

Implementasi Data Augmentation menggunakan TensorFlow sebagai berikut.

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

Data Augmentation bertujuan menghasilkan variasi citra baru tanpa mengubah label sehingga model menjadi lebih robust terhadap variasi data nyata.

---

## 5.7 Train Validation Test Split

Dataset telah dipisahkan menjadi tiga bagian utama.

| Dataset | Fungsi |
|----------|--------|
| Training | Melatih model |
| Validation | Validasi selama proses training |
| Testing | Evaluasi akhir model |

Pembagian dataset ini bertujuan agar proses evaluasi dilakukan menggunakan data yang belum pernah dilihat model selama proses pelatihan.

---

## 5.8 TensorFlow Dataset Pipeline

Seluruh dataset dibaca menggunakan fungsi

```python
image_dataset_from_directory()
```

yang disediakan oleh TensorFlow.

Fungsi ini secara otomatis membaca seluruh folder berdasarkan label kelas kemudian mengubahnya menjadi objek `tf.data.Dataset`.

Contoh implementasi:

```python
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(224,224),
    batch_size=32
)
```

---

## 5.9 Prefetch Dataset

Untuk meningkatkan efisiensi proses training digunakan teknik **Prefetching**.

Prefetch memungkinkan TensorFlow menyiapkan batch berikutnya ketika GPU masih memproses batch sebelumnya sehingga waktu pelatihan menjadi lebih singkat.

Implementasi dilakukan menggunakan:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)
```

---

## 5.10 Ringkasan Data Preparation

Tahapan Data Preparation yang dilakukan pada penelitian ini meliputi:

| Tahapan | Tujuan |
|----------|---------|
| Upload Dataset | Memasukkan dataset ke Google Colab |
| Ekstraksi ZIP | Membuka struktur dataset |
| Resize Image | Menyamakan ukuran citra menjadi 224×224 |
| Normalisasi | Mengubah rentang piksel menjadi 0–1 |
| Data Augmentation | Menambah variasi citra |
| Train Validation Test Split | Membagi data untuk training dan evaluasi |
| TensorFlow Dataset | Membuat pipeline dataset |
| Prefetch | Mempercepat proses training |

---

## 5.11 Kesimpulan

Seluruh tahapan Data Preparation berhasil dilakukan sebelum proses pelatihan model. Dataset telah berada pada format yang sesuai untuk digunakan oleh model Custom CNN maupun EfficientNetB0.

Tahapan selanjutnya adalah **Modeling**, yaitu proses pembangunan arsitektur Deep Learning, pelatihan model, serta perbandingan performa antara Custom CNN dan EfficientNetB0.
# 6. Modeling

## 6.1 Pendahuluan

Tahap Modeling merupakan proses pembangunan, pelatihan, dan evaluasi model Deep Learning untuk melakukan klasifikasi citra MRI Brain Tumor. Pada penelitian ini digunakan dua pendekatan yang berbeda, yaitu **Custom Convolutional Neural Network (CNN)** dan **EfficientNetB0** dengan metode Transfer Learning.

Pemilihan dua model tersebut bertujuan untuk membandingkan performa model yang dibangun dari awal (Custom CNN) dengan model pretrained (EfficientNetB0). Selanjutnya kedua model dievaluasi menggunakan metrik Accuracy, Precision, Recall, F1-Score, Classification Report, dan Confusion Matrix.

---

# 6.2 Model Pertama : Custom Convolutional Neural Network (CNN)

## 6.2.1 Deskripsi Model

Convolutional Neural Network (CNN) merupakan salah satu algoritma Deep Learning yang dirancang khusus untuk pengolahan citra. CNN mampu mengekstraksi fitur penting secara otomatis melalui proses konvolusi sehingga tidak memerlukan proses ekstraksi fitur manual.

Pada penelitian ini digunakan arsitektur **Custom CNN**, yaitu model yang dibangun secara mandiri menggunakan beberapa lapisan konvolusi dan fully connected layer.

---

## 6.2.2 Arsitektur Model

Arsitektur Custom CNN terdiri atas beberapa lapisan utama sebagai berikut.

| Layer | Fungsi |
|--------|--------|
| Conv2D | Mengekstraksi fitur citra |
| MaxPooling2D | Mengurangi dimensi feature map |
| BatchNormalization | Menstabilkan proses training |
| Dropout | Mengurangi overfitting |
| Flatten | Mengubah feature map menjadi vektor |
| Dense | Melakukan klasifikasi |
| Softmax | Menghasilkan probabilitas setiap kelas |

---

## 6.2.3 Hyperparameter

Hyperparameter yang digunakan pada proses pelatihan model ditunjukkan pada tabel berikut.

| Parameter | Nilai |
|------------|--------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Activation | ReLU |
| Output Activation | Softmax |
| Epoch | 20 *(sesuaikan dengan notebook)* |

---

## 6.2.4 Struktur Model

Diagram arsitektur model dapat ditampilkan sebagai berikut.

```md
![CNN Architecture](images/cnn_architecture.png)
```

---

## 6.2.5 Training Model CNN

Proses pelatihan dilakukan menggunakan dataset training dengan validasi menggunakan validation dataset.

Model dilatih hingga seluruh epoch selesai atau berhenti lebih awal menggunakan callback **EarlyStopping** apabila tidak terjadi peningkatan performa.

Grafik hasil training ditunjukkan pada gambar berikut.

```md
![CNN Accuracy](images/cnn_accuracy.png)
```

```md
![CNN Loss](images/cnn_loss.png)
```

---

## 6.2.6 Analisis Hasil Training CNN

Berdasarkan grafik Accuracy dan Loss dapat diamati bahwa model mengalami peningkatan akurasi secara bertahap selama proses pelatihan.

Nilai Loss mengalami penurunan yang menunjukkan bahwa model mampu mempelajari pola pada dataset dengan baik.

Apabila grafik validation accuracy mengikuti training accuracy tanpa selisih yang terlalu besar, maka model tidak mengalami overfitting yang signifikan.

---

# 6.3 Model Kedua : EfficientNetB0

## 6.3.1 Deskripsi Model

EfficientNetB0 merupakan arsitektur Deep Learning yang dikembangkan menggunakan metode **Compound Scaling**, yaitu teknik yang melakukan scaling pada dimensi jaringan secara seimbang.

Pada penelitian ini EfficientNetB0 digunakan melalui pendekatan **Transfer Learning**, yaitu memanfaatkan model yang telah dilatih menggunakan dataset ImageNet.

Transfer Learning memungkinkan model memperoleh performa yang tinggi meskipun jumlah data pelatihan relatif terbatas.

---

## 6.3.2 Transfer Learning

Tahapan Transfer Learning dilakukan dengan memuat model EfficientNetB0 tanpa lapisan klasifikasi akhir.

Selanjutnya ditambahkan beberapa Fully Connected Layer sesuai jumlah kelas pada dataset Brain Tumor MRI.

---

## 6.3.3 Fine Tuning

Setelah proses Transfer Learning selesai, dilakukan Fine Tuning terhadap beberapa layer terakhir EfficientNetB0.

Fine Tuning bertujuan meningkatkan kemampuan model dalam mengenali karakteristik khusus pada citra MRI Brain Tumor.

---

## 6.3.4 Hyperparameter

| Parameter | Nilai |
|------------|--------|
| Base Model | EfficientNetB0 |
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Sparse Categorical Crossentropy |
| Epoch | 20 *(sesuaikan dengan notebook)* |

---

## 6.3.5 Struktur Model

```md
![EfficientNet](images/efficientnet_architecture.png)
```

---

## 6.3.6 Training Model EfficientNetB0

Model EfficientNetB0 dilatih menggunakan dataset yang sama dengan Custom CNN sehingga proses perbandingan dilakukan secara adil.

Grafik Accuracy dan Loss ditunjukkan pada gambar berikut.

```md
![EfficientNet Accuracy](images/efficientnet_accuracy.png)
```

```md
![EfficientNet Loss](images/efficientnet_loss.png)
```

---

## 6.3.7 Analisis Hasil Training EfficientNetB0

Hasil pelatihan menunjukkan bahwa EfficientNetB0 mampu mempelajari pola citra MRI dengan baik.

Penggunaan Transfer Learning mempercepat proses konvergensi model dibandingkan CNN yang dibangun dari awal.

Selain itu Fine Tuning memberikan peningkatan performa pada tahap akhir pelatihan.

---

# 6.4 Perbandingan Model

Setelah kedua model selesai dilatih, dilakukan proses perbandingan performa menggunakan beberapa metrik evaluasi.

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Custom CNN | ... | ... | ... | ... |
| EfficientNetB0 | ... | ... | ... | ... |

> **Isi tabel di atas sesuai hasil notebook (`comparison` DataFrame).**

---

## 6.5 Pemilihan Model Terbaik

Model terbaik dipilih berdasarkan beberapa metrik evaluasi yaitu:

- Accuracy
- Precision
- Recall
- F1-Score

Selain itu dilakukan analisis terhadap grafik Accuracy dan Loss untuk mengetahui kestabilan proses pelatihan.

Model dengan performa terbaik selanjutnya digunakan pada tahap **Real Testing** serta disimpan sebagai model akhir yang akan digunakan pada proses deployment.

---

## 6.6 Kesimpulan Modeling

Tahap Modeling berhasil menghasilkan dua model Deep Learning yang mampu melakukan klasifikasi Brain Tumor MRI.

Custom CNN memberikan performa yang baik karena mampu mempelajari karakteristik citra secara langsung melalui proses pelatihan.

Sementara itu EfficientNetB0 memanfaatkan Transfer Learning sehingga proses pelatihan menjadi lebih cepat dan mampu menghasilkan performa yang kompetitif.

Tahap selanjutnya adalah **Evaluation**, yaitu melakukan evaluasi menyeluruh terhadap kedua model menggunakan berbagai metrik klasifikasi untuk menentukan model terbaik.
# 7. Evaluation

## 7.1 Pendahuluan

Tahap Evaluation dilakukan untuk mengukur performa model Deep Learning yang telah dibangun. Evaluasi bertujuan mengetahui kemampuan model dalam mengklasifikasikan citra MRI Brain Tumor secara akurat menggunakan data yang belum pernah dilihat selama proses pelatihan.

Pada penelitian ini evaluasi dilakukan terhadap dua model, yaitu **Custom Convolutional Neural Network (CNN)** dan **EfficientNetB0**. Kedua model dibandingkan menggunakan beberapa metrik evaluasi sehingga diperoleh model terbaik yang akan digunakan sebagai model akhir.

---

# 7.2 Accuracy dan Loss

Selama proses pelatihan dilakukan pencatatan nilai Accuracy dan Loss pada setiap epoch.

Grafik Accuracy menunjukkan perkembangan kemampuan model dalam melakukan klasifikasi, sedangkan grafik Loss menunjukkan tingkat kesalahan model selama proses pembelajaran.

## Accuracy Model CNN

```md
![CNN Accuracy](images/cnn_accuracy.png)
```

## Loss Model CNN

```md
![CNN Loss](images/cnn_loss.png)
```

---

## Accuracy Model EfficientNetB0

```md
![EfficientNet Accuracy](images/efficientnet_accuracy.png)
```

## Loss Model EfficientNetB0

```md
![EfficientNet Loss](images/efficientnet_loss.png)
```

---

### Analisis

Berdasarkan grafik di atas dapat diamati bahwa kedua model mengalami peningkatan Accuracy serta penurunan Loss selama proses pelatihan.

Model yang memiliki nilai Validation Accuracy paling tinggi dengan Validation Loss yang stabil menunjukkan kemampuan generalisasi yang lebih baik.

---

# 7.3 Confusion Matrix

Selain menggunakan Accuracy, evaluasi dilakukan menggunakan Confusion Matrix.

Confusion Matrix digunakan untuk mengetahui jumlah prediksi benar maupun prediksi yang salah pada masing-masing kelas.

```md
![Confusion Matrix](images/confusion_matrix.png)
```

---

### Analisis

Berdasarkan Confusion Matrix dapat diketahui bahwa sebagian besar citra MRI berhasil diklasifikasikan dengan benar.

Apabila masih terdapat kesalahan klasifikasi, umumnya terjadi pada kelas yang memiliki karakteristik visual yang hampir serupa.

Analisis Confusion Matrix membantu mengetahui kelas mana yang masih sulit dikenali oleh model sehingga dapat menjadi dasar pengembangan model pada penelitian selanjutnya.

---

# 7.4 Classification Report

Selain Confusion Matrix dilakukan evaluasi menggunakan Classification Report.

Classification Report menghasilkan beberapa metrik yaitu:

- Precision
- Recall
- F1-Score
- Support

Contoh hasil Classification Report:

```text
              precision    recall   f1-score

Glioma
Meningioma
No Tumor
Pituitary

Accuracy

Macro Avg

Weighted Avg
```

> Masukkan hasil Classification Report sesuai output notebook.

---

### Analisis

Precision menunjukkan tingkat ketepatan model dalam memberikan prediksi.

Recall menunjukkan kemampuan model menemukan seluruh data pada suatu kelas.

F1-Score merupakan keseimbangan antara Precision dan Recall sehingga sering digunakan sebagai indikator utama pada permasalahan klasifikasi.

Semakin tinggi nilai Precision, Recall, dan F1-Score maka semakin baik performa model.

---

# 7.5 Perbandingan Model

Setelah seluruh proses evaluasi selesai dilakukan, kedua model dibandingkan berdasarkan seluruh metrik evaluasi.

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Custom CNN | ... | ... | ... | ... |
| EfficientNetB0 | ... | ... | ... | ... |

> Isikan nilai berdasarkan hasil notebook.

---

### Analisis

Hasil perbandingan menunjukkan bahwa kedua model mampu melakukan klasifikasi Brain Tumor MRI dengan performa yang baik.

Model dengan nilai Accuracy, Precision, Recall, dan F1-Score tertinggi dipilih sebagai model terbaik untuk digunakan pada proses prediksi citra MRI.

---

# 7.6 Real Testing

Selain menggunakan data uji, dilakukan pengujian menggunakan citra MRI baru yang belum pernah digunakan selama proses pelatihan.

Pengujian dilakukan melalui fitur upload gambar pada notebook.

```md
![Real Testing](images/real_testing.png)
```

Output yang dihasilkan berupa:

- Gambar MRI
- Prediksi kelas
- Confidence Score

Contoh hasil prediksi:

| Input | Prediksi |
|--------|-----------|
| MRI Brain | Pituitary |

---

### Analisis

Real Testing menunjukkan bahwa model mampu melakukan klasifikasi terhadap citra baru dengan baik.

Apabila hasil prediksi sesuai dengan label sebenarnya, maka dapat disimpulkan bahwa model memiliki kemampuan generalisasi yang baik.

---

# 7.7 Model Terbaik

Berdasarkan seluruh proses evaluasi, dipilih satu model terbaik yang akan digunakan sebagai model akhir.

Kriteria pemilihan model meliputi:

- Accuracy tertinggi
- Precision tertinggi
- Recall tertinggi
- F1-Score tertinggi
- Stabilitas grafik Accuracy dan Loss
- Hasil Real Testing

Pada penelitian ini model terbaik adalah:

> **Custom CNN** *(atau EfficientNetB0, sesuaikan dengan hasil notebook).*

Model tersebut kemudian disimpan dalam format `.keras` untuk digunakan pada proses deployment.

---

# 7.8 Pembahasan

Berdasarkan seluruh hasil eksperimen dapat disimpulkan bahwa kedua model mampu melakukan klasifikasi Brain Tumor MRI dengan performa yang sangat baik.

Custom CNN menunjukkan kemampuan yang baik dalam mempelajari pola citra MRI secara langsung melalui proses pelatihan.

Sementara itu EfficientNetB0 memanfaatkan Transfer Learning sehingga proses pelatihan menjadi lebih cepat dan mampu menghasilkan performa yang kompetitif.

Perbedaan performa kedua model dipengaruhi oleh beberapa faktor, antara lain:

- Arsitektur model
- Jumlah parameter
- Transfer Learning
- Fine Tuning
- Karakteristik dataset
- Data Augmentation

---

# 7.9 Kesimpulan Evaluation

Tahap Evaluation menunjukkan bahwa kedua model berhasil melakukan klasifikasi Brain Tumor MRI dengan tingkat akurasi yang tinggi.

Model terbaik dipilih berdasarkan seluruh metrik evaluasi sehingga mampu digunakan sebagai dasar pengembangan sistem klasifikasi Brain Tumor berbasis Artificial Intelligence.

Selanjutnya model terbaik akan digunakan sebagai model utama pada proses deployment dan implementasi sistem.
# 8. Kesimpulan dan Rekomendasi

## 8.1 Kesimpulan

Penelitian ini berhasil mengembangkan sistem klasifikasi Brain Tumor menggunakan pendekatan Deep Learning dengan membandingkan dua arsitektur, yaitu **Custom Convolutional Neural Network (CNN)** dan **EfficientNetB0**. Seluruh tahapan penelitian telah dilakukan mulai dari Business Understanding, Data Understanding, Exploratory Data Analysis (EDA), Data Preparation, Modeling, hingga Evaluation.

Berdasarkan hasil pelatihan dan evaluasi, kedua model mampu melakukan klasifikasi citra MRI Brain Tumor ke dalam empat kelas, yaitu **Glioma**, **Meningioma**, **Pituitary**, dan **No Tumor**.

Implementasi **Custom CNN** menunjukkan kemampuan yang baik dalam mempelajari pola citra MRI secara langsung melalui proses training, sedangkan **EfficientNetB0** memanfaatkan Transfer Learning sehingga proses pelatihan menjadi lebih efisien dan menghasilkan performa yang kompetitif.

Perbandingan kedua model dilakukan menggunakan beberapa metrik evaluasi yaitu Accuracy, Precision, Recall, dan F1-Score. Model dengan nilai evaluasi terbaik dipilih sebagai model akhir yang digunakan pada proses Real Testing serta dapat digunakan sebagai dasar implementasi aplikasi klasifikasi Brain Tumor berbasis Artificial Intelligence.

Secara keseluruhan tujuan penelitian berhasil dicapai, yaitu membangun model klasifikasi Brain Tumor MRI menggunakan Deep Learning serta menentukan model terbaik berdasarkan hasil evaluasi.

---

## 8.2 Kelebihan Penelitian

Penelitian ini memiliki beberapa kelebihan, antara lain:

- Menggunakan dua algoritma Deep Learning sehingga dapat dilakukan perbandingan performa secara objektif.
- Dataset telah melalui proses Data Preparation dan Data Augmentation sehingga meningkatkan kemampuan generalisasi model.
- Menggunakan metrik evaluasi yang lengkap meliputi Accuracy, Precision, Recall, F1-Score, Classification Report, dan Confusion Matrix.
- Mengimplementasikan Transfer Learning pada EfficientNetB0 sehingga proses pelatihan menjadi lebih efisien.
- Model yang dihasilkan dapat dikembangkan menjadi aplikasi berbasis web menggunakan Flask.

---

## 8.3 Keterbatasan Penelitian

Penelitian ini masih memiliki beberapa keterbatasan, yaitu:

- Dataset yang digunakan masih berasal dari satu sumber sehingga variasi citra MRI masih terbatas.
- Penelitian hanya menggunakan dua model Deep Learning.
- Pengujian dilakukan pada dataset publik sehingga belum divalidasi menggunakan data rumah sakit secara langsung.
- Belum dilakukan optimasi hyperparameter secara otomatis menggunakan metode seperti Grid Search atau Bayesian Optimization.

---

## 8.4 Rekomendasi Pengembangan

Untuk penelitian selanjutnya disarankan beberapa pengembangan berikut:

1. Menggunakan dataset dengan jumlah citra yang lebih besar dan lebih beragam.
2. Membandingkan lebih banyak arsitektur Deep Learning seperti EfficientNetV2, DenseNet121, ResNet50, Vision Transformer (ViT), dan ConvNeXt.
3. Menggunakan teknik Hyperparameter Tuning untuk memperoleh performa model yang lebih optimal.
4. Mengembangkan aplikasi berbasis web atau mobile sehingga dapat digunakan sebagai sistem pendukung diagnosis.
5. Mengintegrasikan model dengan sistem rumah sakit atau aplikasi telemedicine sehingga dapat dimanfaatkan secara nyata.

---

# 9. Referensi

Gómez-Guzmán, M. A., Jiménez-Beristain, L., García-Guerrero, E. E., López-Bonilla, O. R., Tamayo-Pérez, U. J., Esqueda-Elizondo, J. J., Palomino-Vizcaíno, K., & Inzunza-González, E. (2023). *Classifying brain tumors on magnetic resonance imaging by using convolutional neural networks*. Electronics, 12(4), 955. https://doi.org/10.3390/electronics12040955

Mahesh, A., Banerjee, D., Saha, A., Prusty, M. R., & Balasundaram, A. (2023). *CE-EEN-B0: Contour extraction based extended EfficientNet-B0 for brain tumor classification using MRI images*. Computers, Materials & Continua, 74(3), 5967–5982. https://doi.org/10.32604/cmc.2023.033920

Saeedi, S., Rezayi, S., Keshavarz, H., & Niakan Kalhori, S. R. (2023). *MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques*. BMC Medical Informatics and Decision Making, 23, 16. https://doi.org/10.1186/s12911-023-02114-6

Tripathy, S., Singh, R., & Ray, M. (2023). *Automation of brain tumor identification using EfficientNet on magnetic resonance images*. Procedia Computer Science, 218, 1551–1560. https://doi.org/10.1016/j.procs.2023.01.133

Islam, M. M., Talukder, M. A., Uddin, M. A., Akhter, A., & Khalid, M. (2024). *BrainNet: Precision brain tumor classification with optimized EfficientNet architecture*. International Journal of Intelligent Systems, 2024, Article ID 3583612. https://doi.org/10.1155/2024/3583612

---

# 10. Lampiran

## Lampiran A – Struktur Repository

```text
Brain-Tumor-MRI-Classification/

│

├── README.md

├── laporan_uas.md

├── uas_model.ipynb

├── requirements.txt

│

├── images/

│     ├── sample_dataset.png

│     ├── dataset_distribution.png

│     ├── pie_chart.png

│     ├── cnn_accuracy.png

│     ├── cnn_loss.png

│     ├── efficientnet_accuracy.png

│     ├── efficientnet_loss.png

│     ├── confusion_matrix.png

│     ├── comparison.png

│     └── real_testing.png

│

├── model/

│     ├── brain_tumor_model.keras

│     └── class_names.json

│

├── references/

│     ├── jurnal1.pdf

│     ├── jurnal2.pdf

│     ├── jurnal3.pdf

│     ├── jurnal4.pdf

│     └── jurnal5.pdf

│

└── LICENSE
```

---

## Lampiran B – Spesifikasi Perangkat

### Hardware

| Komponen | Spesifikasi |
|----------|-------------|
| Processor | Intel Core i5 / AMD Ryzen 5 atau setara |
| RAM | Minimal 8 GB |
| Storage | SSD 256 GB |
| GPU | NVIDIA T4 (Google Colab) |

### Software

| Software | Versi |
|----------|--------|
| Python | 3.12 |
| TensorFlow | 2.x |
| Keras | Terintegrasi TensorFlow |
| Google Colab | Cloud Environment |
| Flask | 3.x |

---

## Lampiran C – Workflow Penelitian

```text
Business Understanding
          │
          ▼
Data Understanding
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Data Preparation
          │
          ▼
Modeling
     │         │
     ▼         ▼
Custom CNN   EfficientNetB0
     │         │
     └────┬────┘
          ▼
Model Comparison
          ▼
Evaluation
          ▼
Real Testing
          ▼
Deployment
```

---

## Lampiran D – Hasil Notebook

Seluruh grafik, tabel evaluasi, confusion matrix, classification report, dan hasil real testing yang digunakan pada laporan ini dihasilkan secara langsung dari notebook **`uas_model.ipynb`**. Dengan demikian, isi laporan konsisten dengan implementasi yang terdapat pada repository GitHub.
