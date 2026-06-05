# 🎓 StudySync AI — Interactive EDA Dashboard

Dashboard interaktif berbasis **Streamlit** untuk mengeksplorasi dataset **Junyi Academy Learning Activity**. Dashboard ini dibuat berdasarkan hasil Exploratory Data Analysis (EDA) dan Business Questions dari Capstone Project **StudySync AI**.

---

## 📋 Deskripsi Proyek

**StudySync AI** bertujuan membangun sistem rekomendasi adaptif untuk platform e-learning menggunakan data aktivitas belajar nyata dari **Junyi Academy** (Taiwan). Dashboard ini menyajikan insight dari proses EDA secara visual dan interaktif.

### Dataset
- **Sumber:** [Kaggle — Junyi Academy Learning Activity](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)
- **Lisensi:** CC-BY-NC-SA-4.0
- **Files:**
  - `Log_Problem.csv` — Log aktivitas pengerjaan soal (16+ juta baris)
  - `Info_Content.csv` — Informasi konten/topik (1,330 topik)
  - `Info_UserData.csv` — Data profil pengguna (72,758 siswa)

---

## 🚀 Cara Menjalankan

### 1. Clone / Download Repository

```bash
git clone https://github.com/<username>/studysync-dashboard.git
cd studysync-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Siapkan Data

**Opsi A — Menggunakan data yang sudah diproses (Recommended):**

Letakkan salah satu file berikut di folder yang sama dengan `dashboard.py`:
- `studysync_ai_ready_dataset.csv` ← output akhir dari notebook EDA
- `Processed_Dataset_500k.csv` atau `Processed_Dataset.csv`

**Opsi B — Menggunakan raw dataset dari Kaggle:**

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (butuh kaggle.json)
kaggle datasets download -d junyiacademy/learning-activity-public-dataset-by-junyi-academy
unzip learning-activity-public-dataset-by-junyi-academy.zip
```

Letakkan `Log_Problem.csv`, `Info_Content.csv`, dan `Info_UserData.csv` di folder yang sama.

**Opsi C — Tanpa data (Demo Mode):**

Jalankan langsung tanpa file apapun. Dashboard akan generate **synthetic demo data** secara otomatis yang merepresentasikan pola dataset asli.

### 4. Jalankan Dashboard

```bash
streamlit run dashboard.py
```

Dashboard akan terbuka otomatis di browser: `http://localhost:8501`

---

## 📊 Fitur Dashboard

### Tab 1 — EDA: Distribusi
- Distribusi tingkat kesulitan topik (bar + pie chart)
- Distribusi success rate dan waktu per soal (histogram)
- Success rate & jumlah attempts per difficulty level
- Heatmap korelasi antar fitur numerik

### Tab 2 — Business Questions
| # | Pertanyaan |
|---|-----------|
| Q1 | Apakah siswa yang belajar lebih lama pada topik sulit punya success rate lebih tinggi? |
| Q2 | Bagaimana pola waktu per soal: siswa sukses tinggi (≥70%) vs sukses rendah (<70%)? |
| Q3 | Adakah perbedaan jumlah soal dikerjakan berdasarkan tingkat kesulitan? |
| Q4 | Bagaimana hubungan total waktu belajar dengan success rate? |

### Tab 3 — User Insights
- Distribusi siswa per grade/kelas
- Distribusi gender
- Distribusi poin & badge
- Rasio self-coach vs teacher-assisted

### Tab 4 — Raw Data
- Preview dataset dengan kontrol jumlah baris
- Statistik deskriptif
- Laporan missing values
- Export filtered data ke CSV

### Sidebar Filters
- Filter berdasarkan **difficulty level**
- Filter berdasarkan **success rate range**
- Filter berdasarkan **max total time spent**
- Toggle anotasi nilai pada chart

---

## 🗂️ Struktur File

```
studysync-dashboard/
├── dashboard.py              # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # Dokumentasi ini
├── .gitignore                # File yang dikecualikan dari Git
│
├── (opsional — letakkan di sini)
│   ├── studysync_ai_ready_dataset.csv
│   ├── Processed_Dataset.csv
│   ├── Log_Problem.csv
│   ├── Info_Content.csv
│   └── Info_UserData.csv
```

> ⚠️ **Catatan:** File CSV dataset **tidak diikutsertakan** di repository karena ukurannya sangat besar. Download dari Kaggle sesuai instruksi di atas.

---

## 🔬 EDA Highlights

Berdasarkan analisis dari notebook, beberapa temuan utama:

1. **Dominasi topik mudah** — Mayoritas topik memiliki difficulty `easy`, mencerminkan fokus pada materi dasar SD/SMP.
2. **Success rate menurun seiring kesulitan** — Easy: ~82%, Normal: ~68%, Hard: ~54%.
3. **Persistence pada topik sulit** — Siswa justru mengerjakan lebih banyak soal di topik `hard`.
4. **Waktu belajar & keberhasilan** — Ada korelasi positif, namun dengan _diminishing returns_ setelah titik optimal.
5. **Self-coach dominan** — Mayoritas siswa belajar mandiri, menegaskan pentingnya adaptive recommendation.

---

## 🛠️ Tech Stack

| Library | Kegunaan |
|---------|----------|
| Streamlit | Framework dashboard interaktif |
| Plotly | Visualisasi interaktif (bar, scatter, heatmap, dll) |
| Pandas | Manipulasi dan analisis data |
| NumPy | Operasi numerik |
| Matplotlib / Seaborn | Visualisasi tambahan |

---

## 👤 Author

Capstone Project — StudySync AI  
Dataset: Junyi Academy (junyiacademy) via Kaggle

---

## 📄 Lisensi

Dataset dilisensikan di bawah [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Kode dashboard ini untuk keperluan pendidikan dan non-komersial.
