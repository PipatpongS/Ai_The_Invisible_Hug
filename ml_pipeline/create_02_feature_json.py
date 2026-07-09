import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 02: Feature Extraction (สกัดฟีเจอร์จากข้อมูล WESAD)\n",
    "ขั้นตอนนี้เราจะแปลงข้อมูลดิบระดับมิลลิวินาที (700Hz) ให้กลายเป็นข้อมูลเชิงสถิติ (Features) ทีละ 15 วินาที (Sliding Window)\n",
    "\n",
    "โดยจะสกัดเฉพาะข้อมูลที่ **ตรงกับเซ็นเซอร์บน ESP32 ของคุณ** เท่านั้น ได้แก่:\n",
    "1. **BPM / HR Variability (RMSSD)** - ได้จากการคำนวณคลื่น ECG ให้เหมือนเซ็นเซอร์ MAX30105 ของคุณ\n",
    "2. **EDA (Mean, Std)** - ตรงกับเซ็นเซอร์ GSR ของคุณ\n",
    "3. **Temp (Mean, Slope)** - ตรงกับเซ็นเซอร์ MAX30205 ของคุณ\n",
    "4. **ACC (SVM, Var)** - ตรงกับ MPU6050 ของคุณ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.signal import find_peaks\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "print(\"Libraries loaded!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. สร้างฟังก์ชันคำนวณและสกัดข้อมูล (Feature Extraction Functions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_hr_features(ecg_signal, fs=700):\n",
    "    # 1. หาจุดพีคของคลื่นหัวใจ (R-peaks) \n",
    "    # ระยะห่างขั้นต่ำ 0.4 วินาที (คืออัตราการเต้นสูงสุด 150 BPM)\n",
    "    peaks, _ = find_peaks(ecg_signal, distance=int(0.4*fs), height=np.mean(ecg_signal))\n",
    "    \n",
    "    if len(peaks) < 2:\n",
    "        return 0, 0, 0\n",
    "        \n",
    "    # 2. คำนวณระยะห่างระหว่างพีค (RR intervals) เป็นวินาที\n",
    "    rr_intervals = np.diff(peaks) / fs\n",
    "    \n",
    "    # 3. สกัดฟีเจอร์\n",
    "    mean_hr = 60.0 / np.mean(rr_intervals)      # อัตราการเต้นของหัวใจเฉลี่ย (BPM)\n",
    "    sdnn = np.std(rr_intervals)                 # ความแปรปรวน (SDNN)\n",
    "    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) # RMSSD (ตัวชี้วัดความเครียดที่สำคัญ)\n",
    "    \n",
    "    return mean_hr, sdnn, rmssd\n",
    "\n",
    "def get_eda_features(eda_signal):\n",
    "    mean_eda = np.mean(eda_signal)\n",
    "    std_eda = np.std(eda_signal)\n",
    "    return mean_eda, std_eda\n",
    "\n",
    "def get_temp_features(temp_signal, fs=700):\n",
    "    mean_temp = np.mean(temp_signal)\n",
    "    # หาความชัน (Slope) ของอุณหภูมิในช่วงนี้\n",
    "    x = np.arange(len(temp_signal))\n",
    "    slope = np.polyfit(x, temp_signal, 1)[0] * fs # slope ต่อวินาที\n",
    "    return mean_temp, slope\n",
    "\n",
    "def get_acc_features(acc_signal):\n",
    "    # ACC มาเป็น 3 แกน (x, y, z)\n",
    "    # คำนวณ Signal Vector Magnitude (SVM) รวม 3 แกน\n",
    "    svm = np.sqrt(np.sum(np.square(acc_signal), axis=1))\n",
    "    mean_svm = np.mean(svm)\n",
    "    var_svm = np.var(svm)\n",
    "    return mean_svm, var_svm\n",
    "\n",
    "print(\"Feature extraction functions ready!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. ประมวลผล Sliding Window\n",
    "แบ่งข้อมูลเป็นหน้าต่าง (Window) ขนาด 15 วินาที\n",
    "*เนื่องจากข้อมูลทุก Subject รวมกันมีขนาดใหญ่มาก ในเบื้องต้นเราจะดึงมาแค่ S2, S3, S4 ก่อนเพื่อความรวดเร็วครับ*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fs = 700\n",
    "window_sec = 15\n",
    "window_size = window_sec * fs\n",
    "\n",
    "base_path = 'data/WESAD'\n",
    "subjects = ['S2', 'S3', 'S4'] # เลือกมา 3 คนก่อนเพื่อทำ Prototype อย่างรวดเร็ว\n",
    "\n",
    "features_list = []\n",
    "\n",
    "for subject in subjects:\n",
    "    print(f\"Processing {subject}...\")\n",
    "    file_path = os.path.join(base_path, subject, f\"{subject}.pkl\")\n",
    "    \n",
    "    if not os.path.exists(file_path):\n",
    "        print(f\"File not found: {file_path}\")\n",
    "        continue\n",
    "        \n",
    "    with open(file_path, 'rb') as f:\n",
    "        data = pickle.load(f, encoding='latin1')\n",
    "        \n",
    "    labels = data['label']\n",
    "    chest_data = data['signal']['chest']\n",
    "    \n",
    "    ecg = chest_data['ECG'].flatten()\n",
    "    eda = chest_data['EDA'].flatten()\n",
    "    temp = chest_data['Temp'].flatten()\n",
    "    acc = chest_data['ACC']\n",
    "    \n",
    "    # Sliding window\n",
    "    for start_idx in range(0, len(labels) - window_size, window_size):\n",
    "        end_idx = start_idx + window_size\n",
    "        \n",
    "        # ดึง label ที่มีเยอะที่สุดใน window นี้\n",
    "        window_labels = labels[start_idx:end_idx]\n",
    "        # หาโหมด (Mode) ของ label\n",
    "        values, counts = np.unique(window_labels, return_counts=True)\n",
    "        majority_label = values[np.argmax(counts)]\n",
    "        \n",
    "        # เอาเฉพาะ Baseline (1) และ Stress (2)\n",
    "        # ข้ามข้อมูลที่ไม่มีการแบ่งคลาส (0) หรือตอนอื่นๆ (3,4)\n",
    "        if majority_label not in [1, 2]:\n",
    "            continue\n",
    "            \n",
    "        # --- สกัดฟีเจอร์ ---\n",
    "        mean_hr, sdnn, rmssd = get_hr_features(ecg[start_idx:end_idx], fs)\n",
    "        mean_eda, std_eda = get_eda_features(eda[start_idx:end_idx])\n",
    "        mean_temp, slope_temp = get_temp_features(temp[start_idx:end_idx], fs)\n",
    "        mean_svm, var_svm = get_acc_features(acc[start_idx:end_idx])\n",
    "        \n",
    "        # --- เก็บลง List ---\n",
    "        features_list.append({\n",
    "            'Subject': subject,\n",
    "            'Mean_HR_BPM': mean_hr,\n",
    "            'SDNN': sdnn,\n",
    "            'RMSSD': rmssd,\n",
    "            'Mean_EDA': mean_eda,\n",
    "            'Std_EDA': std_eda,\n",
    "            'Mean_Temp': mean_temp,\n",
    "            'Slope_Temp': slope_temp,\n",
    "            'Mean_ACC_SVM': mean_svm,\n",
    "            'Var_ACC_SVM': var_svm,\n",
    "            'Label': int(majority_label)\n",
    "        })\n",
    "\n",
    "# แปลงเป็น DataFrame และเซฟ\n",
    "df_features = pd.DataFrame(features_list)\n",
    "df_features.to_csv('wesad_extracted_features.csv', index=False)\n",
    "\n",
    "print(\"\\n--- Feature Extraction เสร็จสมบูรณ์! ---\")\n",
    "print(f\"จำนวนแถวข้อมูลทั้งหมด: {len(df_features)} หน้าต่าง (Windows)\")\n",
    "print(df_features.head())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. ขั้นตอนถัดไป\n",
    "ไฟล์ `wesad_extracted_features.csv` ถูกสร้างขึ้นเรียบร้อยแล้ว \n",
    "คุณสามารถเช็คไฟล์ด้านซ้ายมือได้เลยครับ ข้อมูลพร้อมสำหรับเอาไปเทรน AI (Stacking Ensemble) ในขั้นตอนถัดไปแล้ว!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('02_Feature_Extraction.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
