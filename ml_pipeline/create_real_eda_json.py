import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis (EDA) - ข้อมูลจริง (WESAD Dataset)\n",
    "ใน Notebook นี้ เราจะทำการโหลดข้อมูลของจริงจากโฟลเดอร์ S2 (Subject 2) มาวิเคราะห์กันครับ!\n",
    "เราจะเน้นไปที่เซ็นเซอร์หน้าอก (RespiBAN) เพราะเป็นไปตามที่คุณต้องการในแผนงานครับ"
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
    "import matplotlib.pyplot as plt\n",
    "\n",
    "print(\"Libraries loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. โหลดข้อมูลของ Subject 2 (S2)\n",
    "ไฟล์ข้อมูลถูกเก็บในรูปแบบ Pickle (.pkl) เราจะทำการโหลดและดูโครงสร้างของมันครับ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "file_path = 'data/WESAD/S2/S2.pkl'\n",
    "\n",
    "with open(file_path, 'rb') as f:\n",
    "    # Dataset นี้ถูกเซฟด้วย Python 2 จึงต้องใช้ encoding='latin1'\n",
    "    s2_data = pickle.load(f, encoding='latin1')\n",
    "\n",
    "print(\"Keys in S2 data:\", s2_data.keys())\n",
    "print(\"Keys in RespiBAN (Chest) signal:\", s2_data['signal']['chest'].keys())\n",
    "\n",
    "labels = s2_data['label']\n",
    "chest_eda = s2_data['signal']['chest']['EDA'].flatten()\n",
    "chest_temp = s2_data['signal']['chest']['Temp'].flatten()\n",
    "\n",
    "print(f\"Total data points: {len(labels)}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. พล็อตกราฟสัญญาณ EDA และ Temperature\n",
    "เนื่องจากข้อมูลมีความถี่ 700 Hz (วินาทีละ 700 จุด) รวมแล้วมีหลายล้านจุด เพื่อไม่ให้กราฟช้าเกินไป เราจะดึงมาแสดงผลแค่ทุกๆ 1 วินาที (Downsample) ครับ\n",
    "\n",
    "* **Label 1 (สีเขียว)** = สภาวะปกติ (Baseline)\n",
    "* **Label 2 (สีแดง)** = สภาวะเครียด (Stress)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ลดขนาดข้อมูลโดยดึงมา 1 จุด ทุกๆ 700 จุด (1 Hz)\n",
    "fs = 700\n",
    "downsample_rate = fs\n",
    "\n",
    "time_axis = np.arange(0, len(labels)) / fs\n",
    "\n",
    "time_ds = time_axis[::downsample_rate]\n",
    "eda_ds = chest_eda[::downsample_rate]\n",
    "temp_ds = chest_temp[::downsample_rate]\n",
    "labels_ds = labels[::downsample_rate]\n",
    "\n",
    "plt.figure(figsize=(15, 8))\n",
    "\n",
    "# พล็อตกราฟ EDA\n",
    "ax1 = plt.subplot(2, 1, 1)\n",
    "plt.plot(time_ds / 60, eda_ds, color='blue', label='EDA (Skin Conductance)')\n",
    "plt.ylabel('EDA (uS)')\n",
    "plt.title('Subject 2: EDA and Temperature over Time (Minutes)')\n",
    "\n",
    "# สร้างฟังก์ชันช่วยไฮไลท์พื้นหลัง\n",
    "def highlight_background(ax, time_arr, label_arr):\n",
    "    # หาจุดเปลี่ยนสถานะ\n",
    "    changes = np.where(np.diff(label_arr) != 0)[0]\n",
    "    start_idx = 0\n",
    "    for idx in changes:\n",
    "        end_idx = idx\n",
    "        val = label_arr[start_idx]\n",
    "        if val == 2:\n",
    "            ax.axvspan(time_arr[start_idx]/60, time_arr[end_idx]/60, color='red', alpha=0.2)\n",
    "        elif val == 1:\n",
    "            ax.axvspan(time_arr[start_idx]/60, time_arr[end_idx]/60, color='green', alpha=0.2)\n",
    "        start_idx = end_idx + 1\n",
    "    # ไฮไลท์บล็อกสุดท้าย\n",
    "    val = label_arr[start_idx]\n",
    "    if val == 2:\n",
    "        ax.axvspan(time_arr[start_idx]/60, time_arr[-1]/60, color='red', alpha=0.2)\n",
    "    elif val == 1:\n",
    "        ax.axvspan(time_arr[start_idx]/60, time_arr[-1]/60, color='green', alpha=0.2)\n",
    "\n",
    "highlight_background(ax1, time_ds, labels_ds)\n",
    "plt.legend(loc='upper right')\n",
    "\n",
    "# พล็อตกราฟ Temperature\n",
    "ax2 = plt.subplot(2, 1, 2)\n",
    "plt.plot(time_ds / 60, temp_ds, color='orange', label='Temperature')\n",
    "plt.xlabel('Time (Minutes)')\n",
    "plt.ylabel('Temp (°C)')\n",
    "\n",
    "highlight_background(ax2, time_ds, labels_ds)\n",
    "plt.legend(loc='upper right')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
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

with open('01_EDA_WESAD.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
