import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 03: โมเดล AI (Stacking Ensemble Training)\n",
    "ในขั้นตอนที่ 3 นี้ เราจะนำข้อมูลฟีเจอร์ที่สกัดได้จากไฟล์ `wesad_extracted_features.csv` มาเทรนโมเดล AI กันครับ!\n",
    "\n",
    "ตามสถาปัตยกรรม **\"The Invisible Hug\"** ที่เราออกแบบไว้ โมเดลของเราจะเป็นแบบ **Stacking Ensemble** ประกอบด้วย:\n",
    "1. **Level-0 (Base Learners):**\n",
    "   - **Random Forest:** รับหน้าที่วิเคราะห์ฟีเจอร์ทางระบบประสาทอัตโนมัติ (EDA, HR) เพราะมีความซับซ้อนและไม่เป็นเส้นตรง (Non-linear)\n",
    "   - **Logistic Regression:** รับหน้าที่วิเคราะห์ฟีเจอร์ทางกายภาพ (ACC, Temp) เพื่อแยกแยะว่าการเปลี่ยนแปลงเกิดจากการขยับตัวธรรมดาหรือความเครียด\n",
    "2. **Level-1 (Meta-Learner):**\n",
    "   - **Logistic Regression:** นำคำตอบจาก Level-0 ทั้งสองตัวมาชั่งน้ำหนักและตัดสินใจขั้นสุดท้าย"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import GroupKFold, cross_val_predict\n",
    "from sklearn.ensemble import RandomForestClassifier, StackingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "print(\"Libraries loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. โหลดข้อมูลและแบ่งกลุ่มฟีเจอร์ (Feature Grouping)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# โหลดข้อมูล\n",
    "df = pd.read_csv('wesad_extracted_features.csv')\n",
    "\n",
    "# ฟีเจอร์ระบบประสาท (Neural) สำหรับ Random Forest\n",
    "neural_features = ['Mean_HR_BPM', 'SDNN', 'RMSSD', 'Mean_EDA', 'Std_EDA']\n",
    "\n",
    "# ฟีเจอร์ทางกายภาพ (Physical) สำหรับ Logistic Regression\n",
    "physical_features = ['Mean_Temp', 'Slope_Temp', 'Mean_ACC_SVM', 'Var_ACC_SVM']\n",
    "\n",
    "X = df[neural_features + physical_features]\n",
    "y = df['Label']\n",
    "groups = df['Subject']\n",
    "\n",
    "print(f\"จำนวนข้อมูลทั้งหมด: {len(df)} กรอบเวลา (Windows)\")\n",
    "print(\"จำนวนข้อมูลแต่ละคลาส:\\n\", y.value_counts(sort=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. สร้างโครงสร้างโมเดล Stacking Ensemble\n",
    "เราจะใช้ `ColumnTransformer` เพื่อบังคับให้ Base Learner แต่ละตัวมองเห็นเฉพาะฟีเจอร์ที่ได้รับมอบหมายเท่านั้น"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# สร้างตัวคัดกรองฟีเจอร์และปรับสเกล (Standard Scaler)\n",
    "neural_transformer = ColumnTransformer([\n",
    "    ('select_neural', StandardScaler(), neural_features)\n",
    "], remainder='drop')\n",
    "\n",
    "physical_transformer = ColumnTransformer([\n",
    "    ('select_physical', StandardScaler(), physical_features)\n",
    "], remainder='drop')\n",
    "\n",
    "# สร้าง Base Learners (Level-0)\n",
    "rf_pipeline = Pipeline([\n",
    "    ('prep', neural_transformer),\n",
    "    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))\n",
    "])\n",
    "\n",
    "lr_pipeline = Pipeline([\n",
    "    ('prep', physical_transformer),\n",
    "    ('lr', LogisticRegression(random_state=42))\n",
    "])\n",
    "\n",
    "estimators = [\n",
    "    ('Neural_RF', rf_pipeline),\n",
    "    ('Physical_LR', lr_pipeline)\n",
    "]\n",
    "\n",
    "# สร้าง Meta-Learner (Level-1)\n",
    "stacking_model = StackingClassifier(\n",
    "    estimators=estimators,\n",
    "    final_estimator=LogisticRegression(),\n",
    "    cv=3 # Cross-validation ภายในช่วงเทรน\n",
    ")\n",
    "\n",
    "print(\"Stacking Ensemble Model Architecture created!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. ทดสอบความแม่นยำด้วยวิธี GroupKFold (Cross-Validation)\n",
    "เพื่อป้องกัน AI ทุจริตแอบจำข้อมูลส่วนตัว (Data Leakage) เราจะแบ่งกลุ่มสอบตาม `Subject ID` \n",
    "แปลว่า AI จะไม่เคยเห็นข้อมูลของคนที่กำลังถูกทดสอบมาก่อนเลย! นี่คือการทดสอบในสภาพแวดล้อมโลกความเป็นจริงครับ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# เนื่องจากเราดึงข้อมูลมาแค่ 3 คน (S2, S3, S4) เราจะแบ่ง KFold = 3 ครับ\n",
    "gkf = GroupKFold(n_splits=3)\n",
    "y_pred = cross_val_predict(stacking_model, X, y, groups=groups, cv=gkf)\n",
    "\n",
    "print(\"\\n=== รายงานผลความแม่นยำ (Classification Report) ===\")\n",
    "print(classification_report(y, y_pred, target_names=['Baseline (1)', 'Stress (2)']))\n",
    "\n",
    "# พล็อตกราฟ Confusion Matrix\n",
    "cm = confusion_matrix(y, y_pred)\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=['Baseline', 'Stress'], \n",
    "            yticklabels=['Baseline', 'Stress'])\n",
    "plt.ylabel('ความจริง (True Label)')\n",
    "plt.xlabel('AI ทำนาย (Predicted Label)')\n",
    "plt.title('Confusion Matrix: Stacking Ensemble')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. เทรนโมเดลจริงและเซฟเพื่อเตรียมนำไปใช้บน ESP32\n",
    "ขั้นตอนนี้เราจะสอนมันด้วยข้อมูลทั้งหมดที่มี แล้วเซฟโมเดลออกมาครับ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "stacking_model.fit(X, y)\n",
    "\n",
    "joblib.dump(stacking_model, 'stacking_model.pkl')\n",
    "print(\"\\n--- โมเดลถูกเทรนและเซฟลงไฟล์ 'stacking_model.pkl' เรียบร้อยแล้ว! ---\")\n",
    "print(\"คุณสามารถดูไฟล์โมเดลที่เซฟไว้ได้ที่แถบซ้ายมือเลยครับ\")"
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

with open('03_Model_Training.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
