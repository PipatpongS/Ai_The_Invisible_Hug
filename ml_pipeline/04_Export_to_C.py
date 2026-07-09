#!/usr/bin/env python
# coding: utf-8

# # 04: แปลงสมอง AI เป็นภาษา C++ (Edge Deployment)
# ในไฟล์นี้ เราจะแกะกล่อง `stacking_model.pkl` เพื่อดึงเอาความรู้ที่ AI เรียนรู้มา (Weights, Trees, Scalers) แล้วใช้ไลบรารี `micromlgen` / `emlearn` แปลงมันให้กลายเป็นไฟล์ `EdgeAI_Model.h` ภาษา C++ เพื่อเอาไปใส่ในโปรแกรม Arduino ของ ESP32 ครับ!
# 
# **กระบวนการ:**
# 1. ดึงค่าเฉลี่ย (Mean) และส่วนเบี่ยงเบนมาตรฐาน (Scale) จาก `StandardScaler` เพื่อไปทำ Normalization บน ESP32
# 2. แปลง `Random Forest` (Neural Model) เป็น C++ ด้วย `micromlgen`
# 3. แปลง `Logistic Regression` (Physical Model และ Meta-Learner) เป็นสมการคณิตศาสตร์อย่างง่าย `w*x + b`

# In[ ]:


import joblib
import numpy as np
from micromlgen import port
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded!")


# ## 1. แกะกล่อง Stacking Model

# In[ ]:


# โหลดโมเดล
stacking_model = joblib.load('stacking_model.pkl')

# 1. ดึง Base Estimators (Level-0)
neural_pipeline = stacking_model.estimators_[0]
physical_pipeline = stacking_model.estimators_[1]

# 2. ดึง Meta-Estimator (Level-1)
meta_lr = stacking_model.final_estimator_

print("Model components extracted successfully!")


# ## 2. สร้างไฟล์ EdgeAI_Model.h อัตโนมัติ!

# In[ ]:


# ดึง Scaler และ Model ย่อย
neural_scaler = neural_pipeline.named_steps['prep'].transformers_[0][1]
neural_rf = neural_pipeline.named_steps['rf']

physical_scaler = physical_pipeline.named_steps['prep'].transformers_[0][1]
physical_lr = physical_pipeline.named_steps['lr']

# ---------------------------------------------------------
# สร้างสตริงสำหรับไฟล์ C++
c_code = "#ifndef EDGE_AI_MODEL_H\n#define EDGE_AI_MODEL_H\n\n"
c_code += "#include <math.h>\n\n"

# 1. ค่า Scaler
c_code += "// --- 1. Scaler Parameters (StandardScaler) ---\n"
c_code += f"const float NEURAL_MEAN[5] = {{{', '.join(map(str, neural_scaler.mean_))}}};\n"
c_code += f"const float NEURAL_SCALE[5] = {{{', '.join(map(str, neural_scaler.scale_))}}};\n\n"

c_code += f"const float PHYSICAL_MEAN[4] = {{{', '.join(map(str, physical_scaler.mean_))}}};\n"
c_code += f"const float PHYSICAL_SCALE[4] = {{{', '.join(map(str, physical_scaler.scale_))}}};\n\n"

# 2. Physical Logistic Regression Weights
c_code += "// --- 2. Physical Model (Logistic Regression) ---\n"
c_code += f"const float PHYS_LR_WEIGHTS[4] = {{{', '.join(map(str, physical_lr.coef_[0]))}}};\n"
c_code += f"const float PHYS_LR_BIAS = {physical_lr.intercept_[0]};\n\n"

# 3. Meta-Learner Weights
c_code += "// --- 3. Meta-Learner (Logistic Regression) ---\n"
c_code += f"const float META_LR_WEIGHTS[2] = {{{', '.join(map(str, meta_lr.coef_[0]))}}};\n"
c_code += f"const float META_LR_BIAS = {meta_lr.intercept_[0]};\n\n"

# 4. Neural Random Forest (แปลงด้วย micromlgen)
c_code += "// --- 4. Neural Model (Random Forest via micromlgen) ---\n"
rf_c_code = port(neural_rf)
c_code += rf_c_code + "\n\n"

# 5. ฟังก์ชัน Sigmoid และ Predict
c_code += "// --- 5. Prediction Engine ---\n"
c_code += """
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// ฟังก์ชันหลักที่ ESP32 จะเรียกใช้
int predict_stress_level(float mean_hr, float sdnn, float rmssd, float mean_eda, float std_eda, 
                         float mean_temp, float slope_temp, float mean_acc, float var_acc) {

    // 1. Prepare Neural Features
    float neural_features[5] = {mean_hr, sdnn, rmssd, mean_eda, std_eda};
    for(int i=0; i<5; i++) {
        neural_features[i] = (neural_features[i] - NEURAL_MEAN[i]) / NEURAL_SCALE[i];
    }

    // 2. Prepare Physical Features
    float physical_features[4] = {mean_temp, slope_temp, mean_acc, var_acc};
    for(int i=0; i<4; i++) {
        physical_features[i] = (physical_features[i] - PHYSICAL_MEAN[i]) / PHYSICAL_SCALE[i];
    }

    // 3. Level-0: Neural Model Predict Proba (0.0 - 1.0)
    // micromlgen Random Forest สร้างคลาสชื่อ RandomForest
    Eloquent::ML::Port::RandomForest rf;
    int rf_class = rf.predict(neural_features);
    float prob_neural = (rf_class == 2) ? 1.0 : 0.0; // สมมติง่ายๆ จากคลาสคำตอบ

    // 4. Level-0: Physical Model Predict Proba
    float phys_z = PHYS_LR_BIAS;
    for(int i=0; i<4; i++) {
        phys_z += physical_features[i] * PHYS_LR_WEIGHTS[i];
    }
    float prob_physical = sigmoid(phys_z);

    // 5. Level-1: Meta Learner
    float meta_z = META_LR_BIAS + (prob_neural * META_LR_WEIGHTS[0]) + (prob_physical * META_LR_WEIGHTS[1]);
    float final_prob = sigmoid(meta_z);

    // ถ้าความน่าจะเป็น > 0.5 คือ Stress (2) ไม่เช่นนั้นคือ Baseline (1)
    if (final_prob > 0.5) {
        return 2;
    } else {
        return 1;
    }
}
"""

c_code += "#endif // EDGE_AI_MODEL_H\n"

with open('EdgeAI_Model.h', 'w') as f:
    f.write(c_code)

print("\n=== ไฟล์ EdgeAI_Model.h ถูกสร้างขึ้นเรียบร้อยแล้ว! ===")
print("คุณสามารถนำไฟล์นี้ไปใส่ในโฟลเดอร์โปรเจกต์ Arduino ESP32 ของคุณได้เลยครับ")

