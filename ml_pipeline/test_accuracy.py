import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

def get_hr_features(ecg_signal, fs=700):
    peaks, _ = find_peaks(ecg_signal, distance=int(0.4*fs), height=np.mean(ecg_signal))
    if len(peaks) < 2: return 0, 0, 0
    rr = np.diff(peaks) / fs
    mean_hr = 60.0 / np.mean(rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if len(rr) > 1 else 0
    return mean_hr, sdnn, rmssd

def get_eda_features(eda_signal): return np.mean(eda_signal), np.std(eda_signal)
def get_temp_features(temp_signal, fs=700): 
    return np.mean(temp_signal), np.polyfit(np.arange(len(temp_signal)), temp_signal, 1)[0]*fs
def get_acc_features(acc_signal): 
    svm = np.sqrt(np.sum(np.square(acc_signal), axis=1))
    return np.mean(svm), np.var(svm)

fs = 700
window_size = 15 * fs
subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
base_path = 'data/WESAD'

features_list = []
for subject in subjects:
    file_path = os.path.join(base_path, subject, f"{subject}.pkl")
    if not os.path.exists(file_path): continue
    
    with open(file_path, 'rb') as f: data = pickle.load(f, encoding='latin1')
    labels = data['label']
    ecg = data['signal']['chest']['ECG'].flatten()
    eda = data['signal']['chest']['EDA'].flatten()
    temp = data['signal']['chest']['Temp'].flatten()
    acc = data['signal']['chest']['ACC']
    
    subject_features = []
    baseline_features = []
    
    for start_idx in range(0, len(labels) - window_size, window_size):
        end_idx = start_idx + window_size
        majority_label = np.bincount(labels[start_idx:end_idx]).argmax()
        if majority_label not in [1, 2]: continue
        
        mean_hr, sdnn, rmssd = get_hr_features(ecg[start_idx:end_idx], fs)
        mean_eda, std_eda = get_eda_features(eda[start_idx:end_idx])
        mean_temp, slope_temp = get_temp_features(temp[start_idx:end_idx], fs)
        mean_svm, var_svm = get_acc_features(acc[start_idx:end_idx])
        
        feat_dict = {
            'Subject': subject, 'Mean_HR_BPM': mean_hr, 'SDNN': sdnn, 'RMSSD': rmssd,
            'Mean_EDA': mean_eda, 'Std_EDA': std_eda, 'Mean_Temp': mean_temp,
            'Slope_Temp': slope_temp, 'Mean_ACC_SVM': mean_svm, 'Var_ACC_SVM': var_svm,
            'Label': int(majority_label)
        }
        subject_features.append(feat_dict)
        if majority_label == 1:
            baseline_features.append(feat_dict)
            
    # Baseline Normalization
    if len(baseline_features) > 0:
        base_df = pd.DataFrame(baseline_features)
        base_means = base_df.mean(numeric_only=True)
        
        for feat in subject_features:
            for k in ['Mean_HR_BPM', 'Mean_EDA', 'Mean_Temp', 'Mean_ACC_SVM']:
                feat[f'{k}_norm'] = feat[k] - base_means[k]
                
    features_list.extend(subject_features)

df = pd.DataFrame(features_list)
df.to_csv('wesad_extracted_features_v2.csv', index=False)

# Train model
neural_features = ['Mean_HR_BPM_norm', 'SDNN', 'RMSSD', 'Mean_EDA_norm', 'Std_EDA']
physical_features = ['Mean_Temp_norm', 'Slope_Temp', 'Mean_ACC_SVM_norm', 'Var_ACC_SVM']

X = df[neural_features + physical_features]
y = df['Label']
groups = df['Subject']

neural_transformer = ColumnTransformer([('n', StandardScaler(), neural_features)], remainder='drop')
physical_transformer = ColumnTransformer([('p', StandardScaler(), physical_features)], remainder='drop')

estimators = [
    ('Neural_RF', Pipeline([('prep', neural_transformer), ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))])),
    ('Physical_LR', Pipeline([('prep', physical_transformer), ('lr', LogisticRegression(random_state=42))]))
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
y_pred = cross_val_predict(stacking_model, X, y, groups=groups, cv=GroupKFold(n_splits=5))

print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
