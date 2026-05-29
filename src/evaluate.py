import os
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import shap
import config
from model import SensorTransformerAutoencoder

def main():
    model_save_path = os.path.join(config.TRAIN_DIR, "transformer_autoencoder.pth")
    scaler_save_path = os.path.join(config.TRAIN_DIR, "transformer_scaler.pkl")
    final_cols_save_path = os.path.join(config.TRAIN_DIR, "final_sensor_cols.pkl")
    imputer_save_path = os.path.join(config.TRAIN_DIR, "imputer.pkl")

    with open(final_cols_save_path, 'rb') as f:
        final_sensor_cols = pickle.load(f)
    with open(scaler_save_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(imputer_save_path, 'rb') as f:
        imputer = pickle.load(f)

    model = SensorTransformerAutoencoder(num_sensors=len(final_sensor_cols)).to(config.DEVICE)
    model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
    model.eval()

    ng_files_to_analyze = {}
    for i in range(1, 10):
        for type_str in ['chg', 'dchg']:
            file_name = f"Test{i:02d}_NG_{type_str}.csv"
            label_name = f"Test{i:02d}_NG_{type_str}_Label.csv"
            data_path = os.path.join(config.TEST_DIR, file_name)
            label_path = os.path.join(config.LABEL_DIR, label_name)
            
            if os.path.exists(data_path) and os.path.exists(label_path):
                ng_files_to_analyze[f"Test{i:02d}_NG_{type_str}"] = {'data': data_path, 'label': label_path}

    bg_df = pd.read_csv(os.path.join(config.TRAIN_DIR, "1000_chg.csv"))
    bg_data = bg_df[final_sensor_cols]
    bg_imputed = imputer.transform(bg_data)
    bg_scaled = scaler.transform(bg_imputed)
    background_data = shap.kmeans(bg_scaled, 10)

    def get_anomaly_score_for_shap(numpy_data):
        tensor_data = torch.FloatTensor(numpy_data).to(config.DEVICE)
        with torch.no_grad():
            reconstructed = model(tensor_data)
        errors = torch.mean((reconstructed - tensor_data)**2, dim=1)
        return errors.cpu().numpy()

    explainer = shap.KernelExplainer(get_anomaly_score_for_shap, background_data)

    all_shap_values = []
    all_shap_data = []

    for file_name, paths in ng_files_to_analyze.items():
        print(f"--- 파일 분석 중: {file_name}.csv ---")
        test_ng_df = pd.read_csv(paths['data'])
        label_df = pd.read_csv(paths['label'])

        test_ng_data_filtered = test_ng_df[final_sensor_cols]
        test_ng_imputed = imputer.transform(test_ng_data_filtered)
        test_ng_scaled = scaler.transform(test_ng_imputed)
        true_labels = label_df['label'].values[:len(test_ng_scaled)]

        anomaly_scores = get_anomaly_score_for_shap(test_ng_scaled)
        precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_f1 = np.max(f1_scores)
        print(f"  -> F1-Score: {best_f1:.4f}")

        top_5_anomaly_indices = np.argsort(anomaly_scores)[-5:]
        samples_to_explain = test_ng_scaled[top_5_anomaly_indices]

        shap_values = explainer.shap_values(samples_to_explain)
        all_shap_values.append(shap_values)
        all_shap_data.append(samples_to_explain)

    final_shap_values = np.concatenate(all_shap_values, axis=0)
    final_shap_data = np.concatenate(all_shap_data, axis=0)

    print("--- 🔬 글로벌 SHAP 요약 플롯 (핵심 원인 센서) 저장 중 ---")
    shap.summary_plot(final_shap_values, final_shap_data, feature_names=final_sensor_cols, max_display=20, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    print("✅ SHAP 요약 플롯이 현재 디렉토리의 'shap_summary.png'로 저장되었습니다.")
    
if __name__ == "__main__":
    main()