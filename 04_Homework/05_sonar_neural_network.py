import os
os.environ['KERAS_BACKEND'] = 'torch'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
import keras
from keras import layers, models

def main():
    print("=== Neural Networks Classification - Sonar Signals (Mines vs. Rocks) ===")
    
    # 1. Paths
    base_dir = Path(".")
    csv_candidates = [
        base_dir / "data" / "sonar.csv",
        base_dir / "data" / "MAL_downloadable materials_session 2" / "Homework" / "sonar.csv"
    ]
    csv_path = next(p for p in csv_candidates if p.exists())
    out_dir = base_dir / "04_Homework" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load dataset
    # Dataset has no header, 60 frequency attributes + 1 class label
    df = pd.read_csv(csv_path, header=None)
    print(f"Loaded sonar dataset with shape: {df.shape}")
    
    X_raw = df.iloc[:, :60].values
    y_raw = df.iloc[:, 60].values
    
    # 3. Transformations:
    # Target (dependent variable): convert categorical 'R' (Rock) -> 0, 'M' (Mine/Metal) -> 1
    y = np.where(y_raw == 'M', 1, 0)
    
    # 4. Train-test split (80:20 ratio, random_state=42 with stratification)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_raw, y, df.index, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Normalization of independent variables (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Baselines for comparison (Logistic Regression, SVM RBF, Random Forest)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    p_lr = lr.predict(X_test_scaled)
    acc_lr = float(accuracy_score(y_test, p_lr))
    f1_lr = float(f1_score(y_test, p_lr))
    
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    p_svm = svm.predict(X_test_scaled)
    acc_svm = float(accuracy_score(y_test, p_svm))
    f1_svm = float(f1_score(y_test, p_svm))
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    p_rf = rf.predict(X_test_scaled)
    acc_rf = float(accuracy_score(y_test, p_rf))
    f1_rf = float(f1_score(y_test, p_rf))
    
    # 7. Build Keras Neural Network model
    keras.utils.set_random_seed(42)
    model = models.Sequential([
        layers.Input(shape=(60,)),
        layers.Dense(32, activation='relu', name="dense_1"),
        layers.Dropout(0.25, name="dropout_1"),
        layers.Dense(16, activation='relu', name="dense_2"),
        layers.Dropout(0.15, name="dropout_2"),
        layers.Dense(1, activation='sigmoid', name="output")
    ], name="Sonar_MLP_Classifier")
    
    # 8. Compile model: Adam optimizer + binary_crossentropy loss + accuracy metric
    learning_rate = 0.001
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 9. Train model with validation split, batch size and epochs
    epochs = 60
    batch_size = 16
    val_split = 0.15
    print(f"Training model: epochs={epochs}, batch_size={batch_size}, validation_split={val_split}...")
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # 10. Predict on test set X_test
    y_prob = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_prob >= 0.50).astype(int)
    
    # 11. Metrics calculation
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_rep_dict = classification_report(y_test, y_pred, target_names=["Skála (Rock)", "Mina (Metal)"], output_dict=True)
    clf_rep_text = classification_report(y_test, y_pred, target_names=["Skála (Rock)", "Mina (Metal)"])
    
    print("\n--- Test Set Classification Report ---")
    print(clf_rep_text)
    print(f"Accuracy : {acc:.4f} ({acc*100:.1f} %)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    
    # 12. Curves (ROC & Precision-Recall)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
    
    # 13. Training history curves
    hist_dict = {
        "epochs": list(range(1, epochs + 1)),
        "loss": [round(float(v), 4) for v in history.history["loss"]],
        "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
        "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
        "val_accuracy": [round(float(v), 4) for v in history.history["val_accuracy"]]
    }
    
    # 14. Sample test predictions
    test_df = df.loc[idx_test].copy()
    test_df["actual"] = y_test
    test_df["predicted"] = y_pred
    test_df["prob_mine"] = np.round(y_prob, 4)
    test_df["is_correct"] = test_df["actual"] == test_df["predicted"]
    
    samples = []
    for _, r in test_df.head(25).iterrows():
        # First 5 frequency values as acoustic preview
        freq_preview = [round(float(r[col]), 3) for col in range(5)]
        samples.append({
            "freq_preview": freq_preview,
            "actual": "Kovová mina (M)" if r["actual"] == 1 else "Skála (R)",
            "predicted": "Kovová mina (M)" if r["predicted"] == 1 else "Skála (R)",
            "prob_mine": float(r["prob_mine"]),
            "is_correct": bool(r["is_correct"])
        })
        
    # 15. Representative signal profiles (Average Rock vs Average Mine profile across 60 bands)
    rock_profile = [round(float(v), 4) for v in df[df[60] == 'R'].iloc[:, :60].mean().values]
    mine_profile = [round(float(v), 4) for v in df[df[60] == 'M'].iloc[:, :60].mean().values]
    
    # 16. Save precomputed results and Keras model
    results = {
        "metadata": {
            "dataset": "sonar.csv",
            "source": "UCI Machine Learning Repository (Connectionist Bench)",
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_count": 60,
            "target_distribution": {
                "mines_metal": int((y == 1).sum()),
                "rocks": int((y == 0).sum()),
                "mine_ratio": round(float(y.mean()), 4)
            }
        },
        "network_architecture": {
            "input_dim": 60,
            "layers": [
                {"name": "Input", "type": "InputLayer", "shape": [60]},
                {"name": "Dense 1", "type": "Dense", "units": 32, "activation": "relu", "params": 60 * 32 + 32},
                {"name": "Dropout 1", "type": "Dropout", "rate": 0.25},
                {"name": "Dense 2", "type": "Dense", "units": 16, "activation": "relu", "params": 32 * 16 + 16},
                {"name": "Dropout 2", "type": "Dropout", "rate": 0.15},
                {"name": "Output", "type": "Dense", "units": 1, "activation": "sigmoid", "params": 16 * 1 + 1}
            ],
            "total_params": int(model.count_params()),
            "optimizer": f"Adam (lr={learning_rate})",
            "loss_function": "binary_crossentropy",
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": val_split
        },
        "baselines": {
            "logistic_regression": {"accuracy": round(acc_lr, 4), "f1": round(f1_lr, 4)},
            "svm_rbf": {"accuracy": round(acc_svm, 4), "f1": round(f1_svm, 4)},
            "random_forest": {"accuracy": round(acc_rf, 4), "f1": round(f1_rf, 4)},
            "keras_mlp": {"accuracy": round(acc, 4), "f1": round(f1, 4)}
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "classification_report": clf_rep_dict,
            "classification_report_text": clf_rep_text
        },
        "training_history": hist_dict,
        "acoustic_profiles": {
            "frequencies": list(range(1, 61)),
            "average_rock": rock_profile,
            "average_mine": mine_profile
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 50)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 50)]]
        },
        "precision_recall_curve": {
            "precision": [round(float(x), 4) for x in p_curve[::max(1, len(p_curve) // 50)]],
            "recall": [round(float(x), 4) for x in r_curve[::max(1, len(r_curve) // 50)]]
        },
        "sample_predictions": samples
    }
    
    with open(out_dir / "sonar_nn_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved precomputed results to {out_dir / 'sonar_nn_precomputed.json'}")
    
    # Save model weights and scaler
    model.save(out_dir / "sonar_nn_model.keras")
    print(f"Saved Keras model to {out_dir / 'sonar_nn_model.keras'}")

if __name__ == "__main__":
    main()
