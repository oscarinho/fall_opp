# Guardian Eye — Fall Detection via Pose-Based Posture Classification

## 📘 Overview

**Guardian Eye (fall_opp)** is a hybrid fall detection system that integrates:

- **YOLO Pose Large** for spatial keypoint estimation, and  
- **a Gradient-Boosted Posture Classifier (XGBoost)** with temporal reasoning for robust fall identification.

The system analyzes pose geometry (angles, proportions, dispersion) and uses a temporal voting mechanism to filter transient actions (bending, sitting) while maintaining high recall for real falls.

---

## 🧩 Core Components

| Module | Description |
|:-------|:-------------|
| `extract_pose_features.py` | Extracts geometric and anatomical features from YOLO Pose keypoints (per-frame). |
| `annotate_stickman_samples.py` | Visualizes skeletons and annotated poses for debugging and dataset inspection. |
| `video_infer.py` | Full inference pipeline (video → annotated output + CSV logs). |
| `run_all.sh` | Batch processor for all chutes and cameras in `data/`. |
| `models/` | Contains trained posture classifier (`posture_clf_xgb.json` + `posture_clf_meta.json`). |
| `data/` | Multi-camera fall dataset (`chute01–23` with `cam1–cam8`). |
| `outputs/` | Generated videos, frame-level scores, and fall event logs. |

---


## ⚙️ Installation

```bash
# Create environment
conda create -n fall_opp python=3.10 -y
conda activate fall_opp

# Install dependencies
pip install ultralytics==8.* opencv-python numpy pandas xgboost
```

💡 *On Apple Silicon (M-series), YOLO Pose runs on CPU due to an MPS bug in pose models.*

---

## 🚀 Training Summary

**Classifier:** XGBoost 2.x (binary:logistic)  
**Feature set:** 25 pose-derived geometric and anatomical features  
**Validation:** 5-fold stratified cross-validation + early stopping

### Exported artifacts
- `posture_clf_xgb.json`
- `posture_clf_meta.json`

### Final model (as exported)

| Parameter | Value |
|:-----------|:-------|
| Best iteration | 971 |
| Best AUC(valid) | 0.9622 |
| Threshold (recall≥0.90) | 0.4898 |

### Cross-validation results

| Metric | Logistic Regression | XGBoost |
|:-------|:--------------------|:---------|
| Accuracy | 0.844 ± 0.011 | 0.912 ± 0.005 |
| Precision (Fall) | 0.847 ± 0.018 | 0.915 ± 0.010 |
| Recall (Fall) | 0.812 ± 0.017 | 0.894 ± 0.002 |
| F1 (Fall) | 0.829 ± 0.011 | 0.904 ± 0.005 |
| AUC | 0.916 ± 0.010 | 0.963 ± 0.007 |

---

## 🎥 Video Inference Pipeline

### Single video inference

```bash
python src/video_infer.py   --source data/chute02/cam1.avi   --models_dir models   --pose_weights yolo11l-pose.pt   --imgsz 960 --conf 0.20 --fps_proc 15   --window 15 --min_votes 5 --cooldown 75   --save_video --save_csv   --outdir outputs   --mirror_dirs   --run_id tunedV2   --timestamp
```

**Outputs:**

- Annotated video → `outputs/ann_video/chute02/cam1_tunedV2_annot.mp4`
- Scores CSV → `outputs/scores_csv/chute02/cam1_tunedV2.scored.csv`
- Event log → `outputs/events/chute02/cam1_tunedV2_events.csv`

### Batch processing (all videos)

```bash
chmod +x run_all.sh
./run_all.sh
```

---

## 🧠 Temporal Reasoning Logic

The posture classifier runs per frame, and temporal coherence is introduced via a voting + cooldown mechanism:

```python
if votes_in_window >= min_votes and cooldown == 0:
    fall_alert = True
```

**Default parameters:**

- Window = 15 frames  
- Min votes = 5  
- Cooldown = 75 frames

This smooths out transient false detections and stabilizes fall alerts.

---

## 📊 Results & Observations

| Observation | Description |
|:-------------|:-------------|
| ✅ High recall & AUC | AUC(valid)=0.9622, Recall(Fall)=0.894 |
| ⚠ Occasional false positives | Occur during bending/sitting; mitigated by temporal logic |
| 🧭 Temporal voting | Reduces false alarms ≈ 35 % vs per-frame classification |
| 🔒 Confidence gating | Ignores low-pose-quality frames (`num_kp_vis < 8`, `mean_vis < 0.12`) |
| ⚙️ Cooldown per ID | Prevents repeated alerts for the same fall |
| 📄 Auto logging | All confirmed events logged with timestamp & probability |

---

## 🧬 Future Work

- Integrate Temporal CNN/LSTM to learn motion continuity (`angle_deg`, `center_y_rel`, vertical velocity).  
- Combine YOLO fall detections with pose-based classification for hybrid decision-making.  
- Calibrate per-camera thresholds dynamically based on environment statistics.  
- Deploy with ONNX/TensorRT for real-time edge inference.

---

## 🧾 References

- [Ultralytics YOLOv8 Pose Docs](https://docs.ultralytics.com/tasks/pose/)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- Guardian Eye — Temporal CNN Approach (Annex 11)  
- Multiple Cameras Fall Dataset (used for chute01–23)

---

## 🧑‍💻 Authors

**Guardian Eye Team 2**  
[@oscarinho](https://github.com/oscarinho) — *Lead Developer, ML/AI Design*  

Contributors: AI, Data, and Systems Integration students (MNA)

---

## 🏁 License

This repository is released under the **MIT License**.  
Use and modification are allowed with proper citation of the Guardian Eye Fall Detection System.

---

## 📦 Citation

> Ponce, O. (2025). *Guardian Eye: Pose-Based Fall Detection via Temporal Posture Classification (v1.0).*  
> GitHub Repository: [https://github.com/oscarinho/fall_opp](https://github.com/oscarinho/fall_opp)
