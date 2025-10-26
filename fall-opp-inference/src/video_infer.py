# -*- coding: utf-8 -*-
"""
video_infer.py — Fall OPP inference (video -> annotated output + CSVs)

Requisitos:
  - ultralytics==8.*
  - opencv-python
  - numpy, pandas
  - xgboost (2.x)
Ejecución ejemplo:
  python src/video_infer.py \
    --source data/chute01/cam1.avi \
    --models_dir models \
    --pose_weights yolo11l-pose.pt \
    --imgsz 960 --conf 0.20 --fps_proc 15 \
    --window 15 --min_votes 5 --cooldown 75 \
    --save_video --save_csv \
    --outdir outputs \
    --mirror_dirs \
    --run_id tunedV2 \
    --timestamp
"""
import os, sys, json, time, math, argparse, collections
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix
from ultralytics import YOLO

# Importa ingeniería de features desde tu script existente
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
try:
    from extract_pose_features import (
        extract_pose_features_from_kps  # debe devolver dict con tus features
    )
except Exception as e:
    raise ImportError(
        "No se pudo importar extract_pose_features.extract_pose_features_from_kps. "
        "Asegúrate de que 'src/extract_pose_features.py' exista y exporte esa función."
    ) from e


# -------------------------
# Utilidades de dibujo / matemáticas
# -------------------------
def draw_bbox(img, xyxy, color=(0, 255, 0), th=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)


def put_text(img, text, org, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, xx2 - xx1), max(0, yy2 - yy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-6)


class SimpleTracker:
    """Asociación por IoU máx; IDs persistentes con TTL."""
    def __init__(self, iou_th=0.2, ttl=30):
        self.iou_th = iou_th
        self.ttl = ttl
        self.next_id = 1
        self.tracks = {}  # id -> dict(bbox, kps, ttl)

    def update(self, dets_xyxy, dets_kps):
        assigned = {}
        used = set()

        # Actualizar / asociar tracks existentes
        for tid, t in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, bb in enumerate(dets_xyxy):
                if j in used:
                    continue
                i = iou(t["bbox"], bb)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0 and best_iou >= self.iou_th:
                self.tracks[tid] = {"bbox": dets_xyxy[best_j], "kps": dets_kps[best_j], "ttl": self.ttl}
                assigned[tid] = best_j
                used.add(best_j)
            else:
                # decrementa ttl y borra si expira
                self.tracks[tid]["ttl"] -= 1
                if self.tracks[tid]["ttl"] <= 0:
                    del self.tracks[tid]

        # Crear nuevos tracks para no asignados
        for j, bb in enumerate(dets_xyxy):
            if j in used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": bb, "kps": dets_kps[j], "ttl": self.ttl}
            assigned[tid] = j

        # Devolver lista (tid, bbox, kps)
        out = []
        for tid, j in assigned.items():
            out.append((tid, self.tracks[tid]["bbox"], self.tracks[tid]["kps"]))
        return out


def temporal_vote_update(state, pid, pred, window=15, min_votes=5, cooldown=75):
    """
    state: dict pid -> {"dq": deque(maxlen=window), "cool": int}
    Pred: 0/1 del clasificador por frame para ese pid.
    """
    if pid not in state:
        state[pid] = {"dq": collections.deque(maxlen=window), "cool": 0}
    st = state[pid]
    if st["cool"] > 0:
        st["cool"] -= 1
    st["dq"].append(int(pred))
    votes = sum(st["dq"])
    fired = (st["cool"] == 0) and (votes >= min_votes)
    if fired:
        st["cool"] = cooldown
    return fired, votes, st["cool"]


# -------------------------
# IO de modelo / salidas
# -------------------------
def load_model_and_meta(models_dir):
    models_dir = Path(models_dir)
    booster = Booster()
    booster.load_model(str(models_dir / "posture_clf_xgb.json"))
    with open(models_dir / "posture_clf_meta.json", "r") as f:
        meta = json.load(f)
    features = list(meta["features_used"])
    thr = float(meta["threshold"])
    return booster, features, thr


def init_outputs(out_root, video_path, *, run_id=None, mirror_dirs=False, use_timestamp=False):
    """
    Crea rutas de salida con:
      - parent en el nombre (p.ej. chute01_cam1)
      - mirror_dirs=True -> subcarpetas por parent (chuteXX)
      - run_id -> sufijo manual
      - timestamp -> sufijo yyyyMMdd-HHMMSS
    """
    out_root = Path(out_root)
    vp = Path(video_path)
    parent = vp.parent.name or "root"
    stem = vp.stem

    # Construir nombre base
    base = f"{parent}_{stem}"
    if run_id:
        base = f"{base}_{run_id}"
    if use_timestamp:
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{base}_{ts}"

    # Directorios
    if mirror_dirs:
        ann_dir = out_root / "ann_video" / parent
        csv_dir = out_root / "scores_csv" / parent
        evt_dir = out_root / "events" / parent
    else:
        ann_dir = out_root / "ann_video"
        csv_dir = out_root / "scores_csv"
        evt_dir = out_root / "events"

    ann_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    evt_dir.mkdir(parents=True, exist_ok=True)

    ann_path = ann_dir / f"{base}_annot.mp4"
    csv_path = csv_dir / f"{base}.scored.csv"
    evt_path = evt_dir / f"{base}_events.csv"
    return ann_path, csv_path, evt_path


# -------------------------
# Bucle principal
# -------------------------
def main():
    ap = argparse.ArgumentParser("Fall OPP Inference")
    ap.add_argument("--source", required=True, help="Ruta al video (p.ej. data/chute01/cam1.avi)")
    ap.add_argument("--models_dir", default="models", help="Carpeta con posture_clf_xgb.json & meta.json")
    ap.add_argument("--pose_weights", default="yolo11l-pose.pt", help="Pesos YOLO Pose (Ultralytics)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--fps_proc", type=int, default=15, help="FPS objetivo (frame skipping)")
    ap.add_argument("--window", type=int, default=15, help="Ventana de votos")
    ap.add_argument("--min_votes", type=int, default=5, help="Mínimo de votos para alerta")
    ap.add_argument("--cooldown", type=int, default=75, help="Frames para enfriar alertas por ID")
    ap.add_argument("--outdir", default="outputs", help="Directorio raíz de salidas")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--mirror_dirs", action="store_true", help="Reflejar estructura por chute en outputs")
    ap.add_argument("--run_id", default="", help="Sufijo opcional para evitar colisiones")
    ap.add_argument("--timestamp", action="store_true", help="Añade sufijo de timestamp a los outputs")
    args = ap.parse_args()

    # Cargar clasificador y meta
    booster, features, thr = load_model_and_meta(args.models_dir)

    # YOLO Pose (CPU por bug MPS en modelos Pose)
    print("⚙️  Ejecutando YOLO Pose en CPU (MPS Pose bug)")
    pose = YOLO(args.pose_weights)
    pose.fuse()

    # Video IO
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERR] No se pudo abrir: {args.source}")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(1, args.fps_proc))))

    ann_path, csv_path, evt_path = init_outputs(
        args.outdir, args.source,
        run_id=args.run_id or None,
        mirror_dirs=args.mirror_dirs,
        use_timestamp=args.timestamp
    )

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(ann_path), fourcc, float(args.fps_proc), (W, H))

    # Estado
    tracker = SimpleTracker(iou_th=0.2, ttl=30)
    vote_state = {}
    records = []  # por frame/persona
    events = []   # eventos de alerta

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Frame skipping para alcanzar fps_proc
        if frame_idx % step != 0:
            continue

        # Pose inference (Ultralytics)
        res = pose.predict(frame, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
        det_boxes = res.boxes.xyxy.cpu().numpy() if getattr(res, "boxes", None) is not None else np.empty((0, 4))
        det_kps = res.keypoints.data.cpu().numpy() if getattr(res, "keypoints", None) is not None else np.empty((0, 17, 3))

        # Tracking
        tracks = tracker.update(det_boxes, det_kps)

        # Clasificación por track
        frame_rows = []
        for pid, bb, kps in tracks:
            proba, pred, fired, votes = 0.0, 0, 0, 0
            # Features desde keypoints
            valid_kps = kps is not None and isinstance(kps, np.ndarray) and kps.shape == (17, 3)
            feats = extract_pose_features_from_kps(kps) if valid_kps else None

            # Reglas de confianza mínima
            enough_kp = feats and (feats.get("num_kp_vis", 0) >= 8) and (feats.get("mean_vis", 0.0) >= 0.12)

            if enough_kp:
                # Construir fila con EXACTAMENTE las features usadas por el modelo
                row = {k: np.nan for k in features}
                for k in features:
                    if k in feats:
                        row[k] = feats[k]
                    elif k == "aspect_bbox":
                        x1, y1, x2, y2 = bb
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        row[k] = float(w / h)
                    elif k == "area_bbox":
                        x1, y1, x2, y2 = bb
                        row[k] = float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))

                X = pd.DataFrame([row], columns=features)
                dmat = DMatrix(X, feature_names=features)
                proba = float(booster.predict(dmat)[0])
                pred = int(proba >= thr)
            else:
                proba, pred = 0.0, 0  # sin confianza -> no alertar

            # Voto temporal + cooldown
            fired, votes, cool = temporal_vote_update(vote_state, pid, pred, args.window, args.min_votes, args.cooldown)

            # Dibujo
            color = (0, 0, 255) if fired else ((0, 165, 255) if pred else (0, 255, 0))
            draw_bbox(frame, bb, color, 2)
            txt = f"id:{pid} p:{proba:.2f} v:{votes}/{args.window}"
            put_text(frame, txt, (int(bb[0]), max(20, int(bb[1]) - 8)), color)

            # Registro por frame
            frame_rows.append({
                "frame": frame_idx,
                "pid": pid,
                "proba": proba,
                "pred": pred,
                "alert": int(fired),
                "x1": float(bb[0]), "y1": float(bb[1]), "x2": float(bb[2]), "y2": float(bb[3])
            })

            # Evento
            if fired:
                ts = frame_idx / float(args.fps_proc)
                events.append({"frame": frame_idx, "sec": ts, "pid": pid, "proba": proba})

        # Flush de registros
        records.extend(frame_rows)

        # Video anotado
        if writer is not None:
            writer.write(frame)

        # Visual live opcional:
        # cv2.imshow("fall-opp inference", frame)
        # if (cv2.waitKey(1) & 0xFF) == 27:
        #     break

    cap.release()
    if writer is not None:
        writer.release()
    # cv2.destroyAllWindows()

    # Guardar CSVs
    if args.save_csv:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(csv_path, index=False)
        Path(evt_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(events).to_csv(evt_path, index=False)

    print("[DONE]")
    if args.save_video:
        print(" video :", ann_path)
    if args.save_csv:
        print(" scores:", csv_path)
        print(" events:", evt_path)


if __name__ == "__main__":
    main()
