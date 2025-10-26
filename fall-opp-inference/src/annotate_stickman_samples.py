import os
import argparse
import random
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# COCO-17 connections (pares de puntos para líneas)
COCO_EDGES = [
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8,10),    # right arm
    (11,12),           # hips
    (5,11), (6,12),    # torso links
    (11,13), (13,15),  # left leg
    (12,14), (14,16),  # right leg
    (0,5), (0,6),      # nose to shoulders
    (0,11), (0,12),    # nose to hips (optional)
]

def denorm_xyxy(row):
    W = int(row.get("img_W", 0)); H = int(row.get("img_H", 0))
    if W <= 0 or H <= 0: return None
    cx, cy, w, h = row["x_center"], row["y_center"], row["width"], row["height"]
    bx = cx * W; by = cy * H; bw = w * W; bh = h * H
    x1 = int(max(0, bx - bw/2)); y1 = int(max(0, by - bh/2))
    x2 = int(min(W-1, bx + bw/2)); y2 = int(min(H-1, by + bh/2))
    return (x1, y1, x2, y2)

def put_label_card(img, x, y, lines, color=(0,255,0), scale=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 6
    # calcular tamaño del cuadro
    text_w = 0; text_h_total = 0; line_hs = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        text_w = max(text_w, tw)
        line_hs.append(th); text_h_total += th + 4
    box_w = text_w + pad*2; box_h = text_h_total + pad*2

    x0 = max(0, x); y0 = max(0, y - box_h - 4)
    x1 = min(img.shape[1]-1, x0 + box_w); y1 = min(img.shape[0]-1, y0 + box_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,0,0), -1)
    alpha = 0.5
    img[y0:y1, x0:x1] = cv2.addWeighted(overlay[y0:y1, x0:x1], alpha, img[y0:y1, x0:x1], 1-alpha, 0)

    yy = y0 + pad + line_hs[0]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x0+pad, yy), font, scale, color, thickness, cv2.LINE_AA)
        if i+1 < len(line_hs): yy += line_hs[i+1] + 4

def fmt(v, nd=2, dash="—"):
    try:
        if pd.isna(v): return dash
        if isinstance(v, (int, np.integer)): return f"{int(v)}"
        return f"{float(v):.{nd}f}"
    except Exception:
        return dash

def draw_stickman(img, kps_xyv, color=(255,255,0), kpt_radius=3, thickness=2, vis_th=0.05):
    """Dibuja keypoints y segmentos COCO-17 si visibilidad > vis_th"""
    xy = kps_xyv[:, :2]; v = kps_xyv[:, 2]
    # puntos
    for i, (x, y) in enumerate(xy):
        if v[i] > vis_th:
            cv2.circle(img, (int(x), int(y)), kpt_radius, color, -1)
    # líneas
    for a, b in COCO_EDGES:
        if v[a] > vis_th and v[b] > vis_th:
            pa = (int(xy[a,0]), int(xy[a,1]))
            pb = (int(xy[b,0]), int(xy[b,1]))
            cv2.line(img, pa, pb, color, thickness)

def best_pose_for_bbox(det_boxes_xyxy, det_kps, gt_xyxy, min_iou=0.05):
    """Empareja un bbox del CSV con la detección de pose más cercana por IoU"""
    best_iou, best_idx = 0.0, -1
    x1g,y1g,x2g,y2g = gt_xyxy
    area_gt = max(0.0, (x2g-x1g)) * max(0.0, (y2g-y1g))
    for j, db in enumerate(det_boxes_xyxy):
        x1 = max(x1g, db[0]); y1 = max(y1g, db[1])
        x2 = min(x2g, db[2]); y2 = min(y2g, db[3])
        iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
        inter = iw*ih
        area_db = max(0.0, (db[2]-db[0])) * max(0.0, (db[3]-db[1]))
        union = area_gt + area_db - inter + 1e-6
        iou = inter/union if union>0 else 0.0
        if iou > best_iou:
            best_iou, best_idx = iou, j
    return (best_idx if best_iou >= min_iou else -1), best_iou

def annotate_sample(row, model, outdir, imgsz=640, conf=0.25):
    img_path = row["image"]
    if not os.path.exists(img_path): return False, f"missing image: {img_path}"
    img = cv2.imread(img_path)
    if img is None: return False, f"cv2.imread failed: {img_path}"

    # Re-ejecutar pose solo para esta imagen
    res = model.predict(img, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    det_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
    det_kps   = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.empty((0,17,3))

    # BBox del CSV (para colorear y tarjetita)
    xyxy = denorm_xyxy(row)
    class_name = str(row.get("class_name", ""))
    has_pose_csv = not pd.isna(row.get("angle_deg", np.nan))

    if class_name.lower() == "fall":
        box_color = (0,0,255)
    else:
        box_color = (0,255,0)

    # Emparejar con la mejor pose cerca del bbox (si hay)
    kps_sel = None
    if xyxy is not None and len(det_boxes) > 0:
        idx, iou = best_pose_for_bbox(det_boxes, det_kps, np.array(xyxy), min_iou=0.05)
        if idx >= 0:
            kps_sel = det_kps[idx]

    # Dibujo
    if xyxy is not None:
        x1,y1,x2,y2 = xyxy
        cv2.rectangle(img, (x1,y1), (x2,y2), box_color, 2)

        # Tarjeta con features del CSV (lo que “sacaste” realmente)
        lines = [
            f"class: {class_name}",
            f"ang: {fmt(row.get('angle_deg'))}° | asp_pose: {fmt(row.get('aspect_pose'))}",
            f"asp_bbox: {fmt(row.get('aspect_bbox'))}",
            f"kp_vis: {fmt(row.get('num_kp_vis'),0)}/{fmt(row.get('num_kp'),0)} | vis_mean: {fmt(row.get('mean_vis'))}",
        ]
        put_label_card(img, x1, y1, lines, color=box_color, scale=0.6, thickness=2)

    # Stickman (si lo logramos emparejar)
    if kps_sel is not None and kps_sel.size == (17*3):
        draw_stickman(img, kps_sel, color=(255,255,0), kpt_radius=3, thickness=2, vis_th=0.05)

    # Guardar
    split = str(row.get("split", "unknown"))
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_split_dir = os.path.join(outdir, split)
    os.makedirs(out_split_dir, exist_ok=True)
    out_path = os.path.join(out_split_dir, f"{base}_stick.jpg")
    ok = cv2.imwrite(out_path, img)
    return ok, out_path if ok else "imwrite failed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pose_features.csv")
    ap.add_argument("--model", default="yolo11l-pose.pt", help="YOLO pose weight for re-annotation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--outdir", default="annots_stick")
    ap.add_argument("--split", default=None, help="filter: train/valid/test")
    ap.add_argument("--class_name", default=None, help="filter: fall/normal")
    ap.add_argument("--sample", type=int, default=25, help="random sample size")
    ap.add_argument("--only_missing_pose", action="store_true", help="only rows with NaN angle_deg in CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # filtros
    if args.split:
        df = df[df["split"] == args.split]
    if args.class_name:
        df = df[df["class_name"].str.lower() == args.class_name.lower()]
    if args.only_missing_pose:
        df = df[df["angle_deg"].isna()]

    # muestreo
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    print(f"[INFO] Selected rows: {len(df)}")

    # Carga modelo pose (CPU por bug MPS en pose)
    model = YOLO(args.model)
    model.fuse()

    ok, fail = 0, 0
    for _, row in df.iterrows():
        success, msg = annotate_sample(row, model, args.outdir, imgsz=args.imgsz, conf=args.conf)
        if success:
            ok += 1
            print("[OK] ", msg)
        else:
            fail += 1
            print("[WARN]", msg)
    print(f"[DONE] saved={ok}, failed={fail}, outdir={args.outdir}")

if __name__ == "__main__":
    main()
