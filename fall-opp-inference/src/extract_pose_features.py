# src/extract_pose_features.py
import os, glob, math, argparse, warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

# ---------------------------
# Utilidades
# ---------------------------
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

KP_NAMES_COCO17 = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = \
     0,    5,     6,     7,     8,     9,     10,    11,    12,    13,    14,    15,    16

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_image_path(image_dir: str, stem: str) -> Optional[str]:
    for ext in IMG_EXTS:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def bbox_features_from_yolo(x, y, w, h):
    aspect_bbox = w / (h + 1e-6)
    area_bbox = w * h
    return aspect_bbox, area_bbox

def angle_trunk(pts):
    # Ángulo hombros->caderas respecto a vertical (0 = vertical, 90 = horizontal)
    shoulder_mid = pts[[L_SHO, R_SHO]].mean(axis=0)
    hip_mid      = pts[[L_HIP, R_HIP]].mean(axis=0)
    vec = hip_mid - shoulder_mid  # y hacia abajo en imagen
    ang = math.degrees(math.atan2(abs(vec[0]), abs(vec[1]) + 1e-6))
    return float(ang), shoulder_mid, hip_mid

def limb_angle(pt_up, pt_down) -> float:
    # Ángulo vs vertical (0=vertical, 90=horizontal)
    v = pt_down - pt_up
    return float(math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-6)))

def pose_box_stats(valid_xy):
    xs, ys = valid_xy[:,0], valid_xy[:,1]
    w = float(xs.max() - xs.min() + 1e-6)
    h = float(ys.max() - ys.min() + 1e-6)
    aspect = float(w / h)
    area   = float(w * h)
    var_x  = float(np.var(xs))
    var_y  = float(np.var(ys))
    std_x  = float(np.std(xs))
    std_y  = float(np.std(ys))
    center_y = float(ys.mean())
    return w, h, area, aspect, var_x, var_y, std_x, std_y, center_y

def euclid(a, b):
    return float(np.linalg.norm(a - b))

def extract_pose_features_from_kps(xyv: np.ndarray) -> dict:
    """
    xyv: (17,3) -> x,y,visibility (o conf>0)
    Devuelve features robustas de postura.
    """
    xy = xyv[:, :2]
    v  = xyv[:, 2]
    valid = v > 0
    if valid.sum() < 4:
        return None  # insuficiente

    valid_xy = xy[valid]
    # Caja del esqueleto y dispersión
    pose_w, pose_h, pose_area, aspect_pose, var_x, var_y, std_x, std_y, center_y = pose_box_stats(valid_xy)

    # Tronco
    ang_deg, shoulder_mid, hip_mid = angle_trunk(xy)

    # Anchos clave
    shoulder_width = euclid(xy[L_SHO], xy[R_SHO]) if (v[L_SHO]>0 and v[R_SHO]>0) else np.nan
    hip_width      = euclid(xy[L_HIP], xy[R_HIP]) if (v[L_HIP]>0 and v[R_HIP]>0) else np.nan
    torso_len      = euclid(shoulder_mid, hip_mid)

    # Estimación de largo corporal: nariz a tobillos (promedio izq/der si existen)
    ankles = []
    if v[L_ANK] > 0: ankles.append(xy[L_ANK])
    if v[R_ANK] > 0: ankles.append(xy[R_ANK])
    if len(ankles) > 0 and v[NOSE] > 0:
        body_len_est = float(np.mean([euclid(xy[NOSE], a) for a in ankles]))
    else:
        body_len_est = np.nan

    shoulder_hip_ratio = (shoulder_width / (hip_width + 1e-6)) if (not np.isnan(shoulder_width) and not np.isnan(hip_width)) else np.nan

    # Altura relativa de la cabeza: (nose.y - hip_mid.y)/pose_h  (positiva si nose está debajo)
    head_rel_height = np.nan
    if v[NOSE] > 0:
        head_rel_height = float((xy[NOSE,1] - hip_mid[1]) / (pose_h + 1e-6))

    center_y_rel = float((center_y - hip_mid[1]) / (pose_h + 1e-6))

    # Ángulos de extremidades (vs vertical)
    thigh_angle_L = limb_angle(xy[L_HIP], xy[L_KNE]) if (v[L_HIP]>0 and v[L_KNE]>0) else np.nan
    thigh_angle_R = limb_angle(xy[R_HIP], xy[R_KNE]) if (v[R_HIP]>0 and v[R_KNE]>0) else np.nan
    shank_angle_L = limb_angle(xy[L_KNE], xy[L_ANK]) if (v[L_KNE]>0 and v[L_ANK]>0) else np.nan
    shank_angle_R = limb_angle(xy[R_KNE], xy[R_ANK]) if (v[R_KNE]>0 and v[R_ANK]>0) else np.nan
    upperarm_L    = limb_angle(xy[L_SHO], xy[L_ELB]) if (v[L_SHO]>0 and v[L_ELB]>0) else np.nan
    upperarm_R    = limb_angle(xy[R_SHO], xy[R_ELB]) if (v[R_SHO]>0 and v[R_ELB]>0) else np.nan

    # Conteos de keypoints
    num_kp     = int(len(xy))
    num_kp_vis = int(valid.sum())
    mean_vis   = float(np.mean(v[valid])) if valid.any() else 0.0

    return dict(
        angle_deg=ang_deg,
        pose_w=pose_w, pose_h=pose_h, pose_area=pose_area, aspect_pose=aspect_pose,
        var_x=var_x, var_y=var_y, std_x=std_x, std_y=std_y, center_y=center_y,
        shoulder_width=shoulder_width, hip_width=hip_width, torso_len=torso_len,
        body_len_est=body_len_est, shoulder_hip_ratio=shoulder_hip_ratio,
        head_rel_height=head_rel_height, center_y_rel=center_y_rel,
        thigh_angle_L=thigh_angle_L, thigh_angle_R=thigh_angle_R,
        shank_angle_L=shank_angle_L, shank_angle_R=shank_angle_R,
        upperarm_L=upperarm_L, upperarm_R=upperarm_R,
        num_kp=num_kp, num_kp_vis=num_kp_vis, mean_vis=mean_vis
    )

def run_pose_inference(model: YOLO, image_path: str, conf=0.25, imgsz=640):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.predict(image_path, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
    return res[0]

# ---------------------------
# Lectura etiquetas YOLO + Pose
# ---------------------------
def extract_split(split_dir: str, class_names: List[str], model: YOLO, conf=0.25, imgsz=640):
    rows = []
    label_dir = os.path.join(split_dir, "labels")
    image_dir = os.path.join(split_dir, "images")
    for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = find_image_path(image_dir, stem)
        if img_path is None:
            continue

        # BBoxes (puede haber más de una por imagen)
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        # Pose inference por imagen (una sola vez)
        result = run_pose_inference(model, img_path, conf=conf, imgsz=imgsz)
        kps_all = getattr(result, "keypoints", None)
        boxes   = getattr(result, "boxes", None)

        # Para mapear bbox YOLO->persona detectada por pose, hacemos emparejamiento simple por IoU con result.boxes
        det_boxes_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0,4))
        det_kps = kps_all.data.cpu().numpy() if kps_all is not None else np.empty((0,17,3))

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = map(float, parts)
            cls_id = int(cls_id)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

            # Features de la etiqueta (normalizadas YOLO)
            aspect_bbox, area_bbox = bbox_features_from_yolo(x, y, w, h)

            # Convertimos bbox YOLO (cx,cy,w,h) normalized -> aproximación XYXY pixel si conocemos tamaño
            # Para eso, tomamos la resolución desde result.orig_shape
            H, W = result.orig_shape[:2]
            bx = x * W; by = y * H; bw = w * W; bh = h * H
            x1 = bx - bw/2; y1 = by - bh/2; x2 = bx + bw/2; y2 = by + bh/2
            gt = np.array([x1,y1,x2,y2], dtype=float)

            # Emparejar con la detección de pose más cercana por IoU (si existe)
            best_iou, best_idx = 0.0, -1
            for j, db in enumerate(det_boxes_xyxy):
                xx1 = max(gt[0], db[0]); yy1 = max(gt[1], db[1])
                xx2 = min(gt[2], db[2]); yy2 = min(gt[3], db[3])
                iw = max(0.0, xx2-xx1); ih = max(0.0, yy2-yy1)
                inter = iw*ih
                area_gt = (gt[2]-gt[0])*(gt[3]-gt[1])
                area_db = (db[2]-db[0])*(db[3]-db[1])
                union = area_gt + area_db - inter + 1e-6
                iou = inter/union
                if iou > best_iou:
                    best_iou, best_idx = iou, j

            pose_dict = {}
            if best_idx >= 0 and det_kps.shape[0] > best_idx:
                pose_dict = extract_pose_features_from_kps(det_kps[best_idx])
                if pose_dict is None:
                    pose_dict = {}
            else:
                # No hubo emparejamiento de pose confiable (dejamos NaN en pose features)
                pose_dict = {}

            row = {
                "split": os.path.basename(split_dir),
                "image": img_path,
                "class_id": cls_id,
                "class_name": cls_name,
                "x_center": x, "y_center": y, "width": w, "height": h,
                "aspect_bbox": aspect_bbox, "area_bbox": area_bbox,
                "img_W": W, "img_H": H
            }
            row.update({k: pose_dict.get(k, np.nan) for k in [
                "angle_deg","pose_w","pose_h","pose_area","aspect_pose",
                "var_x","var_y","std_x","std_y","center_y",
                "shoulder_width","hip_width","torso_len","body_len_est",
                "shoulder_hip_ratio","head_rel_height","center_y_rel",
                "thigh_angle_L","thigh_angle_R","shank_angle_L","shank_angle_R",
                "upperarm_L","upperarm_R","num_kp","num_kp_vis","mean_vis"
            ]})
            # placeholder para compatibilidad futura (velocidad/h/s en video)
            row["vy_hps"] = np.nan

            rows.append(row)

    return rows

# ---------------------------
# MAIN
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", default="data.yaml", help="Ruta a data.yaml (Roboflow/YOLO)")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="Modelo Ultralytics pose")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out_csv", default="pose_features.csv")
    args = ap.parse_args()

    data = load_yaml(args.data_yaml)
    class_names = data["names"]

    # Cargar YOLO Pose (CPU — bug MPS para pose en macOS)
    print("Loading YOLO Pose on CPU…")
    pose_model = YOLO(args.model)
    pose_model.fuse()

    all_rows = []
    root = os.path.dirname(args.data_yaml)
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            continue
        print(f"[Split] {split}")
        rows = extract_split(split_dir, class_names, pose_model, conf=args.conf, imgsz=args.imgsz)
        all_rows.extend(rows)
        print(f"  +{len(rows)} rows")

    df = pd.DataFrame(all_rows)

    # Orden sugerido de columnas
    first_cols = [
        "split","image","class_id","class_name","img_W","img_H",
        "x_center","y_center","width","height","aspect_bbox","area_bbox",
        "angle_deg","aspect_pose","pose_w","pose_h","pose_area",
        "var_x","var_y","std_x","std_y","center_y","center_y_rel",
        "shoulder_width","hip_width","shoulder_hip_ratio","torso_len","body_len_est",
        "head_rel_height",
        "thigh_angle_L","thigh_angle_R","shank_angle_L","shank_angle_R",
        "upperarm_L","upperarm_R","num_kp","num_kp_vis","mean_vis",
        "vy_hps"
    ]
    # Añadir cualquier otra columna residual al final
    cols = [c for c in first_cols if c in df.columns] + [c for c in df.columns if c not in first_cols]
    df = df.reindex(columns=cols)

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] CSV generado: {args.out_csv}  (filas: {len(df)}, columnas: {len(df.columns)})")

if __name__ == "__main__":
    main()
