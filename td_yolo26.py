import re
import sys
import site
import colorsys
from typing import Dict, List, Optional

# Some TouchDesigner builds embed a Python that ignores the user site-packages.
# When pip installs without admin rights, packages go to %APPDATA%\Python\Python311\site-packages
# which is NOT in sys.path on those builds. Add it here so imports work either way.
try:
    _user_site = site.getusersitepackages()
    if _user_site and _user_site not in sys.path:
        sys.path.append(_user_site)
except Exception:
    pass

YOLO = None
cv2 = None
np = None

_MODEL_CACHE: Dict[str, "YOLO"] = {}
_TRACK_STATE = {"next": 0, "tracks": {}}
_FRAME_STATE = {"counter": 0, "last_result": None}


def onSetupParameters(scriptOp):
    try:
        scriptOp.destroyCustomPars()
    except Exception:
        pass

    page = scriptOp.appendCustomPage("YOLO26")
    pg_model = page.appendMenu("Model", label="Model")
    p_model = pg_model[0]
    p_model.menuNames = (
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
        "yolo26s-seg.pt",
        "yolo26m-seg.pt",
        "yolo26l-seg.pt",
        "yolo26s-pose.pt",
        "yolo26m-pose.pt",
        "yolo26l-pose.pt",
        "custom",
    )
    p_model.menuLabels = (
        "Nano (det)",
        "Small (det)",
        "Medium (det)",
        "Large (det)",
        "XL (det)",
        "Small (seg)",
        "Medium (seg)",
        "Large (seg)",
        "Small (pose)",
        "Medium (pose)",
        "Large (pose)",
        "Custom path",
    )

    page.appendStr("Modelpath", label="CustomPath")

    page.appendStr("Prompt", label="PromptClasses")

    page.appendFloat("Conf", label="Confidence")

    pg_output = page.appendMenu("Output", label="Output")
    p_output = pg_output[0]
    p_output.menuNames = ("overlay", "mask", "data")
    p_output.menuLabels = ("Overlay (YOLO draw)", "Binary Mask", "Data only (passthrough)")

    pg_tracker = page.appendMenu("Tracker", label="Tracker")
    p_tracker = pg_tracker[0]
    p_tracker.menuNames = ("none", "bytetrack", "botsort")
    p_tracker.menuLabels = ("None", "ByteTrack", "BoT-SORT")

    p_segid = page.appendInt("Segid", label="SegmentationID")[0]
    p_segid.default = -1  # -1 = toutes les segmentations

    p_skip = page.appendInt("Skipframes", label="Skip Frames")[0]
    p_skip.default = 0
    p_skip.min = 0
    p_skip.max = 10
    p_skip.normMin = 0
    p_skip.normMax = 10

    page.appendPulse("Reload", label="ReloadModel")

    p_devstatus = page.appendStr("Devicestatus", label="Device")[0]
    p_devstatus.readOnly = True
    p_devstatus.default = "—"
    return


def onPulse(par):
    if par.name == "Reload":
        _MODEL_CACHE.clear()
        _FRAME_STATE["counter"] = 0
        _FRAME_STATE["last_result"] = None
        return True
    return False


def _normalize_names(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return names
    if isinstance(names, (list, tuple)):
        return {i: n for i, n in enumerate(names)}
    return {}


def _parse_classes(prompt: str, names: Dict[int, str], scriptOp) -> Optional[List[int]]:
    if not prompt:
        return None

    tokens = [t.strip().lower() for t in re.split(r"[;,]+", prompt) if t.strip()]
    if not tokens:
        return None

    name_to_id = {str(v).lower(): k for k, v in names.items()}
    classes: List[int] = []

    for token in tokens:
        if token.isdigit():
            classes.append(int(token))
        elif token in name_to_id:
            classes.append(name_to_id[token])
        else:
            scriptOp.addWarning(f"Classe inconnue ignorée : {token}")

    return classes or None


def _get_model(model_path: str):
    model_path = model_path or "yolo26s.pt"
    model = _MODEL_CACHE.get(model_path)
    if model is None:
        model = YOLO(model_path)
        _MODEL_CACHE[model_path] = model
    return model


def _box_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-9)


def _assign_track_ids(xyxy, clss):
    global _TRACK_STATE
    tracks = _TRACK_STATE["tracks"]
    next_id = _TRACK_STATE["next"]
    track_ids = []
    used = set()

    for i, box in enumerate(xyxy):
        cls = int(clss[i]) if clss is not None else -1
        best_id = None
        best_iou = 0.0
        for tid, tinfo in tracks.items():
            if tinfo["cls"] != cls or tid in used:
                continue
            iou = _box_iou(box, tinfo["box"])
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        if best_id is not None and best_iou >= 0.3:
            track_ids.append(best_id)
            tracks[best_id] = {"cls": cls, "box": box}
            used.add(best_id)
        else:
            tid = next_id
            next_id += 1
            tracks[tid] = {"cls": cls, "box": box}
            track_ids.append(tid)
            used.add(tid)

    _TRACK_STATE["next"] = next_id
    return track_ids


def _build_mask(result, height: int, width: int, scriptOp, seg_id: int, track_ids=None) -> "np.ndarray":
    if result.masks is None:
        scriptOp.addWarning("Pas de masks dans la sortie — utilisez Output=overlay ou un modèle -seg.")
        return np.zeros((height, width, 4), dtype=np.uint8)

    mask = result.masks.data  # torch.Tensor [N, H, W]
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim != 3:
        mask_np = mask_np[None, ...]

    n_masks = mask_np.shape[0]

    if seg_id == -2:
        out = np.zeros((height, width, 4), dtype=np.uint8)
        total = n_masks
        for i in range(n_masks):
            if track_ids is not None and i < len(track_ids):
                color = _color_from_id(int(track_ids[i]))
            else:
                color = _unique_color(i, total)
            m = (mask_np[i] > 0.5)
            if not m.any():
                continue
            out[m] = (*color, 255)
        return out

    if seg_id is not None and seg_id >= 0:
        target_idx = None
        if track_ids is not None:
            tid_list = track_ids.tolist()
            if seg_id in tid_list:
                target_idx = tid_list.index(seg_id)
        if target_idx is None and seg_id < n_masks:
            target_idx = seg_id
        if target_idx is not None and target_idx < n_masks:
            combined = mask_np[target_idx]
        else:
            scriptOp.addWarning(f"Segmentation id {seg_id} inexistant, combinaison totale utilisée.")
            combined = mask_np.sum(axis=0)
    else:
        combined = mask_np.sum(axis=0)

    mask_bin = (combined > 0.5).astype(np.uint8) * 255
    return np.dstack([mask_bin, mask_bin, mask_bin, mask_bin])


def _palette(idx: int) -> tuple:
    colors = [
        (255, 56, 56),
        (56, 255, 56),
        (56, 56, 255),
        (255, 224, 32),
        (0, 255, 255),
        (255, 0, 255),
        (255, 149, 0),
    ]
    return colors[idx % len(colors)]


def _unique_color(idx: int, total: int) -> tuple:
    h = (idx % max(1, total)) / float(max(1, total))
    s = 0.65
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _color_from_id(tid: int) -> tuple:
    h = (tid * 0.61803398875) % 1.0
    s = 0.7
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def onCook(scriptOp):
    global YOLO, cv2, np

    if YOLO is None:
        try:
            from ultralytics import YOLO as _YOLO
            YOLO = _YOLO
        except Exception as e:
            scriptOp.addError(f"ultralytics manquant ou invalide : {e}")
            return

    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except Exception as e:
            scriptOp.addError(f"opencv-python manquant : {e}")
            return

    if np is None:
        try:
            import numpy as _np
            np = _np
        except Exception as e:
            scriptOp.addError(f"numpy manquant : {e}")
            return
    if not scriptOp.inputs:
        scriptOp.addWarning("Connectez un TOP en entrée.")
        return

    in_top = scriptOp.inputs[0]
    frame = in_top.numpyArray()
    if frame is None:
        scriptOp.addWarning("Frame vide.")
        return

    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    if frame.shape[2] < 3:
        frame = np.repeat(frame, 3, axis=2)

    # frame_td : conserve l'orientation TD (bottom-up) pour le mode "data" (passthrough)
    frame_td = frame
    frame_cv = cv2.flip(frame, 0)  # cv2 coords (top-down) pour YOLO
    rgb = frame_cv[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    model_choice = str(scriptOp.par.Model.eval())
    if model_choice == "custom":
        model_path = scriptOp.par.Modelpath.eval().strip()
        model = _get_model(model_path)
    else:
        model = _get_model(model_choice)

    names = _normalize_names(getattr(model, "names", getattr(model.model, "names", {})))
    prompt = scriptOp.par.Prompt.eval().strip()
    classes = _parse_classes(prompt, names, scriptOp)

    try:
        output_mode = str(scriptOp.par.Output.eval())
    except Exception:
        try:
            output_mode = scriptOp.par.Output.menuNames[scriptOp.par.Output.index]
        except Exception:
            output_mode = "overlay"

    try:
        conf = float(scriptOp.par.Conf.eval())
    except Exception:
        conf = 0.25
    if conf <= 0 or conf > 1:
        conf = 0.25

    try:
        seg_id = int(scriptOp.par.Segid.eval())
    except Exception:
        seg_id = -1

    tracker_choice = str(scriptOp.par.Tracker.eval()) if hasattr(scriptOp.par, "Tracker") else "none"

    # Auto-detect imgsz depuis la taille du buffer (arrondi au multiple de 32 superieur, contrainte YOLO)
    _max_dim = max(bgr.shape[0], bgr.shape[1])
    imgsz = ((_max_dim + 31) // 32) * 32
    if imgsz < 64:
        imgsz = 64

    try:
        skip_frames = int(scriptOp.par.Skipframes.eval()) if hasattr(scriptOp.par, "Skipframes") else 0
    except Exception:
        skip_frames = 0
    if skip_frames < 0:
        skip_frames = 0

    predict_device = 0 if cuda_ok else "cpu"

    counter = _FRAME_STATE["counter"]
    _FRAME_STATE["counter"] = counter + 1
    run_inference = True
    if skip_frames > 0 and _FRAME_STATE["last_result"] is not None and (counter % (skip_frames + 1)) != 0:
        run_inference = False

    if run_inference:
        try:
            if tracker_choice in ("bytetrack", "botsort"):
                result = model.track(
                    source=bgr,
                    conf=conf,
                    classes=classes if classes else None,
                    tracker=f"{tracker_choice}.yaml",
                    persist=True,
                    verbose=False,
                    imgsz=imgsz,
                    device=predict_device,
                )[0]
            else:
                result = model.predict(
                    source=bgr,
                    conf=conf,
                    classes=classes if classes else None,
                    verbose=False,
                    imgsz=imgsz,
                    device=predict_device,
                )[0]
        except Exception as e:
            msg = str(e)
            if tracker_choice in ("bytetrack", "botsort") and "lap" in msg.lower():
                scriptOp.addWarning("Tracker requiert le module 'lap' (pip install lapx). Fallback en mode None.")
                tracker_choice = "none"
                try:
                    result = model.predict(
                        source=bgr,
                        conf=conf,
                        classes=classes if classes else None,
                        verbose=False,
                        imgsz=imgsz,
                        device=predict_device,
                    )[0]
                except Exception as e2:
                    scriptOp.addError(f"YOLO predict error: {e2}")
                    return
            else:
                scriptOp.addError(f"YOLO predict error: {e}")
                return
        _FRAME_STATE["last_result"] = result
    else:
        result = _FRAME_STATE["last_result"]

    try:
        target = model.model
        if hasattr(model, "predictor") and model.predictor is not None:
            pm = getattr(model.predictor, "model", None)
            if pm is not None:
                inner = getattr(pm, "model", None)
                if inner is not None and hasattr(inner, "parameters"):
                    target = inner
                elif hasattr(pm, "parameters"):
                    target = pm
        p = next(target.parameters())
        dev = p.device
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else 0
            dev_str = f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
        else:
            dev_str = "cpu"
        if hasattr(scriptOp.par, "Devicestatus") and str(scriptOp.par.Devicestatus.eval()) != dev_str:
            scriptOp.par.Devicestatus.val = dev_str
    except Exception:
        pass

    boxes = getattr(result, "boxes", None)
    xyxy = confs = clss = track_ids = None
    if boxes is not None and boxes.data is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else None
        try:
            track_ids = boxes.id.cpu().numpy().astype(int)
        except Exception:
            track_ids = None
        if track_ids is None:
            track_ids = _assign_track_ids(xyxy, clss)

    masks = getattr(result, "masks", None)
    if output_mode == "mask":
        if masks is not None and masks.data is not None and masks.data.shape[0] > 0:
            out_img = _build_mask(result, rgb.shape[0], rgb.shape[1], scriptOp, seg_id, track_ids)
            out_img = cv2.flip(out_img, 0)
        else:
            out_img = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    elif output_mode == "data":
        # Passthrough: source frame intact, pas de result.plot() => gros gain perf
        # Le dessin des boxes/labels se fait cote TD via les DATs detections/pose_points
        if frame_td.shape[2] >= 4:
            out_img = frame_td[:, :, :4]
        else:
            out_img = np.empty((frame_td.shape[0], frame_td.shape[1], 4), dtype=np.uint8)
            out_img[..., :3] = frame_td[..., :3]
            out_img[..., 3] = 255
    else:
        annotated = result.plot()  # BGR uint8 avec overlay natif
        annotated = cv2.flip(annotated, 0)  # vertical flip (TD coord) — SIMD, contiguous out
        out_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGBA)

    det_dat = op('detections')  # Table DAT optionnel (bboxes)
    pose_dat = op('pose_points')  # Table DAT optionnel (keypoints)
    h_td = rgb.shape[0]
    if xyxy is not None:
        if det_dat is not None:
            det_dat.clear()
            det_dat.appendRow(['id', 'Object Type', 'Confidence', 'X_Center', 'Y_Center', 'Width', 'Height'])
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                cid = int(clss[i]) if clss is not None else -1
                lbl = str(names.get(cid, cid))
                confv = float(confs[i]) if confs is not None else 0.0
                tid = track_ids[i] if track_ids is not None and i < len(track_ids) else i
                ty1, ty2 = h_td - y2, h_td - y1
                cx = (x1 + x2) / 2.0
                cy = (ty1 + ty2) / 2.0
                w_box = x2 - x1
                h_box = ty2 - ty1
                det_dat.appendRow([
                    tid,
                    lbl,
                    f"{confv:.3f}",
                    f"{cx:.1f}",
                    f"{cy:.1f}",
                    f"{w_box:.1f}",
                    f"{h_box:.1f}",
                ])

    kps = getattr(result, "keypoints", None)
    if pose_dat is not None:
        pose_dat.clear()
        if kps is not None and kps.xy is not None:
            kxy = kps.xy.cpu().numpy()  # [N, K, 2]
            pose_dat.appendRow(['det_id', 'kp_id', 'x', 'y'])
            for det_id, pts in enumerate(kxy):
                for kp_id, (kx, ky) in enumerate(pts):
                    ty = h_td - ky
                    pose_dat.appendRow([det_id, kp_id, f"{kx:.1f}", f"{ty:.1f}"])
    else:
        if det_dat is not None:
            det_dat.clear()
        if pose_dat is not None:
            pose_dat.clear()

    out_img = np.ascontiguousarray(out_img)
    scriptOp.copyNumpyArray(out_img)
    return
