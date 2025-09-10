# --- Standard library ---
import glob
import os
import re
import json, datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Union

# --- Third-party ---
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


# module-level holders (auto-bound by load_roi_data_from_h5_folder)
_images = _masks = _filenames = _relevant_frames = _all_volumes = None

# holders for true ED/ES/INFLOW (H5 frame indices)
ed_index_map = {}         # {patient_id: int}
es_index_map = {}         # {patient_id: int or -1}
inflow_indices_map = {}   # {patient_id: [ints]}
volumes_map = {}          # {patient_id: {frame_index: volume}}

# constants for segmentation classes
num_classes = 3
class_mapping = {"myocardium": 1, "blood_pool": 2}


def bloodpool_volume_3d(mask_hws, min_component_area=50, require_overlap=None):
    """
    mask_hws: (H,W,S) with classes 0/1/2; class 2 = blood pool.
    Removes tiny components slice-wise before counting and (optionally) requires
    overlap with a given ED LV footprint to reduce OT/aortic artifacts.
    """
    S = mask_hws.shape[2]
    vol = 0
    for s in range(S):
        sl = (mask_hws[..., s] == 2)

        # small-component suppression
        lab = label(sl)
        keep = np.zeros_like(sl, dtype=bool)
        for rp in regionprops(lab):
            if rp.area >= min_component_area:
                keep[lab == rp.label] = True
        sl = keep

        # optional overlap gating
        if require_overlap is not None:
            sl = sl & require_overlap[..., s]

        vol += int(sl.sum())
    return vol


def pick_ed_es_inflow_from_masks(mask_hwsf, roi_frames):
    """
    mask_hwsf: (H,W,S,F) int mask with classes 0/1/2
    roi_frames: 1D array/list of frame indices (H5 indices) that contain any ROI
    Returns: ED_index, ES_index (or -1), inflow_indices (list), volumes (dict)

    Rule (keeps your acquisition convention & adds physiology check):
      - ED  = earliest ROI frame (min index)
      - ES  = among remaining ROI frames, min blood-pool volume
      - INFLOW = all other ROI frames
      - Prints a warning if the largest blood pool is not at ED
    """
    if roi_frames is None or len(roi_frames) == 0:
        return -1, -1, [], {}

    roi_frames = sorted(int(f) for f in roi_frames)
    ED = roi_frames[0]

    # Build ED LV footprint for overlap gating (myocardium OR blood pool)
    ed_mask = mask_hwsf[..., ED]
    ed_lv_footprint = (ed_mask == 2) | (ed_mask == 1)

    # Compute guarded volumes per ROI frame
    vols = {}
    for f in roi_frames:
        vols[f] = bloodpool_volume_3d(mask_hwsf[..., f],
                                      min_component_area=50,
                                      require_overlap=ed_lv_footprint)

    # Physiology check: max should occur at ED (or very close)
    f_largest = max(roi_frames, key=lambda f: vols[f])
    if f_largest != ED:
        print(f"⚠️ Largest LV blood pool at frame {f_largest}, "
              f"but ED (by rule) is {ED}. (vol_ED={vols[ED]}, vol_max={vols[f_largest]})")

    # ES = min among non-ED ROI frames (tie → earliest)
    if len(roi_frames) > 1:
        others = [f for f in roi_frames if f != ED]
        ES = min(others, key=lambda f: (vols[f], f))
    else:
        ES = -1

    inflow = [f for f in roi_frames if f not in {ED, ES}]
    return ED, ES, inflow, vols



def load_roi_data_from_h5_folder(h5_folder: str):
    """
    Loads all image/mask slices from frames labeled with ROI ...
    (kept original behavior/prints; now also infers ED/ES/INFLOW at load-time)
    """
    h5_files = sorted(glob.glob(os.path.join(h5_folder, 'Mouse*.h5')), key=extract_mouse_number)

    images, masks, processed_filenames = [], [], []
    relevant_frames, all_volumes, all_masks = {}, {}, {}

    # clear/load-time maps for this session
    global ed_index_map, es_index_map, inflow_indices_map, volumes_map
    ed_index_map.clear()
    es_index_map.clear()
    inflow_indices_map.clear()
    volumes_map.clear()

    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as hf:
            img = hf['Images'][:]  # (H, W, S, F)
            msk = hf['Masks'][:]   # (H, W, S, F)
            roi_frames = hf['Frames_with_ROI'][:]

        patient_id = os.path.splitext(os.path.basename(h5_path))[0]
        relevant_frames[patient_id] = roi_frames.tolist()
        all_volumes[patient_id] = img
        all_masks[patient_id]   = msk

        # --- NEW: derive true ED/ES/INFLOW from the Masks on load ---
        ED_idx, ES_idx, inflow_idx, vols = pick_ed_es_inflow_from_masks(msk, roi_frames)
        ed_index_map[patient_id] = int(ED_idx)
        es_index_map[patient_id] = int(ES_idx)
        inflow_indices_map[patient_id] = list(inflow_idx)
        volumes_map[patient_id] = vols

        # Collect 2D slices with ROI (unchanged)
        for f in roi_frames:
            if not np.any(msk[:, :, :, f] > 0):
                continue
            for s in range(img.shape[2]):
                images.append(img[:, :, s, f])
                masks.append(msk[:, :, s, f])
                processed_filenames.append(f"{patient_id}_sl{s:02d}_fr{f:02d}")

    # Save mask check results for later printing (unchanged)
    masks_check = []
    for m, fname in zip(masks, processed_filenames):
        u = np.unique(m)
        if not set(u.tolist()).issubset({0, 1, 2}):
            masks_check.append((fname, u))

    # --- auto-bind for plotting convenience (unchanged) ---
    global _images, _masks, _filenames, _relevant_frames, _all_volumes
    _images = np.asarray(images)
    _masks = np.asarray(masks)
    _filenames = processed_filenames
    _relevant_frames = relevant_frames
    _all_volumes = all_volumes

    return (
        _images,
        _masks,
        _filenames,
        _relevant_frames,
        all_volumes,
        all_masks,
        masks_check
    )


def summarise_mice_and_masks(relevant_frames, masks_check):
    """Print patient classification, then dataset + mask summary.

    Notes:
      - ED+ES frames are assumed to be the first two entries of each patient's
        Frames_with_ROI list; any further frames are counted as Inflow.
      - Uses module-level _filenames / _masks populated by load_roi_data_from_h5_folder().
      - NEW: If load-time ED/ES/INFLOW were computed, prefer them; otherwise fall back.
    """
    # (kept structure and headings the same)
    ed_es_patients = []
    ed_es_inflow_patients = []

    # prefer load-time maps if available
    for pid, frames in relevant_frames.items():
        if pid in ed_index_map:
            ED = ed_index_map[pid]
            ES = es_index_map.get(pid, -1)
            INFLOW = inflow_indices_map.get(pid, [])
            # classify by presence of inflow
            if INFLOW:
                ed_es_inflow_patients.append(pid)
            else:
                ed_es_patients.append(pid)
        else:
            # fallback to assumption: first two = ED+ES, rest inflow
            if len(frames) > 2:
                ed_es_inflow_patients.append(pid)
            else:
                ed_es_patients.append(pid)

    print(f"\nPatients with ED+ES+Inflow: {len(ed_es_inflow_patients)}")
    for pid in ed_es_inflow_patients:
        print("  ", pid)

    print(f"\nPatients with ED+ES only: {len(ed_es_patients)}")
    for pid in ed_es_patients:
        print("  ", pid)

    # ---- Dataset summary ----
    total_2d_images = 0
    if _all_volumes is not None:
        for vol in _all_volumes.values():  # (H, W, S, F)
            if vol.ndim == 4:
                total_2d_images += vol.shape[2] * vol.shape[3]

    roi_labeled_2d = 0
    if _masks is not None:
        roi_labeled_2d = int(np.sum([m.max() > 0 for m in _masks]))

    # ---- New: split ROI-labeled 2D counts into ED+ES vs Inflow ----
    ed_es_roi = inflow_roi = 0
    if _masks is not None and _filenames is not None:
        # Per-patient frame sets from load-time decision
        fname_re = re.compile(r'^(?P<pid>.+?)_sl(?P<sl>\d+)_fr(?P<fr>\d+)$')

        for m, fname in zip(_masks, _filenames):
            if m.max() <= 0:
                continue  # only count slices that actually contain ROI
            mm = fname_re.match(fname)
            if not mm:
                continue
            pid = mm.group('pid')
            f = int(mm.group('fr'))

            if pid in ed_index_map:
                ED = ed_index_map[pid]
                ES = es_index_map.get(pid, -1)
                INFLOW = set(inflow_indices_map.get(pid, []))
                if f == ED or f == ES:
                    ed_es_roi += 1
                elif f in INFLOW:
                    inflow_roi += 1
            else:
                # Fallback to old behavior: first two = ED+ES; rest inflow
                rf = relevant_frames.get(pid, [])
                if f in rf[:2]:
                    ed_es_roi += 1
                elif f in rf[2:]:
                    inflow_roi += 1

    num_patients = len(relevant_frames)

    print("\nDataset summary:")
    print(f"  Number of patients: {num_patients}")
    print(f"  Total number of 2D images (all slices × all frames): {total_2d_images}")
    print(f"  Total number of ROI-labeled images (with segmentation): {roi_labeled_2d}")
    # Newly added detail lines (kept wording):
    print(f"    Total number of ED + ES ROI-labeled images: {ed_es_roi}")
    print(f"    Total number of Inflow ROI-labeled images: {inflow_roi}")

    # Optional sanity note if something doesn't sum up exactly
    leftover = roi_labeled_2d - (ed_es_roi + inflow_roi)
    if leftover > 0:
        print(f"    (Unclassified ROI-labeled images: {leftover})")

    # ---- Mask value check LAST ----
    if masks_check:
        print(f"\nWARNING: Found {len(masks_check)} mask(s) with values outside [0,1,2]:")
        for fname, u in masks_check[:10]:
            print(f"  {fname}: {u}")
        if len(masks_check) > 10:
            print(f"  ...and {len(masks_check)-10} more.")
    else:
        print(f"\nAll {roi_labeled_2d} masks with ROI have unique values [0,1,2]")


def plot_dicom_with_mask(mouse_index, slice_number, frame_index):
    """
    QC visualization:
      - Shows the original image
      - Shows the segmentation mask
      - Shows binary masks for each class
    """
    if any(x is None for x in (_images, _masks, _filenames, _relevant_frames, _all_volumes)):
        raise RuntimeError("Data not loaded. Call load_roi_data_from_h5_folder(...) first.")

    pid   = list(_relevant_frames.keys())[mouse_index]
    frame = _relevant_frames[pid][frame_index]
    fname = f"{pid}_sl{slice_number:02d}_fr{frame:02d}"  # keep consistent with how filenames were built

    try:
        idx = _filenames.index(fname)
    except ValueError:
        raise ValueError(f"Could not find slice '{fname}' in loaded filenames.")

    img = _images[idx]
    msk = _masks[idx]
    sm  = msk

    # --- modified print block ---
    print(f"\nSelected: {pid} (slice={slice_number}, frame={frame})")
    print(f"Shape (H, W, S, F): {_all_volumes[pid].shape}")
    print("Image dtype:", img.dtype, "Mask dtype:", msk.dtype)
    print("Raw mask values:", np.unique(msk))

    # --- visualization ---
    classes = [(0, "Background"), (1, "Myocardium"), (2, "Blood Pool")]
    fig, axs = plt.subplots(1, 5, figsize=(18, 5))

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original Image"); axs[0].axis("off")

    axs[1].imshow(msk, cmap='viridis', vmin=0, vmax=2)
    axs[1].set_title("Original Mask"); axs[1].axis("off")

    for ax, (i, label) in zip(axs[2:], classes):
        ax.imshow((sm == i).astype(int), cmap='viridis', vmin=0, vmax=1)
        ax.set_title(label); ax.axis("off")

    fig.tight_layout()
    plt.show()


def extract_mouse_number(h5_path):
    import re
    basename = os.path.basename(h5_path)
    match = re.search(r'Mouse(\d+)\.h5', basename)
    return int(match.group(1)) if match else 0


# Extract all metadata from filename
def parse_filename(fname):
    """
    Parses a filename like 'Mouse13_sl05_fr12' into:
        - patient_id: 'Mouse13'
        - slice_number: 5
        - frame_number: 12
    """
    parts = fname.split('_')
    patient_id = parts[0]
    slice_number = int(parts[1].replace('sl', ''))
    frame_number = int(parts[2].replace('fr', ''))
    return patient_id, slice_number, frame_number


def train_val_test_dis(
    all_mice,
    train_mice, val_mice, test_mice,
    train_images, train_masks,
    val_images,   val_masks,
    test_images,  test_masks,
):
    print(f"\nDataset distribution (n = {len(all_mice)}):")
    print("Number of training mice:", len(train_mice))
    print("Number of validation mice:", len(val_mice))
    print("Number of test mice:", len(test_mice))

    total_images = len(train_images) + len(val_images) + len(test_images)
    total_masks  = len(train_masks)  + len(val_masks)  + len(test_masks)

    print(f"\nAll images shape: ({total_images}, 256, 256)")
    print(f"All masks shape:  ({total_masks}, 256, 256)")
    print("Training set images shape:", np.array(train_images).shape)
    print("Training set masks shape:",  np.array(train_masks).shape)
    print("Validation set images shape:", np.array(val_images).shape)
    print("Validation set masks shape:",  np.array(val_masks).shape)
    print("Test set images shape:", np.array(test_images).shape)
    print("Test set masks shape:",  np.array(test_masks).shape)

    print("\nTraining mice:")
    for m in sorted(train_mice, key=lambda x: int(x.replace('Mouse', ''))):
        print(" ", m)

    print("\nValidation mice:")
    for m in sorted(val_mice, key=lambda x: int(x.replace('Mouse', ''))):
        print(" ", m)

    print("\nTest mice:")
    for m in sorted(test_mice, key=lambda x: int(x.replace('Mouse', ''))):
        print(" ", m)


def normalise_images(images):
    """
    Per-image min–max normalisation to [0,1].
    Works for (N,H,W), (N,H,W,1) and single (H,W). Returns float32.
    Flat images (max==min) become all zeros.
    """
    arr = np.asarray(images, dtype=np.float32)

    # Squeeze a trailing channel of size 1 if present (e.g., (N,H,W,1))
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 2:  # single image (H,W)
        mn = arr.min()
        mx = arr.max()
        rng = mx - mn
        return ((arr - mn) / rng).astype(np.float32) if rng > 0 else np.zeros_like(arr, dtype=np.float32)

    # Batch case: reduce over the last two dims (H,W)
    mins  = arr.min(axis=(-2, -1), keepdims=True)  # (...,1,1)
    maxs  = arr.max(axis=(-2, -1), keepdims=True)  # (...,1,1)
    denom = maxs - mins                            # (...,1,1)

    out = np.zeros_like(arr, dtype=np.float32)
    np.divide(arr - mins, denom, out=out, where=denom > 0)  # safe, no post-assignment needed
    return out


def data_generator_with_augmentation(images, masks, batch_size, augmentation_pipeline):
    while True:
        indices = np.random.choice(len(images), batch_size)
        batch_images, batch_masks = [], []
        for i in indices:
            # Works for TF EagerTensors; Albumentations needs uint8 images, int masks
            img  = tf.cast(images[i] * 255.0, tf.uint8).numpy()   # (H, W) uint8 in [0,255]
            mask = tf.cast(masks[i],          tf.int32).numpy()   # (H, W) ints {0,1,2}
            # Add the augmented image and mask to the batch
            augmented = augmentation_pipeline(image=img, mask=mask)
            batch_images.append(augmented["image"])  # still uint8
            batch_masks.append(augmented["mask"])    # still int

        # Normalize images to [0,1] and add channel dim
        batch_images = normalise_images(np.asarray(batch_images, dtype=np.float32))[..., None] # (B,H,W,1)
        # One-hot masks AFTER augmentation → (B,H,W,3) float32 with values {0.,1.}
        batch_masks = to_categorical(
            np.asarray(batch_masks, dtype=np.int32), num_classes=num_classes
        ).astype(np.float32)
        yield batch_images, batch_masks


# === In-memory per-patient collation + inference + ED/ES Dice (clean variant) ===

def predict_per_patient_in_memory(
    model,
    patient_images_4d: Dict[str, np.ndarray],
    verbose: int = 0
) -> Dict[str, np.ndarray]:
    """Run inference in memory. Deep-supervision aware (uses last head). Returns {pid: (H,W,S,F) uint8}."""
    preds_by_pid = {}
    for pid, vol in patient_images_4d.items():
        H, W, S, F = vol.shape
        preds = np.zeros((H, W, S, F), dtype=np.uint8)
        for f in range(F):
            frame = vol[:, :, :, f]                      # (H,W,S)
            batch = np.moveaxis(frame, -1, 0)[..., None] # (S,H,W,1)
            out   = model.predict(batch, verbose=verbose)
            probs = out[-1] if isinstance(out, (list, tuple)) else out  # robust DS selection
            labels = np.argmax(probs, axis=-1).astype(np.uint8)         # (S,H,W)
            preds[:, :, :, f] = np.moveaxis(labels, 0, -1)              # -> (H,W,S)
        preds_by_pid[pid] = preds
    return preds_by_pid

@dataclass
class RunConfig:
    model_name: str
    batch_size: int
    epochs: int
    input_shape: tuple
    optimizer: str
    loss: str
    metrics: list
    train_n: int
    val_n: int
    test_n: int
    train_mice: list
    val_mice: list
    test_mice: list

def now_stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def make_run_dir(base="runs", model_name="SEG", when=None):
    when = when or now_stamp()
    run_dir = os.path.join(base, f"{model_name}_{when}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "eval"), exist_ok=True)
    return run_dir

def dump_config_json(run_dir, cfg: RunConfig):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def island_removal(
    segmentation_slices,
    min_slices=2,
    min_area=10,
    distance_threshold=40
):
    """
    Removes inconsistent segmentations robustly:
    - Keeps any component that:
        - Spans at least min_slices slices, AND
        - Is within distance_threshold of the main centroid
    - Ensures large far-away blobs are removed.

    Args:
        segmentation_slices: (H, W, S) array
        min_slices: min number of slices to keep a component
        min_area: min voxel count to keep a component
        distance_threshold: max centroid distance to keep

    Returns:
        cleaned_segmentation: (H, W, S) array
    """
    H, W, S = segmentation_slices.shape
    cleaned_segmentation = np.zeros_like(segmentation_slices, dtype=np.uint8)

    for class_label, label_name in [(1, "myocardium"), (2, "blood_pool")]:
        binary_mask = (segmentation_slices == class_label).astype(np.uint8)
        labels_3d = label(binary_mask, connectivity=1)

        # Collect all candidate components
        regions = [r for r in regionprops(labels_3d) if r.area >= min_area]
        if not regions:
            continue

        # Find the component with the most slices (dominant region)
        dominant_region = max(
            regions,
            key=lambda r: len(set(c[2] for c in r.coords))
        )
        dominant_centroid = np.array(dominant_region.centroid)

        # Now process each component
        for region in regions:
            slices_present = set(c[2] for c in region.coords)
            centroid = np.array(region.centroid)
            distance = np.linalg.norm(centroid - dominant_centroid)

            if len(slices_present) >= min_slices and distance <= distance_threshold:
                # Keep it
                for c in region.coords:
                    cleaned_segmentation[c[0], c[1], c[2]] = class_label

    return cleaned_segmentation



def normalize_4d(vol_hwsf: np.ndarray) -> np.ndarray:
    """
    Per-(H,W) slice min-max normalization to [0,1] across a cine volume.
    Input:  vol_hwsf (H, W, S, F)  -> typically uint16
    Output: float32 (H, W, S, F) in [0,1], matching training normalization.
    """
    H, W, S, F = vol_hwsf.shape
    flat = np.transpose(vol_hwsf, (2, 3, 0, 1)).reshape(S * F, H, W)
    flat = normalise_images(flat)  # reuse your training normalisation
    vol_norm = flat.reshape(S, F, H, W)
    vol_norm = np.transpose(vol_norm, (2, 3, 0, 1)).astype(np.float32)
    return vol_norm


def build_fullcine_for(
    pids: List[str],
    source_volumes: Dict[str, np.ndarray] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    """
    Create {pid: images_4d_full} and {pid: frame_id_list} for the *entire* cine.
    If source_volumes is None, falls back to module-level _all_volumes.
    Normalizes per slice to match training.
    """
    vols = source_volumes if source_volumes is not None else _all_volumes
    if vols is None:
        raise RuntimeError("No volumes available. Call load_roi_data_from_h5_folder(...) "
                           "or pass source_volumes=all_volumes.")
    imgs4d_full, frame_ids = {}, {}
    for pid in pids:
        if pid not in vols:
            raise KeyError(f"{pid} not found in provided volumes.")
        vol = vols[pid].astype(np.float32)  # (H,W,S,F_all)
        vol = normalize_4d(vol)             # per-slice min–max to [0,1]
        imgs4d_full[pid] = vol
        frame_ids[pid] = list(range(vol.shape[3]))  # [0..F_all-1]
    return imgs4d_full, frame_ids



from typing import Optional, List, Dict, Union

WhichType = Union[str, int, List[int]]

# ---------------------------
# Original function (kept; uses 0-based indexing internally for frames)
# ---------------------------
def _resolve_frames(pid, frame_id_list, ed_map, es_map, which):
    """Resolve 'which' into a list of frame indices (0..F-1)."""
    if isinstance(which, str):
        w = which.strip().lower()
        if w == "ed":
            return [frame_id_list.index(ed_map[pid])]
        if w == "es":
            return [frame_id_list.index(es_map[pid])]
        if w in ("all", "every"):
            return list(range(len(frame_id_list)))
        raise ValueError("which must be 'ED', 'ES', 'all', int, or list[int]")
    if isinstance(which, int):
        return [which]
    if isinstance(which, (list, tuple, np.ndarray)):
        return list(which)
    raise ValueError("which must be 'ED', 'ES', 'all', int, or list[int]")



def plot_all_slices_for_frame(
    pid: str,
    images_4d: np.ndarray,                              # (H, W, S, F)
    masks_4d: Optional[np.ndarray] = None,              # optional; not used here
    preds_4d: Optional[np.ndarray] = None,              # (H, W, S, F) labels {0,1,2}
    frame_id_list: Optional[List[int]] = None,          # len = F (H5 indices in cine order)
    ed_map: Optional[Dict] = None,
    es_map: Optional[Dict] = None,
    which: WhichType = "ED",                            # legacy selector
    select_frame: Optional[WhichType] = None,           # preferred alias for 'which'
    save: bool = False,
    save_dir: str = "plots",
    cols: int = 4,
    dpi: int = 150,
    alpha_myo: float = 0.5,
    alpha_blood: float = 0.5,
    print_header: bool = True,                          # suppress to avoid double header
) -> None:
    """
    Montage of all slices for chosen frame(s) with predicted overlays (myo=Blues, blood=jet).
    Only prints the header if print_header=True.
    """
    # resolve selector (alias)
    _which = select_frame if select_frame is not None else which

    if preds_4d is None:
        raise ValueError("preds_4d is required for overlays.")

    H, W, S, F = images_4d.shape
    if preds_4d.shape != images_4d.shape:
        raise ValueError(f"preds_4d shape {preds_4d.shape} must match images_4d {images_4d.shape}.")

    if frame_id_list is None:
        frame_id_list = list(range(F))

    # Pretty header (only if requested)
    if print_header:
        def _pos_or_none(h5_idx):
            if h5_idx is None:
                return None
            try:
                return frame_id_list.index(h5_idx) + 1  # 1-based for display
            except ValueError:
                return None
        ed_pos = _pos_or_none((ed_map or {}).get(pid))
        es_pos = _pos_or_none((es_map or {}).get(pid))
        print(f"\nTest Mouse ID: {pid}")
        print(f"Shape (H, W, S, F): {H}, {W}, {S}, {F}")
        print(f"ED Frame: {ed_pos if ed_pos is not None else 'NA'}")
        print(f"ES Frame: {es_pos if es_pos is not None else 'NA'}")

    # Resolve frame indices (0-based)
    frame_idxs = _resolve_frames(pid, frame_id_list, ed_map or {}, es_map or {}, _which)

    if save:
        os.makedirs(save_dir, exist_ok=True)

    for f_idx in frame_idxs:
        if not (0 <= f_idx < F):
            raise IndexError(f"Frame index {f_idx} out of range 0..{F-1}")

        imgs = images_4d[..., f_idx]  # (H,W,S)
        pred = preds_4d[..., f_idx]   # (H,W,S)

        rows = int(np.ceil(S / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=dpi)
        axes = np.atleast_1d(axes).ravel()

        for s in range(S):
            ax = axes[s]
            ax.imshow(imgs[:, :, s], cmap="gray")

            myo_mask   = (pred[:, :, s] == 1)
            blood_mask = (pred[:, :, s] == 2)

            if myo_mask.any():
                ax.imshow(
                    np.ma.masked_where(~myo_mask, myo_mask.astype(np.uint8)),
                    cmap="Blues", alpha=alpha_myo, vmin=0, vmax=1
                )
            if blood_mask.any():
                ax.imshow(
                    np.ma.masked_where(~blood_mask, blood_mask.astype(np.uint8)),
                    cmap="jet", alpha=alpha_blood, vmin=0, vmax=1
                )

            ax.set_title(f"Slice {s+1}/{S}")
            ax.axis("off")

        for ax in axes[S:]:
            ax.axis("off")

        # Tag ED/ES on title if applicable
        tag = []
        try:
            ed_pos = frame_id_list.index((ed_map or {}).get(pid)) + 1
            if (f_idx + 1) == ed_pos: tag.append("ED")
        except Exception:
            pass
        try:
            es_val = (es_map or {}).get(pid)
            if es_val is not None and es_val >= 0:
                es_pos = frame_id_list.index(es_val) + 1
                if (f_idx + 1) == es_pos: tag.append("ES")
        except Exception:
            pass
        tag_str = f" ({'/'.join(tag)})" if tag else ""

        fig.suptitle(f"{pid}", y=1.02, fontsize=21, fontweight="bold")
        fig.text(0.5, 0.99, f"Frame: {f_idx+1}{tag_str}",
                 ha="center", va="top", fontsize=18, fontweight="bold")

        plt.tight_layout()
        if save:
            out = os.path.join(save_dir, f"{pid}_frame{f_idx+1}.png")
            plt.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.show()


# ---------------------------
# 1-based wrapper for frames (recommended for calling code)
# ---------------------------

def plot_all_slices_for_frame_1based(
    pid: str,
    images_4d: np.ndarray,
    masks_4d: Optional[np.ndarray] = None,     # optional; not used here
    preds_4d: Optional[np.ndarray] = None,
    frame_id_list: Optional[List[int]] = None,
    ed_map: Optional[Dict] = None,
    es_map: Optional[Dict] = None,
    which: WhichType = "ED",                   # legacy selector
    select_frame: Optional[WhichType] = None,  # preferred alias
    print_header: bool = True,
    **kwargs
) -> None:
    """
    Wrapper to allow 1-based frame selection.
    - Strings: "ED", "ES", "all", "every" pass through unchanged.
    - Numeric string "12" → 11 (0-based); "3,7,12" → [2,6,11].
    - Integers/lists: convert from 1-based to 0-based indices.
    """
    sel = select_frame if select_frame is not None else which

    if isinstance(sel, str):
        w = sel.strip().lower()
        if w in ("ed", "es", "all", "every"):
            sel0 = "all" if w == "every" else sel  # keep semantic strings
        elif w.isdigit():
            sel0 = int(w) - 1
        elif "," in w:
            parts = [p.strip() for p in w.split(",")]
            if not all(p.isdigit() for p in parts):
                raise ValueError("Numeric list must be like '3,7,12'.")
            sel0 = [int(p) - 1 for p in parts]
        else:
            raise ValueError("which/select_frame must be 'ED', 'ES', 'all', int, list[int], or numeric string.")
    elif isinstance(sel, int):
        sel0 = sel - 1
    else:
        sel0 = [int(x) - 1 for x in sel]  # list/tuple/np.ndarray of 1-based ints

    return plot_all_slices_for_frame(
        pid=pid,
        images_4d=images_4d,
        masks_4d=masks_4d,
        preds_4d=preds_4d,
        frame_id_list=frame_id_list,
        ed_map=ed_map,
        es_map=es_map,
        which=sel0,
        print_header=print_header,
        **kwargs
    )


# ========= ED/ES display helper (public) =========
def ed_es_positions_for_display(pid, frame_id_list, ed_map, es_map):
    """
    Return (ed_pos_1based, es_pos_1based, ed_idx_0based_or_None, es_idx_0based_or_None)
    by mapping true H5 frame IDs from ed_map/es_map into this cine's frame_id_list.
    """
    ed_h5 = ed_map.get(pid, -1)
    es_h5 = es_map.get(pid, -1)

    def to_zero_based(h5_id):
        try:
            return frame_id_list.index(int(h5_id))
        except Exception:
            return None

    ed0 = to_zero_based(ed_h5)
    es0 = to_zero_based(es_h5) if (es_h5 is not None and es_h5 >= 0) else None
    ed1 = (ed0 + 1) if ed0 is not None else None
    es1 = (es0 + 1) if es0 is not None else None
    return ed1, es1, ed0, es0


# ========= One-call convenience: print per-patient ED/ES Dice, then plot the frames you want =========

def report_and_plot_fullcine_1based(
    pid,
    images_4d, preds_4d, frame_id_list,
    ed_map, es_map,
    which="ED",
    select_frame: Optional[WhichType] = None,  # preferred alias
    save=False,
    masks_4d=None,                 # pass GT to compute Dice
    show_dice=True,
    **plot_kwargs
):
    """
    1) Prints the per-mouse ED/ES Dice summary (if show_dice and masks_4d provided).
    2) Plots the frames requested by 'which' / 'select_frame'.
       (Plot choice is decoupled from how Dice is computed.)
    """
    if show_dice and masks_4d is not None:
        # This prints the single header + the nicely formatted Dice block
        print_patient_ed_es_dice_summary(
            pid=pid,
            images_4d=images_4d,
            preds_4d=preds_4d,
            masks_4d=masks_4d,
            frame_id_list=frame_id_list,
            ed_map=ed_map,
            es_map=es_map
        )
    else:
        # Print a lightweight header only once
        H, W, S, F = images_4d.shape
        ed1, es1, _, _ = ed_es_positions_for_display(pid, frame_id_list, ed_map, es_map)
        print(f"\nTest Mouse ID: {pid}")
        print(f"Shape (H, W, S, F): {H}, {W}, {S}, {F}")
        print("ED Frame:", ed1 if ed1 is not None else "NA")
        print("ES Frame:", es1 if es1 is not None else "NA")
        print("." * 28)
        print("(Dice summary skipped — pass masks_4d and set show_dice=True)")

    # Forward to the plotter WITHOUT printing the header again
    plot_all_slices_for_frame_1based(
        pid=pid,
        images_4d=images_4d,
        preds_4d=preds_4d,
        frame_id_list=frame_id_list,
        ed_map=ed_map,
        es_map=es_map,
        which=which,
        select_frame=select_frame,   # alias supported
        save=save,
        print_header=False,          # avoid duplicate header
        **plot_kwargs
    )
        

# ========= Preds vs GT (ED & ES) =========

def plot_pred_then_gt_for_ed_es(
    pid: str,
    images_4d: np.ndarray,
    masks_4d: np.ndarray,
    preds_4d: np.ndarray,
    frame_id_list: List[int],   # cine order of H5 ids
    ed_map: dict,
    es_map: dict,
    cols: int = 4,
    dpi: int = 150,
    alpha_myo: float = 0.5,
    alpha_blood: float = 0.5,
    row_headers: bool = True,
    save: bool = False,
    save_dir: str = None
) -> None:
    """Plot only ED (and ES if available) for a patient."""
    f_idxs = []
    try:
        f_idxs.append(frame_id_list.index(int(ed_map[pid])))
    except Exception:
        pass
    try:
        es_h5 = es_map.get(pid, -1)
        if es_h5 is not None and es_h5 >= 0:
            f_idxs.append(frame_id_list.index(int(es_h5)))
    except Exception:
        pass

    if not f_idxs:
        print(f"[{pid}] No ED/ES frames could be resolved for plotting.")
        return

    for f_idx in f_idxs:
        out_path = None
        if save:
            label = "ED" if f_idx == f_idxs[0] else "ES"
            base = save_dir or "."
            os.makedirs(base, exist_ok=True)
            out_path = os.path.join(base, f"{pid}_frame{f_idx+1}_{label}_pred_then_gt.png")

        plot_pred_then_gt_for_frame(
            pid=pid,
            images_4d=images_4d,
            masks_4d=masks_4d,
            preds_4d=preds_4d,
            frame_id_list=frame_id_list,
            ed_map=ed_map,
            es_map=es_map,
            select_frame=f_idx,      # pass cine position directly
            cols=cols,
            dpi=dpi,
            alpha_myo=alpha_myo,
            alpha_blood=alpha_blood,
            row_headers=row_headers,
            save=save,
            save_path=out_path
        )

def plot_pred_then_gt_for_frame(
    pid: str,
    images_4d: np.ndarray,        # (H, W, S, F)
    masks_4d: np.ndarray,         # (H, W, S, F)
    preds_4d: np.ndarray,         # (H, W, S, F)
    frame_id_list: List[int],
    ed_map: dict,
    es_map: dict,
    select_frame: WhichType = "ES",   # "ED", "ES", 0-based int, or list[int]
    cols: int = 4,
    dpi: int = 150,
    alpha_myo: float = 0.5,
    alpha_blood: float = 0.5,
    row_headers: bool = True,
    save: bool = False,
    save_path: str = None
) -> None:
    """Predictions on top, Ground Truth below, for one or more frames (one figure per frame)."""
    f_idxs = _resolve_frames(pid, frame_id_list, ed_map, es_map, select_frame)
    if not isinstance(f_idxs, (list, tuple, np.ndarray)):
        f_idxs = [f_idxs]

    for f_idx in f_idxs:
        imgs = images_4d[..., f_idx]  # (H, W, S)
        gt   = masks_4d[...,  f_idx]
        pr   = preds_4d[...,  f_idx]

        H, W, S = imgs.shape
        rows = int(np.ceil(S / cols))
        fig, axes = plt.subplots(rows * 2, cols, figsize=(4 * cols, 4 * rows * 2), dpi=dpi)
        axes = np.atleast_2d(axes)

        def _overlay(ax, im2d, lab2d):
            ax.imshow(im2d, cmap="gray")
            myo   = (lab2d == 1)
            blood = (lab2d == 2)
            if myo.any():
                ax.imshow(np.ma.masked_where(~myo, myo.astype(np.uint8)), cmap="Blues",
                          alpha=alpha_myo, vmin=0, vmax=1)
            if blood.any():
                ax.imshow(np.ma.masked_where(~blood, blood.astype(np.uint8)), cmap="jet",
                          alpha=alpha_blood, vmin=0, vmax=1)
            ax.axis("off")

        # top: Preds, bottom: GT
        for s in range(S):
            r_pred = (s // cols) * 2
            c      = s % cols
            r_gt   = r_pred + 1
            _overlay(axes[r_pred, c], imgs[:, :, s], pr[:, :, s]); axes[r_pred, c].set_title(f"Slice {s+1}/{S} – Pred")
            _overlay(axes[r_gt,  c], imgs[:, :, s], gt[:, :, s]); axes[r_gt,  c].set_title(f"Slice {s+1}/{S} – GT")

        # hide unused cells
        for r in range(rows * 2):
            for c in range(cols):
                idx = (r // 2) * cols + c
                if idx >= S:
                    axes[r, c].axis("off")

        # row headers
        if row_headers and cols > 0:
            first_col = 0
            for r in range(0, rows * 2, 2):
                if 0 <= r < rows * 2:
                    axes[r, first_col].set_ylabel("Predictions", fontsize=12, fontweight="bold",
                                                  rotation=90, labelpad=10)
                if 0 <= r + 1 < rows * 2:
                    axes[r + 1, first_col].set_ylabel("Ground Truth", fontsize=12, fontweight="bold",
                                                      rotation=90, labelpad=10)

        # ED/ES tag
        tag = []
        try:
            ed_pos = frame_id_list.index(ed_map.get(pid)) + 1
            if (f_idx + 1) == ed_pos: tag.append("ED")
        except Exception:
            pass
        try:
            es_val = es_map.get(pid)
            if es_val is not None and es_val >= 0:
                es_pos = frame_id_list.index(es_val) + 1
                if (f_idx + 1) == es_pos: tag.append("ES")
        except Exception:
            pass
        tag_str = f" ({'/'.join(tag)})" if tag else ""
        fig.suptitle(f"{pid} — Frame {f_idx+1}{tag_str}", y=0.995, fontsize=18, fontweight="bold")

        # warn if GT empty
        if not ((gt == 1).any() or (gt == 2).any()):
            print(f"[{pid}] No GT ROI for frame {f_idx+1} — only ED/ES (and some inflow) are annotated.")

        plt.tight_layout()
        if save:
            out = save_path or f"{pid}_frame{f_idx+1}_pred_then_gt.png"
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            plt.savefig(out, bbox_inches="tight", dpi=dpi)
            print(f"Saved: {out}")
        plt.show()
        plt.close(fig)

def export_history_json(history, run_dir):
    data = history.history if hasattr(history, "history") else history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(data, f, indent=2)

def save_loss_and_dice_plots(history, out_dir=".", to_percent=True):
    """
    Saves:
      - loss_plot.png   (train vs val loss)
      - dice_combined.png (train vs val Dice for myocardium & blood ONLY)

    Returns: dict of saved file paths, e.g. {"loss": ".../loss_plot.png", "dice": ".../dice_combined.png"}
    """
    hist = history.history if hasattr(history, "history") else history
    saved = {}

    # --------- Loss plot ---------
    if "loss" in hist and "val_loss" in hist:
        epochs = range(1, len(hist["loss"]) + 1)
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, hist["loss"], label="Training loss")
        plt.plot(epochs, hist["val_loss"], label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
        loss_path = os.path.join(out_dir, "loss_plot.png")
        plt.tight_layout(); plt.savefig(loss_path, bbox_inches="tight"); plt.show()
        print(f"Plots Saved: {loss_path}")
        saved["loss"] = loss_path
    else:
        print("Skipped loss plot (keys not found).")

    # --------- Dice (Myo + Blood only), styled per your spec ---------
    myo_key   = _pick_metric_key(hist, "dice_myo")
    blood_key = _pick_metric_key(hist, "dice_blood")

    curves = []
    if myo_key:
        vk = _val_key_of(myo_key, hist)
        # label, train_vals, val_vals, ls_train, ls_val, c_train, c_val
        if vk: curves.append(("Myocardium Dice", hist[myo_key], hist[vk], "-", "--", "blue", "blue"))
    if blood_key:
        vk = _val_key_of(blood_key, hist)
        if vk: curves.append(("Blood Pool Dice", hist[blood_key], hist[vk], "-", "--", "red", "red"))

    if curves:
        # length based on first available curve
        epochs = range(1, len(curves[0][1]) + 1)
        plt.figure()
        for label, train_vals, val_vals, ls_tr, ls_va, c_tr, c_va in curves:
            y_tr = [v*100 for v in train_vals] if to_percent else train_vals
            y_va = [v*100 for v in val_vals]  if to_percent else val_vals
            # Train vs Val with separate styles/colors
            plt.plot(epochs, y_tr, linestyle=ls_tr, color=c_tr, label=f"Train {label}")
            plt.plot(epochs, y_va, linestyle=ls_va, color=c_va, label=f"Val {label}")

        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Dice (%)" if to_percent else "Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        dice_path = os.path.join(out_dir, "dice_combined.png")
        plt.savefig(dice_path, bbox_inches="tight"); plt.show()
        print(f"Plots Saved: {dice_path}")
        saved["dice"] = dice_path
    else:
        print("Skipped Dice plot (required keys not found).")

    return saved


def _pick_metric_key(hist: dict, suffix: str,
                     prefer_substrings=("deepsup_activation_1_", "deepsup_conv_1_2_", "Decoder1", "decoder1", "")):
    """
    Pick a train-metric key that ends with `suffix` (e.g., 'dice_myo'),
    preferring names that usually correspond to the final deep-supervision head.
    Returns None if not found.
    """
    cands = [k for k in hist.keys() if not k.startswith("val_") and k.endswith(suffix)]
    for pref in prefer_substrings:
        for k in cands:
            if pref in k:
                return k
    return cands[0] if cands else None

def _val_key_of(train_key: str, hist: dict):
    vk = f"val_{train_key}"
    return vk if train_key and vk in hist else None

# ========= Print a per-patient ED/ES Dice summary (independent of what you plot) =========
def print_patient_ed_es_dice_summary(
    pid,
    images_4d,          # (H,W,S,F_all)
    preds_4d,           # (H,W,S,F_all) int {0,1,2}
    masks_4d,           # (H,W,S,F_all) int {0,1,2}
    frame_id_list,      # list[int]
    ed_map, es_map,
    klass_map=class_mapping,
    smooth=1e-6
):
    """
    Prints:
      Test Mouse ID
      Shape (H, W, S, F)
      ED/ES
      <blank line>
      ===========================
      Dice Score (per mouse)
      ===========================
      ...
    Returns: scores dict from dice_for_patient_ed_es_kerasstyle.
    """

    H, W, S, F = images_4d.shape
    ed1, es1, _, _ = ed_es_positions_for_display(pid, frame_id_list, ed_map, es_map)

    # Header
    print(f"Test Mouse ID: {pid}")
    print(f"Shape (H, W, S, F): {H}, {W}, {S}, {F}")
    print("ED Frame:", ed1 if ed1 is not None else "NA")
    print("ES Frame:", es1 if es1 is not None else "NA")
    print()  # blank line
    # print("===========================")
    # print("Dice Score (per mouse)")
    # print("===========================")
    print("===========================")
    print(f"Dice Score ({pid})")
    print("===========================")


    # Scores
    scores = dice_for_patient_ed_es_kerasstyle(
        pid=pid,
        masks_4d=masks_4d.astype(np.uint8),
        preds_4d=preds_4d.astype(np.uint8),
        frame_id_list=list(frame_id_list),
        ed_map=ed_map,
        es_map=es_map,
        class_mapping=klass_map,
        empty_empty_equals_one=True,
        smooth=smooth
    )

    def fmt(key):
        return f"{scores[key]:.4f}" if key in scores and np.isfinite(scores[key]) else "NA"

    print("Diastolic Myocardium:", fmt("diastolic_myocardium"))
    print("Diastolic Blood Pool:", fmt("diastolic_blood_pool"))
    print("Systolic Myocardium: ", fmt("systolic_myocardium"))
    print("Systolic Blood Pool: ", fmt("systolic_blood_pool"))
    print("Average:             ", fmt("average"))

    return scores

def dice_for_patient_ed_es_kerasstyle(
    pid: str,
    masks_4d: np.ndarray,     # (H,W,S,F) int labels {0,1,2}
    preds_4d: np.ndarray,     # (H,W,S,F) int labels {0,1,2}
    frame_id_list,            # list of true H5 frame ids (len=F)
    ed_map: dict,             # {pid: ED_h5_id}
    es_map: dict,             # {pid: ES_h5_id or -1}
    class_mapping: dict,      # {"myocardium":1, "blood_pool":2}
    empty_empty_equals_one: bool = True,  # behavior when both GT and Pred are empty
    smooth: float = 1e-6
) -> dict:
    """
    Compute per-class Dice on the WHOLE short-axis stack (H,W,S) at ED and ES,
    using the same formula as your Keras metrics (but on hard labels).
    Returns a dict with keys:
      diastolic_myocardium, diastolic_blood_pool,
      systolic_myocardium,  systolic_blood_pool,
      average_diastolic,    average_systolic, average
    """
    out = {}
    # Map true H5 ids → positions inside this patient's volume
    ed_h5 = ed_map.get(pid, -1)
    es_h5 = es_map.get(pid, -1)
    ed_pos = _frame_pos_from_h5_id(ed_h5, frame_id_list)
    es_pos = _frame_pos_from_h5_id(es_h5, frame_id_list) if es_h5 is not None and es_h5 >= 0 else None

    def dice_for(class_id: int, frame_pos: int) -> float:
        y_true = (masks_4d[..., frame_pos] == class_id)
        y_pred = (preds_4d[..., frame_pos] == class_id)
        if empty_empty_equals_one and (not y_true.any()) and (not y_pred.any()):
            return 1.0
        return float(dice_keras_binary(y_true, y_pred, smooth=smooth))

    # ED
    if ed_pos is not None:
        for cname, cid in class_mapping.items():
            out[f"diastolic_{cname}"] = dice_for(cid, ed_pos)

    # ES
    if es_pos is not None:
        for cname, cid in class_mapping.items():
            out[f"systolic_{cname}"] = dice_for(cid, es_pos)

    # Averages (clean definition = mean of the four core numbers that exist)
    di = [out[k] for k in ("diastolic_myocardium", "diastolic_blood_pool") if k in out]
    sy = [out[k] for k in ("systolic_myocardium",  "systolic_blood_pool")  if k in out]
    if di: out["average_diastolic"] = float(np.mean(di))
    if sy: out["average_systolic"]  = float(np.mean(sy))
    core = di + sy
    if core: out["average"] = float(np.mean(core))
    return out

def _frame_pos_from_h5_id(h5_id: int, frame_id_list):
    """Return the index (0..F-1) of the H5 frame id in frame_id_list or None."""
    try:
        return frame_id_list.index(int(h5_id))
    except Exception:
        return None
    
def dice_keras_binary(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Keras-style Dice on binary masks (float, flattened):
      Dice = (2*sum(p*g) + eps) / (sum(p) + sum(g) + eps)
    y_true_bin, y_pred_bin: boolean or {0,1} arrays of identical shape.
    """
    y_true_f = y_true_bin.astype(np.float32).ravel()
    y_pred_f = y_pred_bin.astype(np.float32).ravel()
    intersection = np.dot(y_true_f, y_pred_f)
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
