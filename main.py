'''
For Local Deployment
'''
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import matplotlib.patches as patches
from ultralytics import SAM
import numpy as np
import argparse
from pathlib import Path
import csv
import json



def resolve_device(device):
    """Choose Apple Silicon MPS, CUDA, or CPU when --device auto."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# STEP 1: Animal Instance Segmentation
# Detect and segment individual animals using YOLO.
def step1_animal_mask_box(image_path,yolo_seg_path):
    # Load Yolo
    model = YOLO(yolo_seg_path)


    animal_classes = list(range(14, 24))

    # Yolo Inference
    results = model.predict(
        source=image_path,
        conf=0.2,
        iou=0.5,
        agnostic_nms=True,
        nms=True,
        classes=animal_classes,
        device="mps",
        verbose=False
    )

    result = results[0]

    return result

# STEP 1.5: Animal ROI Extraction
# Crop each animal and remove background interference.
def animial_ROI(image_path, result):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    H, W = image_rgb.shape[:2]
    animal_ROI_crops = []

    if result.masks is None:
        return animal_ROI_crops

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy()

    for i, (box, cls, mask) in enumerate(
        zip(boxes, classes, masks)
    ):
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Resize instance mask to original resolution
        if mask.shape != (H, W):
            mask = cv2.resize(
                mask.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

        binary_mask = mask > 0.5

        # Crop image and corresponding instance mask
        crop = image_rgb[y1:y2, x1:x2].copy()
        crop_mask = binary_mask[y1:y2, x1:x2]

        # Replace background with neutral gray
        masked_crop = np.full_like(crop, 127)
        masked_crop[crop_mask] = crop[crop_mask]

        animal_ROI_crops.append({
            "image": masked_crop,
            "original_crop": crop,
            "mask": crop_mask,
            "offset": (x1, y1),
            "class_id": int(cls)
        })

    return animal_ROI_crops


# STEP 2: Eye Detection
# Use Grounding DINO to detect eye bounding boxes.
def step2_animal_eye_box(animal_ROI_crops):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    )

    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)

    dino.eval()

    all_eye_detections = []
    for ani_idx, animal in enumerate(animal_ROI_crops):
        pil_image = Image.fromarray(animal["image"])

        inputs = processor(
            images=pil_image,
            text="Animal two eye.",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = dino(**inputs)

        detections = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[pil_image.size[::-1]]
        )[0]
        all_eye_detections.append(detections)


        print(
            f"Animal {ani_idx+1}: "
            f"{len(detections['boxes'])} eye boxes detected"
        )

    return all_eye_detections


# STEP 2.5: Geometry-based Filtering
# Remove oversized eye candidates using geometric constraints.
def filter_eye_boxes(boxes, scores, image_shape,
                     max_area_ratio=0.10,
                     max_width_ratio=0.40,
                     max_height_ratio=0.35):

    h, w = image_shape[:2]

    valid_boxes = []
    valid_scores = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1

        area_ratio = (bw * bh) / (w * h)
        width_ratio = bw / w
        height_ratio = bh / h

        # Filter large boxes
        if area_ratio > max_area_ratio:
            continue

        if width_ratio > max_width_ratio:
            continue

        if height_ratio > max_height_ratio:
            continue

        valid_boxes.append(box)
        valid_scores.append(score)

    return (
        np.asarray(valid_boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(valid_scores, dtype=np.float32)
    )


def filter_all_animal_eye_boxes(all_eye_detections, animal_ROI_crops):

    filtered_detections = []

    for ani_idx, animal in enumerate(animal_ROI_crops):

        detections = all_eye_detections[ani_idx]

        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()

        # Geometry filter
        valid_boxes, valid_scores = filter_eye_boxes(
            boxes=boxes,
            scores=scores,
            image_shape=animal["image"].shape,
            max_area_ratio=0.10,
            max_width_ratio=0.40,
            max_height_ratio=0.35
        )

        # Convert back to torch.Tensor
        filtered = detections.copy()

        filtered["boxes"] = torch.as_tensor(
            valid_boxes,
            dtype=detections["boxes"].dtype,
            device=detections["boxes"].device
        )

        filtered["scores"] = torch.as_tensor(
            valid_scores,
            dtype=detections["scores"].dtype,
            device=detections["scores"].device
        )

        # Keep labels aligned with filtered boxes
        # （如果有使用 labels，建議原 filter 直接回傳保留的 indices）
        filtered.pop("labels", None)
        filtered.pop("text_labels", None)

        filtered_detections.append(filtered)

        print(
            f"Animal {ani_idx+1}: "
            f"{len(boxes)} → {len(valid_boxes)} eye candidates"
        )

    return filtered_detections

# STEP 3: Eye Segmentation
# Use SAM 2 to generate binary eye masks.
def step3_animal_eye_mask(detections, animal_ROI_crops):
    sam = SAM("models/sam2.1_t.pt")

    all_eye_masks = []

    for ani_idx, animal in enumerate(animal_ROI_crops):

        img = animal["original_crop"]

        boxes = detections[ani_idx]["boxes"].cpu().numpy()
        scores = detections[ani_idx]["scores"].cpu().numpy()

        # Select top 2 eye boxes
        top2 = np.argsort(scores)[::-1][:2]
        eye_boxes = boxes[top2]

        if len(eye_boxes) != 2:
            print(f"Animal {ani_idx+1}: Expected 2 eye boxes")
            all_eye_masks.append([])
            continue

        results_sam = sam.predict(
            source=img,
            bboxes=eye_boxes.tolist(),
            verbose=False
        )

        if results_sam[0].masks is None:
            all_eye_masks.append([])
            continue

        eye_masks = results_sam[0].masks.data.cpu().numpy()

        all_eye_masks.append(eye_masks)

    return all_eye_masks





# STEP 4: Eye Landmark Extraction
# Calculate mask centroids and restore original image coordinates.
def eye_mask_centroid(all_eye_masks, animal_ROI_crops, image_path):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    # 儲存所有動物的眼睛資料
    all_eye_landmarks = []
    all_eye_global_masks = []

    for ani_idx, animal in enumerate(animal_ROI_crops):

        crop = animal["image"]
        x_offset, y_offset = animal["offset"]

        crop_h, crop_w = crop.shape[:2]

        masks = all_eye_masks[ani_idx]

        if len(masks) != 2:
            print(f"animal {ani_idx+1}: Expected 2 eye masks")
            all_eye_landmarks.append(None)
            all_eye_global_masks.append(None)
            continue

        eye_data = []

        for mask in masks:

            # Resize mask to crop resolution if necessary
            if mask.shape != (crop_h, crop_w):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                )

            mask = mask > 0.5

            # Calculate local centroid
            ys, xs = np.where(mask)

            if len(xs) == 0:
                continue

            cx_local = xs.mean()
            cy_local = ys.mean()

            # Convert to original image coordinates
            cx_global = cx_local + x_offset
            cy_global = cy_local + y_offset

            # Restore mask to original image
            global_mask = np.zeros((H, W), dtype=np.uint8)

            global_mask[
                y_offset:y_offset+crop_h,
                x_offset:x_offset+crop_w
            ] = mask.astype(np.uint8)

            eye_data.append({
                "centroid": (cx_global, cy_global),
                "mask": global_mask
            })

        if len(eye_data) != 2:
            all_eye_landmarks.append(None)
            all_eye_global_masks.append(None)
            continue

        # Sort by X coordinate (front-facing dog)
        eye_data.sort(key=lambda e: e["centroid"][0])

        # Anatomical orientation
        right_eye = eye_data[0]
        left_eye = eye_data[1]

        all_eye_landmarks.append({
            "right_eye": right_eye["centroid"],
            "left_eye": left_eye["centroid"]
        })

        all_eye_global_masks.append({
            "right_eye": right_eye["mask"],
            "left_eye": left_eye["mask"]
        })

        print(f"Animal {ani_idx+1}")
        print(" Right eye:", right_eye["centroid"])
        print(" Left eye:", left_eye["centroid"])

    return all_eye_landmarks, all_eye_global_masks


def distance_between_points(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

# STEP 5: Geometric Measurement
# Calculate inter-eye and between-animal right-eye distances.
def get_eye_distance(all_eye_landmarks):

    # SAM 2 predicted eye centroids
    
    valid_animals = {
        i + 1: landmarks
        for i, landmarks in enumerate(all_eye_landmarks)
        if landmarks is not None
    }

    results = {
        "inter_eye_distances": {},
        "between_right_eyes": {}
    }

    # 每隻成功偵測的動物：左右眼距離
    for animal_id, landmarks in valid_animals.items():

        distance = distance_between_points(
            landmarks["right_eye"],
            landmarks["left_eye"]
        )

        results["inter_eye_distances"][animal_id] = distance

    # 任意兩隻動物：右眼距離
    from itertools import combinations

    for (id1, animal1), (id2, animal2) in combinations(
        valid_animals.items(), 2
    ):
        distance = distance_between_points(
            animal1["right_eye"],
            animal2["right_eye"]
        )

        results["between_right_eyes"][
            f"Animal {id1} - Animal {id2}"
        ] = distance

    return results


# STEP 6: Visualization
# Display detection, segmentation, landmarks, and measurements.
def plot_all(image_path, result, animal_ROI_crops, animal_eye_box, animal_eye_masks, all_eye_landmarks, all_eye_global_masks):


    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ==============================================
    # 1. Original Image
    # ==============================================
    plt.figure(figsize=(9, 10))
    plt.imshow(image_rgb)
    plt.title("1. Original COCO Image")
    plt.axis("off")
    plt.show()

    # ==============================================
    # 2. YOLO Animal Instance Segmentation
    # ==============================================
    annotated = result.plot(
        boxes=True,
        masks=True,
        labels=True
    )

    plt.figure(figsize=(9, 10))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("2. YOLO Animal Instance Segmentation")
    plt.axis("off")
    plt.show()

    # ==============================================
    # 3. Grounding DINO Eye Detection
    # ==============================================
    n_animals = len(animal_ROI_crops)

    fig, axes = plt.subplots(
        1, n_animals,
        figsize=(6 * n_animals, 7),
        squeeze=False
    )

    for ani_idx, animal in enumerate(animal_ROI_crops):

        ax = axes[0, ani_idx]
        img = animal["image"]

        detections = animal_eye_box[ani_idx]

        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()

        # Select top 2
        top2 = np.argsort(scores)[::-1][:2]

        ax.imshow(img)

        for box, score in zip(boxes[top2], scores[top2]):

            x1, y1, x2, y2 = box

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                x1, y1 - 5,
                f"Eye {score:.2f}",
                color="yellow",
                fontsize=10,
                weight="bold"
            )

        ax.set_title(
            f"3. Animal {ani_idx+1} - Eye Detection"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ==============================================
    # 4. SAM 2 Eye Segmentation
    # ==============================================
    fig, axes = plt.subplots(
        1, n_animals,
        figsize=(6 * n_animals, 7),
        squeeze=False
    )

    for ani_idx, animal in enumerate(animal_ROI_crops):

        ax = axes[0, ani_idx]
        img = animal["image"]
        masks = animal_eye_masks[ani_idx]

        ax.imshow(img)

        for mask in masks:

            # Match mask resolution with crop
            h, w = img.shape[:2]

            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            ax.imshow(
                np.ma.masked_where(mask <= 0.5, mask),
                cmap="spring",
                alpha=0.5
            )

        ax.set_title(
            f"4. Animal {ani_idx+1} - SAM 2 Eye Masks"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ==============================================
    # 5. Eye Centroids + Distance Measurement
    # ==============================================
    fig, ax = plt.subplots(figsize=(10, 12))

    ax.imshow(image_rgb)

    colors = {
        "right_eye": "lime",
        "left_eye": "cyan"
    }

    # Plot Global Eye Masks and Centroids
    for ani_idx, (landmarks, masks) in enumerate(
        zip(all_eye_landmarks, all_eye_global_masks)
    ):

        if landmarks is None or masks is None:
            continue

        for eye_name in ["right_eye", "left_eye"]:

            mask = masks[eye_name]
            cx, cy = landmarks[eye_name]

            # Mask overlay
            ax.imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="spring",
                alpha=0.45
            )

            # Centroid
            ax.scatter(
                cx, cy,
                color=colors[eye_name],
                edgecolors="black",
                s=65,
                zorder=5
            )

            ax.text(
                cx + 5,
                cy - 8,
                f"A{ani_idx+1} {eye_name}",
                color=colors[eye_name],
                fontsize=9,
                weight="bold"
            )

    # ----------------------------------------------
    # Distance lines
    # ----------------------------------------------
    valid_animals = [
        (i, lm)
        for i, lm in enumerate(all_eye_landmarks)
        if lm is not None
    ]

    measurement_pairs = []

    # Each animal's inter-eye distance
    for ani_idx, landmarks in valid_animals:

        measurement_pairs.append((
            landmarks["right_eye"],
            landmarks["left_eye"],
            f"Animal {ani_idx+1}",
            "yellow"
        ))

    # Between first two valid animals' right eyes
    if len(valid_animals) >= 2:

        idx1, ani1 = valid_animals[0]
        idx2, ani2 = valid_animals[1]

        measurement_pairs.append((
            ani1["right_eye"],
            ani2["right_eye"],
            f"A{idx1+1}-A{idx2+1} Right Eyes",
            "red"
        ))

    for p1, p2, label, color in measurement_pairs:

        d = distance_between_points(p1, p2)

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=2,
            linestyle="--" if color == "red" else "-"
        )

        mid = (np.array(p1) + np.array(p2)) / 2

        ax.text(
            mid[0] + 5,
            mid[1] - 10,
            f"{label}: {d:.2f} px",
            color=color,
            fontsize=10,
            weight="bold",
            bbox=dict(
                facecolor="black",
                alpha=0.5,
                edgecolor="none"
            )
        )

    ax.set_title(
        "5. Eye Segmentation, Centroids & Metrology"
    )

    ax.axis("off")
    plt.tight_layout()
    plt.show()



# Put all together
def get_animal_eye_point_distance_solution1(
                            image_path = "images/000000204800.jpg",
                            yolo_seg_path = "models/yolo26n-seg.pt",
                            sam_model_path = "models/sam2.1_t.pt",
                            plot = False
                            ):

    result = step1_animal_mask_box(image_path, yolo_seg_path)

    animal_ROI_crops = animial_ROI(image_path, result)

    animal_eye_box = step2_animal_eye_box(animal_ROI_crops)

    animal_eye_box = filter_all_animal_eye_boxes(
                        animal_eye_box,
                        animal_ROI_crops
                    )

    animal_eye_masks = step3_animal_eye_mask(animal_eye_box,animal_ROI_crops)

    all_eye_landmarks, all_eye_global_masks = eye_mask_centroid(animal_eye_masks, animal_ROI_crops, image_path)

    results = get_eye_distance(all_eye_landmarks)


    if plot:
        plot_all(image_path, result,animal_ROI_crops, animal_eye_box, animal_eye_masks, all_eye_landmarks, all_eye_global_masks)
        print(results)

    return {
        "animal_result": result,
        "animal_crops": animal_ROI_crops,
        "eye_detections": animal_eye_box,
        "eye_masks": animal_eye_masks,
        "eye_landmarks": all_eye_landmarks,
        "global_eye_masks": all_eye_global_masks,
        "distances": results
    }



def save_results(outputs, image_path, output_dir):
    """Export measurement results and an annotated preview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    landmarks = outputs["eye_landmarks"]
    distances = outputs["distances"]
    animal_result = outputs["animal_result"]

    payload = {
        "image": str(image_path),
        "unit": "px",
        "assumption": "right_eye is the image-left eye of a near-frontal animal",
        "animals": [],
        "inter_eye_distances": distances["inter_eye_distances"],
        "between_right_eyes": distances["between_right_eyes"],
    }
    for idx, lm in enumerate(landmarks, 1):
        cls_id = outputs["animal_crops"][idx-1]["class_id"]
        class_name = animal_result.names[int(cls_id)]
        payload["animals"].append({
            "animal_id": idx,
            "class_name": class_name,
            "right_eye": list(lm["right_eye"]) if lm else None,
            "left_eye": list(lm["left_eye"]) if lm else None,
            "inter_eye_distance_px": distances["inter_eye_distances"].get(idx),
            "status": "success" if lm else "eye_detection_failed",
        })

    json_path = output_dir / f"{stem}_measurements.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / f"{stem}_distances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "measurement_type", "animal_1", "animal_2", "distance_px"])
        for animal_id, distance in distances["inter_eye_distances"].items():
            writer.writerow([stem, "inter_eye", animal_id, "", f"{distance:.4f}"])
        for pair, distance in distances["between_right_eyes"].items():
            ids = pair.replace("Animal ", "").split(" - ")
            writer.writerow([stem, "between_right_eyes", ids[0], ids[1], f"{distance:.4f}"])

    # Render one non-interactive final-image preview.
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(original)
    for idx, lm in enumerate(landmarks, 1):
        if lm is None:
            continue
        right, left = lm["right_eye"], lm["left_eye"]
        ax.plot([right[0], left[0]], [right[1], left[1]], "y-", linewidth=2)
        ax.scatter([right[0]], [right[1]], c="lime", s=40, edgecolors="black")
        ax.scatter([left[0]], [left[1]], c="cyan", s=40, edgecolors="black")
        ax.text(right[0] + 5, right[1] - 7,
                f"A{idx} R | {distances['inter_eye_distances'][idx]:.1f} px",
                color="yellow", fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"})
    from itertools import combinations
    valid = [(i, lm) for i, lm in enumerate(landmarks, 1) if lm is not None]
    for (id1, a), (id2, b) in combinations(valid, 2):
        p1, p2 = a["right_eye"], b["right_eye"]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", alpha=0.65)
    ax.set_title("Animal Eye Metrology (px)")
    ax.axis("off")
    fig.tight_layout()
    figure_path = output_dir / f"{stem}_preview.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return json_path, csv_path, figure_path





def main():
    parser = argparse.ArgumentParser(description="COCO Animal Eye Segmentation & Metrology CLI")
    parser.add_argument("--image", required=True, help="Path to a COCO image (JPG/PNG)")
    parser.add_argument("--yolo", default="models/yolo26n-seg.pt", help="YOLO segmentation weights")
    parser.add_argument("--sam", default="models/sam2.1_t.pt", help="SAM 2.1 weights")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--output", default="results", help="Directory for CSV, JSON, preview")
    parser.add_argument("--yolo-conf", type=float, default=0.20)
    parser.add_argument("--yolo-iou", type=float, default=0.50)
    parser.add_argument("--dino-box-threshold", type=float, default=0.20)
    parser.add_argument("--dino-text-threshold", type=float, default=0.20)
    parser.add_argument("--prompt", default="Animal two eye.")
    parser.add_argument("--plot", action="store_true", help="Also show the original multi-figure plots")
    args = parser.parse_args()

    for kind, path in [("Image", args.image), ("YOLO model", args.yolo), ("SAM model", args.sam)]:
        if not Path(path).is_file():
            parser.error(f"{kind} file not found: {path}")

    print(f"[INFO] Image: {args.image} | Device: {resolve_device(args.device)}")
    outputs = get_animal_eye_point_distance_solution1(
        image_path=args.image,
        yolo_seg_path=args.yolo,
        sam_model_path=args.sam,
        # device=args.device,
        # yolo_conf=args.yolo_conf,
        # yolo_iou=args.yolo_iou,
        # dino_box_threshold=args.dino_box_threshold,
        # dino_text_threshold=args.dino_text_threshold,
        # prompt=args.prompt,
        plot=args.plot,
    )
    json_file, csv_file, image_file = save_results(outputs, args.image, args.output)
    print(f"[RESULT] Animals: {len(outputs['animal_crops'])}; usable eye pairs: "
          f"{len(outputs['distances']['inter_eye_distances'])}")
    for animal_id, distance in outputs['distances']['inter_eye_distances'].items():
        print(f"  Animal {animal_id} inter-eye distance: {distance:.2f} px")
    for pair, distance in outputs['distances']['between_right_eyes'].items():
        print(f"  {pair} right-eye distance: {distance:.2f} px")
    print(f"[SAVED] {json_file}\n[SAVED] {csv_file}\n[SAVED] {image_file}")


if __name__ == "__main__":
    main()
