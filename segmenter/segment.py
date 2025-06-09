import cv2
import os

def segment_pieces(image_path, output_dir="segmenter/crops", max_pieces=5):
    os.makedirs(output_dir, exist_ok=True)

    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Cleaner binary segmentation
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} total contours")

    # Sort and keep only top N
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_pieces]

    if not contours:
        print("No contours found.")
        return

    largest_area = cv2.contourArea(contours[0])
    min_area = largest_area / 10
    print(f"Largest contour area: {int(largest_area)} → Area cutoff: {int(min_area)}")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        print(f"Contour {i}: area={int(area)}, width={w}, height={h}")

        if area < min_area or w < 50 or h < 50:
            print(f"→ Skipping contour {i} (too small)")
            continue

        piece = original[y:y+h, x:x+w]
        print(f"→ Saving piece_{i}.jpg")
        cv2.imwrite(os.path.join(output_dir, f"piece_{i}.jpg"), piece)

if __name__ == "__main__":
    input_image = "segmenter/test_images/test_image_6.jpg"
    segment_pieces(input_image)
