import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from open_clip import create_model, tokenize

image_path = "image3.jpg"

# Load the image using OpenCV
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
cv2.imshow("Uploaded Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# User prompt
text_prompt = input("Enter the text prompt to identify objects: ")

# Load YOLO model
model = YOLO("yolo11n.pt")
results = model(image_path)

# Extract detection results
detections = results[0].boxes.data.cpu().numpy()
boxes = detections[:, :4]
scores = detections[:, 4]
classes = detections[:, 5].astype(int)
class_names = model.names

if len(boxes) == 0:
    raise ValueError("No objects detected in the image!")

# Load CLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = create_model("ViT-B-32", pretrained="openai").to(device).eval()

def preprocess_pil_image(image, device):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return image_tensor


# Show all detected objects
image_with_boxes = image.copy()
for box in boxes:
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
cv2.imshow("All Detected Objects", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Process detections
filtered_boxes = []
selected_box = None
for i, box in enumerate(boxes):
    xmin, ymin, xmax, ymax = map(int, box)
    cropped_object = Image.fromarray(image[ymin:ymax, xmin:xmax])
    
    image_input = preprocess_pil_image(cropped_object, device)
    text_input = tokenize([text_prompt]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        similarity = torch.cosine_similarity(image_features, text_features).item()
    
    if similarity > 0.2:
        filtered_boxes.append((box, class_names[classes[i]], scores[i], similarity))
        if selected_box is None or similarity > selected_box[3]:
            selected_box = (box, class_names[classes[i]], scores[i], similarity)

# Extract and display the object most relevant to the prompt
def extract_with_padding(image, box, padding=20):
    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = map(int, box)
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(w, xmax + padding)
    ymax = min(h, ymax + padding)
    return image[ymin:ymax, xmin:xmax], (xmin, ymin)


def cnn_function(image):
    w, h, _ = image.shape
    return [[h/2, w/2], [h/2+5, w/2], [h/2-5, w/2], [h/2, w/2-5], [h/2, w/2+5]]


if selected_box:
    obj_img, top_left = extract_with_padding(image, selected_box[0])
    cv2.imshow("Extracted Object", obj_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # CNN HERE
    pixel_points = cnn_function(obj_img) 

    # Convert points to full image coordinates
    full_image_points = [(int(x + top_left[0]), int(y + top_left[1])) for x, y in pixel_points]

    # Display points on the full image
    for point in full_image_points:
        cv2.circle(image, point, 3, (0, 255, 0), -1)
    cv2.imshow("Detected Points on Full Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
