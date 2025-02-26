import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from open_clip import create_model, tokenize

# Upload the image
image_path = 'fruits.PNG'

# Load the image using OpenCV
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
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
clip_model = create_model("ViT-B-32", pretrained="openai").to("cuda" if torch.cuda.is_available() else "cpu").eval()

def preprocess_pil_image(image, device):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return image_tensor

# Process detections
filtered_boxes = []
for i, box in enumerate(boxes):
    xmin, ymin, xmax, ymax = map(int, box)
    cropped_object = Image.fromarray(image[ymin:ymax, xmin:xmax])
    
    image_input = preprocess_pil_image(cropped_object, clip_model.device)
    text_input = tokenize([text_prompt]).to(clip_model.device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        similarity = torch.cosine_similarity(image_features, text_features).item()
    
    if similarity > 0.2:
        filtered_boxes.append((box, class_names[classes[i]], scores[i], similarity))

# Display results with OpenCV
for box, label, score, similarity in filtered_boxes:
    xmin, ymin, xmax, ymax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.putText(image, f"{label} ({score:.2f}, sim: {similarity:.2f})", (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv2.imshow("Filtered Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract object with padding
def extract_with_padding(image, box, padding=20):
    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = map(int, box)
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(w, xmax + padding)
    ymax = min(h, ymax + padding)
    return image[ymin:ymax, xmin:xmax]

# Extract and display cropped object
if filtered_boxes:
    obj_img = extract_with_padding(image, filtered_boxes[0][0])
    cv2.imshow("Extracted Object", obj_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
