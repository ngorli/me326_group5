import torch
import numpy as np
import cv2
import speech_recognition as sr
from groundingdino.util.inference import load_model, predict

# Initialize video capture (replace with realsense feed)
cap = cv2.VideoCapture(1)

# Capture audio prompt (probably need to redo with whatever microphone robot has)
def get_audio_prompt():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say the name of the object:")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError:
        print("Error with the speech recognition service")
        return None
    

def cnn_function(image):
    w, h, _ = image.shape
    return [[h/2, w/2], [h/2+5, w/2], [h/2-5, w/2], [h/2, w/2-5], [h/2, w/2+5]]


# Specify paths for the Grounding DINO configuration and weights files.
config_path = "GroundingDINO_SwinT_OGC.py"  
weights_path = "groundingdino_swint_ogc.pth"   

model = load_model(config_path, weights_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 13:  # Press Enter to capture frame 
        image = frame.copy()
        # text_prompt = get_audio_prompt()
        text_prompt = 'Calendar'
        if not text_prompt:
            print("No valid audio input received.")
            continue

        # Perform object detection with Grounding DINO
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

        boxes, logits, phrases = predict(
                                model=model,
                                image=image,
                                caption=text_prompt,
                                box_threshold=BOX_TRESHOLD,
                                text_threshold=TEXT_TRESHOLD
                                )

        if len(boxes) == 0:
            print("No objects detected in the image!")
            continue

        # Process detections: choose the detection with the highest logit score.
        filtered_boxes = []
        selected_box = None
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            filtered_boxes.append((box, phrases[i], logits[i]))
            if selected_box is None or logits[i] > selected_box[2]:
                selected_box = (box, phrases[i], logits[i])

        # Display all detected objects on the captured frame.
        for box, label, _ in filtered_boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Detected Objects", image)
        cv2.waitKey(0)

        # Function to extract an object with some padding
        def extract_with_padding(image, box, padding=20):
            h, w, _ = image.shape
            xmin, ymin, xmax, ymax = map(int, box)
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(w, xmax + padding)
            ymax = min(h, ymax + padding)
            return image[ymin:ymax, xmin:xmax], (xmin, ymin)

        # Extract and display the object most relevant to the prompt.
        if selected_box:
            obj_img, top_left = extract_with_padding(image, selected_box[0])
            cv2.imshow("Extracted Object", obj_img)
            cv2.waitKey(0)

            # Process the cropped image
            pixel_points = cnn_function(obj_img) 

            # Convert points from cropped image coordinates to full image coordinates.
            full_image_points = [(x + top_left[0], y + top_left[1]) for x, y in pixel_points]

            # Display points on the full image.
            for point in full_image_points:
                cv2.circle(image, point, 3, (0, 255, 0), -1)
            cv2.imshow("Detected Points on Full Image", image)
            cv2.waitKey(0)
    
cv2.destroyAllWindows()
cap.release()
