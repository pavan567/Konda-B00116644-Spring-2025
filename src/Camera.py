from tensorflow.keras.models import model_from_json
import cv2
import numpy as np

# Load model architecture from JSON
with open("model.json", "r") as json_file:
    loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)

# Load model weights
model.load_weights("model_weights.h5")

# Define class labels
plant_classes = ['Aloevera','Amla','Amruta_Balli','Arali','Ashoka','Ashwagandha','Avacado','Bamboo','Basale','Betel','Betel_Nut','Brahmi','Castor',
               'Curry_Leaf','Doddapatre','Ekka','Ganike','Gauva','Geranium',
               'Henna','Hibiscus','Honge','Insulin','Jasmine','Lemon','Lemon_grass',
               'Mango','Mint','Nagadali','Neem','Nithyapushpa','Nooni','Papaya',
               'Pepper','Pomegranate','Raktachandini','Rose','Sapota','Tulasi',
               'Wood_sorel']
confidence_threshold = 0.6  # Confidence threshold to detect plants reliably

def image_predict(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]  # Get the first prediction
    class_probabilities = {plant_classes[i]: float(prediction[i]) for i in range(len(plant_classes))}
    
    
    # Get the most confident class
    max_prob = np.max(prediction)
    predicted_class = plant_classes[np.argmax(prediction)]

    # If max confidence is below threshold, return "No Plant Detected"
    if max_prob < confidence_threshold:
        return "No Plant Detected", max_prob, class_probabilities

    return predicted_class, max_prob, class_probabilities


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.default_text = "Waiting for Detection..."  # Start with a neutral message
    
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        if not ret or frame is None:
            # If no frame is captured, return a default message
            self.default_text = "No Camera Feed"
        else:
            try:
                resized_frame = cv2.resize(frame, (224, 224))
                predicted_plant, max_prob, class_probabilities = image_predict(resized_frame)
                
                # Prevent tulsi from being default at start
                if self.default_text == "Waiting for Detection..." and predicted_plant == "tulsi":
                    predicted_plant = "No Plant Detected"

                self.default_text = predicted_plant
                
                # Display the predicted class
                cv2.putText(frame, f"Predicted: {predicted_plant} ({max_prob:.2f})", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display probabilities of all classes
                y_offset = 80
                for plant, prob in class_probabilities.items():
                    cv2.putText(frame, f"{plant}: {prob:.2f}", (50, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 30  # Move text down for next class
                
            except Exception as e:
                print(f"Error: {e}")

        resized_img = cv2.resize(frame, (800, 600))
        _, jpeg = cv2.imencode('.jpg', resized_img)
        return jpeg.tobytes()
