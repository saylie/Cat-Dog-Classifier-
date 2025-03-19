import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Step 1: Load the trained model
model = tf.keras.models.load_model('cat_vs_dog_model.keras')

# Step 2: Load and preprocess the image
img_path = r'C:\Users\Sayli\Downloads\AI\cat_dog_dataset\test\Cat\girl.jpg'  # Update the path
img = image.load_img(img_path, target_size=(128, 128))  # Resize to match model input size

# Convert to array and normalize
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)/255  # Normalize

# Step 3: Make a prediction
prediction = model.predict(img_array)

# Step 4: Print prediction probability
print(f"Prediction score: {prediction[0][0]}")  # Print probability score

# Step 5: Interpret the result
if prediction[0][0] > 0.5:
    print("This is a Dog!")
else:
    print("This is a Cat!")
