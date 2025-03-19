from tensorflow.keras.models import load_model

# Path to the .keras model
model_path = "C:/Users/Sayli/Downloads/AI/Cat_Dog_Classifier/cat_vs_dog_model.keras"

try:
    # Load the .keras model
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Save it as .h5
    model.save("C:/Users/Sayli/Downloads/AI/Cat_Dog_Classifier/cat_vs_dog_model.h5")
    print("Model saved as cat_vs_dog_model.h5")
    
except Exception as e:
    print("Error loading model:", e)
