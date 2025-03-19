from tensorflow.keras.models import load_model

model_path = "C:/Users/Sayli/Downloads/AI/Cat_Dog_Classifier/cat_vs_dog_model.keras"  # Update this if necessary
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
    # If you need to save it in .h5 format, you can do this:
    model.save("cat_vs_dog_model.h5")
    print("Model saved as cat_vs_dog_model.h5")
except Exception as e:
    print("Error loading model:", e)
