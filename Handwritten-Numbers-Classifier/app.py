import os
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Function to preprocess the test image
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 (same size as MNIST digits)
    image = cv2.resize(image, (28, 28))

    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0

    # Add a batch dimension to the image
    image = image.reshape(1, 28, 28, 1)

    return image

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if an image was uploaded
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file.filename != "":
                # Save the uploaded image to a temporary location
                image_path = "temp_image.png"
                image_file.save(image_path)

                # Preprocess the test image
                preprocessed_image = preprocess_image(image_path)

                # Make predictions on the test image
                predictions = model.predict(preprocessed_image)

                # Get the predicted label (digit) from the model's output
                predicted_label = tf.argmax(predictions, axis=1)[0].numpy()

                # Remove the temporary image file
                os.remove(image_path)

                # Return the predicted label as JSON
                return jsonify({"label": str(predicted_label)})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
