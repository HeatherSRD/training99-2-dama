import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Title of the app
st.title("MNIST Digit Classifier Training")

# Description
st.write("""
This app trains a neural network on the MNIST dataset and displays the training process.
""")

# Load MNIST data
st.write("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Display a few sample images
st.write("Sample images from the dataset:")
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].axis('off')
st.pyplot(fig)

# Prepare labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model summary
st.write("Model Summary:")
model.summary(print_fn=lambda x: st.text(x))

# Train the model
epochs = st.slider("Select number of epochs:", 1, 10, 5)
st.write("Training the model...")

history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=2)

# Plot training history
st.write("Training and Validation Accuracy")
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Train Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
st.pyplot(fig)

# Display test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
st.write(f"Test Accuracy: {test_acc:.4f}")
