import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a function to create models with different configurations
def create_model(layers, activation, pooling):
    model = Sequential()
    
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 3)))
    
    for _ in range(layers-1):
        model.add(Conv2D(32, (3, 3), activation=activation))
        
    # Add pooling layer (MaxPooling or AveragePooling)
    if pooling == 'max':
        model.add(MaxPooling2D(pool_size=(2, 2)))
    elif pooling == 'average':
        model.add(AveragePooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Experiment with different configurations of the model
layer_counts = [2, 3, 4]
activations = ['relu', 'sigmoid']
poolings = ['max', 'average']

history_dict = {}

# Example of testing different combinations
for layers in layer_counts:
    for activation in activations:
        for pooling in poolings:
            print(f"Testing configuration: Layers={layers}, Activation={activation}, Pooling={pooling}")
            model = create_model(layers, activation, pooling)
            
            # Train the model
            history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
            
            # Store the history object for later analysis
            history_dict[(layers, activation, pooling)] = history

            # Plot accuracy for each configuration
            plt.plot(history.history['accuracy'], label=f'Train Accuracy ({layers} layers, {activation}, {pooling})')
            plt.plot(history.history['val_accuracy'], label=f'Val Accuracy ({layers} layers, {activation}, {pooling})')

# Finalize plot with proper labels and legend
plt.title('Model Accuracy for Different Configurations')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
