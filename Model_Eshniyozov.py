import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Laden der Bildbeschreibungen
beschreibungen_path = 'C:\\Users\\elesh\\Desktop\\University\\PEiM\\Datensatz Seminararbeit WS23_24\\Datensatz\\Mikroskopbildbeschreibungen.xlsx'
df_beschreibungen = pd.read_excel(beschreibungen_path)

# Vorbereiten der Bildpfade und Labels
train_dir = "C:\\Users\\elesh\\Desktop\\University\\PEiM\\Datensatz Seminararbeit WS23_24\\Datensatz\\Trainingsdaten"
test_dir = "C:\\Users\\elesh\\Desktop\\University\\PEiM\\Datensatz Seminararbeit WS23_24\\Datensatz\\Testdatensatz"


# An dieser Stelle müssen Sie die df_beschreibungen anpassen, um sicherzustellen, dass sie mit den Bildpfaden übereinstimmen



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)




# Angenommen, 'bildpfad' ist die Spalte mit den Pfaden zu den Bildern
# und 'label' die Spalte mit den zugehörigen Labels
print(df_beschreibungen.head())




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Basis-Modell laden
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Neues Modell aufbauen
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(3, activation='softmax')  # 3 Neuronen für 3 Klassen, mit softmax Aktivierung
])

# Modell kompilieren
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])









from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialisieren Sie den ImageDataGenerator mit dem gewünschten Daten-Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    # Hier können Sie auch eine Aufteilung für Trainings- und Validierungsdaten angeben
)
df_beschreibungen['Klasse'] = df_beschreibungen['Klasse'].astype(str)
# Flow_from_dataframe (oder flow_from_directory) anpassen
train_generator = datagen.flow_from_dataframe(
    dataframe=df_beschreibungen,
    directory=None,  # oder Ihren spezifischen Pfad, falls notwendig
    x_col='Pfad',  # Ersetzen Sie dies durch Ihren tatsächlichen Spaltennamen für Bildpfade
    y_col='Klasse',  # Ersetzen Sie dies durch Ihren tatsächlichen Spaltennamen für Klassenlabels
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',  # Für Mehrklassenklassifikation
    batch_size=32,
    shuffle=True,
    # subset='training'  # Nur wenn Sie eine Validierungssplit verwenden
)





from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialisieren des ImageDataGenerator mit einem Validierungssplit von 20%
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% der Daten für die Validierung verwenden
)

# Trainingsdatengenerator
train_generator = datagen.flow_from_dataframe(
    dataframe=df_beschreibungen,
    x_col='Pfad',
    y_col='Klasse',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',  # Gibt an, dass dies der Trainingsdatensatz ist
    shuffle=True
)

# Validierungsdatengenerator
validation_generator = datagen.flow_from_dataframe(
    dataframe=df_beschreibungen,
    x_col='Pfad',
    y_col='Klasse',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Gibt an, dass dies der Validierungsdatensatz ist
    shuffle=True
)










print(train_generator.samples)
print(train_generator.batch_size)
print(validation_generator.samples)
print(validation_generator.batch_size)

validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)

print(validation_steps)









from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    min_delta=0.001,  # Minimum change to qualify as an improvement
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # To print messages when the callback takes action
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)









from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # oder 'softmax' für mehr als zwei Klassen
])

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Generatoren für das Training und die Validierung erstellen
# Hinweis: Verwenden Sie flow_from_directory oder eine ähnliche Methode, um Ihre Bilder zu laden

# model.fit(...) zum Trainieren des Modells










from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.regularizers import l2

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.001)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(3, activation='softmax')
])











from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])







history = model.fit(
    train_generator,
    steps_per_epoch=3,
    epochs=20,  # Adjust the number of epochs based on your needs
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping]  # Continue using early stopping
)







from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)






test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),  # Assuming this is the input size your model expects
    batch_size=32,  # Adjust based on your setup
    class_mode='categorical',  # Or 'binary' for binary classification
    shuffle=False  # Keep data in order to match the outputs with labels
)






import numpy as np


predictions = model.predict(test_generator, steps=np.ceil(test_generator.samples / test_generator.batch_size))







# Assuming your model is a classifier and outputs a probability distribution across classes
predicted_classes = np.argmax(predictions, axis=1)

# Assuming you have the actual class labels for the test set
# (You'd need to obtain these based on your test data setup)
actual_classes = test_generator.classes

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(actual_classes, predicted_classes)
print(f"Accuracy on test set: {accuracy:.2f}")











from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the image file
img_path = 'C:\\Users\\elesh\\Desktop\\University\\PEiM\\Datensatz Seminararbeit WS23_24\\Datensatz\\Trainingsdaten\\S141.jpg'
image = load_img(img_path, target_size=(224, 224))

# Convert the image to a numpy array and scale it
image = img_to_array(image) / 255.0

# Expand dimensions to match the model's input format
image = np.expand_dims(image, axis=0)





prediction = model.predict(image)



# Assuming you have a list of class names that correspond to the model's output
class_names = ['Defekt', 'Class2', 'Class3']  # Example class names

# Get the index of the highest probability
predicted_class_index = np.argmax(prediction)

# Find the class name using the index
predicted_class_name = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class_name}")









from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialisieren des ImageDataGenerator mit einem Validierungssplit von 20%
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% der Daten für die Validierung verwenden
)

# Trainingsdatengenerator
train_generator = datagen.flow_from_dataframe(
    dataframe=df_beschreibungen,
    x_col='Pfad',
    y_col='Klasse',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',  # Gibt an, dass dies der Trainingsdatensatz ist
    shuffle=True
)

# Validierungsdatengenerator
validation_generator = datagen.flow_from_dataframe(
    dataframe=df_beschreibungen,
    x_col='Pfad',
    y_col='Klasse',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # Gibt an, dass dies der Validierungsdatensatz ist
    shuffle=True
)



model.save('my_model.h5')








