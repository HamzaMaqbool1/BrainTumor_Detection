
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array , ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16
from IPython.display import Markdown,display
base_dir="/content/drive/My Drive/BrainScans_extracted/brain scans"
train_dir=os.path.join(base_dir,"Training")
test_dir=os.path.join(base_dir,"Testing")
display(Markdown("<center><b><h1>Brain Tumor Detection<h1><b></center>"))

def show_images_by_folder(folder_path, images_per_row):
    trainingdata_classes = []
    folders_data = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return

    # Collect folders and images
    for folder in sorted(os.listdir(folder_path)):  # Sorted for consistent order
        folder_full_path = os.path.join(folder_path, folder)

        if os.path.isdir(folder_full_path):
            trainingdata_classes.append(folder)

            # Get all image files
            all_images = [f for f in os.listdir(folder_full_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            # Take first 'images_per_row' images
            selected_images = all_images[:images_per_row]

            if selected_images:
                folders_data.append((folder, folder_full_path, selected_images))
                print(f"✓ {folder}: {len(selected_images)} images found")
            else:
                print(f"✗ {folder}: No images found")

    if not folders_data:
        print("\nNo valid folders with images found!")
        return

    # Create plot
    num_folders = len(folders_data)
    fig, axes = plt.subplots(num_folders, images_per_row,
                             figsize=(images_per_row * 3, num_folders * 3))

    # Ensure axes is 2D
    if num_folders == 1:
        axes = axes.reshape(1, -1)

    # Display images row by row
    for row, (folder, folder_path, image_files) in enumerate(folders_data):
        for col in range(images_per_row):
            if col < len(image_files):
                img_path = os.path.join(folder_path, image_files[col])
                try:
                    img = load_img(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f"{image_files[col][:12]}", fontsize=7)
                except Exception as e:
                    axes[row, col].text(0.5, 0.5, 'Error loading',
                                       ha='center', va='center')
            else:
                # Empty cell
                axes[row, col].text(0.5, 0.5, 'No Image',
                                   ha='center', va='center', alpha=0.5)

            axes[row, col].axis('off')

        # Add folder name as row label
        axes[row, 0].set_ylabel(folder, fontsize=10, rotation=0,
                                ha='right', va='center', fontweight='bold')

    plt.suptitle(f'Training Dataset: {num_folders} Classes',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
show_images_by_folder(train_dir,5)

#Model Architecture
base_model=VGG16(weights="imagenet",include_top=False,input_shape=(244,244,3))
#ImageNet classification ussy off kr dea gya hai include_top=False q k VGG16 1.4million images of data py train hai or as a feature
#extractor use ho rha hai
for layer in base_model.layers:
  layer.trainable=False

model=Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

#ImageDataGeneratior and Augmentation
image_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

batch_size=20
epochs=10
train_generator=image_datagen.flow_from_directory(
    train_dir,
    target_size=(244,244),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

#Model training time
model_training=model.fit(
    train_generator,
    epochs=epochs,
)
#Save Model
model.save('/content/drive/My Drive/braintumor.h5')
drive='/content/drive/My Drive'
print(f"Model saved at drive:{drive}")

#Test data Generator
def image_predict(imagepath):
  img=load_img(imagepath,target_size=(244,244))
  img_array=img_to_array(img)
  img_array=np.expand_dims(img_array,axis=0)
  img_array/=255.0
  model=load_model('braintumor.h5')
  prediction=model.predict(img_array)
  class_labels=['glioma','meningioma','notumor','pituitary']
  predicted_class=class_labels[np.argmax(prediction)]
  plt.imshow(img)
  plt.axis('off')
  plt.title(f'Brain Image')
  plt.show()
  print(f"Model Statement:{predicted_class.capitalize()} has detected.")

image_predict("/content/drive/My Drive/BrainScans_extracted/brain scans/Testing/notumor/Te-no_0011.jpg")