import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import yaml
import shutil
import random
import numpy as np

# Paths
dataset_path = 'C:\\Users\\iqbal\\Downloads\\Sign-Language-Detection_BISINDO\\bisindo'
yaml_file = 'bisindo.yaml'
model_save_path = 'best_bisindo_model.pt'
evaluation_save_path = 'evaluation_results'


# Step 4: Preprocess Image Data (Create train/val/test splits and move labels)
def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.1):
    output_image_dir = os.path.join(dataset_dir, output_dir, 'images')
    output_label_dir = os.path.join(dataset_dir, output_dir, 'labels')

    # Remove previous split if exists
    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)
    if os.path.exists(output_label_dir):
        shutil.rmtree(output_label_dir)

    # Helper function to move images and labels

    def move_data(images, split, folder_name):
        split_image_dir = os.path.join(output_image_dir, split, os.path.basename(
            folder_name))  # Use only the folder's name
        split_label_dir = os.path.join(output_label_dir, split, os.path.basename(
            folder_name))  # Use only the folder's name
        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)

        for img in images:
            src_img_path = os.path.join(folder_name, img)
            dst_img_path = os.path.join(split_image_dir, img)
            if src_img_path != dst_img_path:  # Avoid copying to the same location
                shutil.copy(src_img_path, dst_img_path)

            # Handle label file
            label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label_path = os.path.join(folder_name, label_file)
            dst_label_path = os.path.join(split_label_dir, label_file)
            # Avoid copying to the same location
            if os.path.exists(src_label_path) and src_label_path != dst_label_path:
                shutil.copy(src_label_path, dst_label_path)

    # Process Alphabets
    alphabets_dir = os.path.join(dataset_dir, 'Alphabets')
    for letter in os.listdir(alphabets_dir):
        letter_dir = os.path.join(alphabets_dir, letter)
        if os.path.isdir(letter_dir):
            images = [f for f in os.listdir(
                letter_dir) if f.endswith(('.jpg', '.png'))]
            random.shuffle(images)
            train_split = int(len(images) * train_ratio)
            val_split = int(len(images) * (train_ratio + val_ratio))

            train_images = images[:train_split]
            val_images = images[train_split:val_split]
            test_images = images[val_split:]

            for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
                move_data(split_images, split, letter_dir)

    # Process Words (Handle all frames of the same video together)
    words_dir = os.path.join(dataset_dir, 'Words')
    for word in os.listdir(words_dir):
        word_dir = os.path.join(words_dir, word)
        if os.path.isdir(word_dir):
            video_frames = {}
            # Group frames by video
            for img in os.listdir(word_dir):
                if img.endswith(('.jpg', '.png')):
                    # Get the base name of the video
                    video_name = "_".join(img.split("_")[:-1])
                    if video_name not in video_frames:
                        video_frames[video_name] = []
                    video_frames[video_name].append(img)

            # Split videos
            videos = list(video_frames.keys())
            random.shuffle(videos)
            train_split = int(len(videos) * train_ratio)
            val_split = int(len(videos) * (train_ratio + val_ratio))

            train_videos = videos[:train_split]
            val_videos = videos[train_split:val_split]
            test_videos = videos[val_split:]

            for split, split_videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
                for video in split_videos:
                    move_data(video_frames[video], split, word_dir)


# Step 5: Setup YAML file for YOLOv8 with both Alphabets and Words
def setup_yaml(yaml_file_path, dataset_dir):
    names_dict = {i: chr(65 + i) for i in range(26)}  # A-Z classes
    words_folder = os.path.join(dataset_dir, 'Words')

    word_classes = {}
    current_class_index = 26  # Continue numbering after Z (25)

    for word in os.listdir(words_folder):
        if os.path.isdir(os.path.join(words_folder, word)):
            word_classes[current_class_index] = word
            current_class_index += 1

    names_dict.update(word_classes)

    yaml_content = {
        'path': dataset_dir,
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'test': os.path.join('images', 'test'),
        'names': names_dict,
        'nc': len(names_dict)
    }

    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file)

# Step 6: Train YOLOv8 Model


def train_yolo_model(yaml_file, model_save_path, epochs=50, batch_size=16):
    model = YOLO('yolov8n.pt')
    results = model.train(data=yaml_file, epochs=epochs,
                          batch=batch_size, imgsz=640, verbose=True)

    model.save(model_save_path)
    return model

# Step 7: Evaluate YOLOv8 Model and Save Evaluation Results


def evaluate_model(model, yaml_file, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    results = model.val(data=yaml_file)

    y_true = results['val_labels']
    y_pred = results['val_preds']

    class_names = [chr(65 + i) for i in range(26)] + \
        list(model.names.values())[26:]
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)

    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(model.names))
    plt.xticks(tick_marks, model.names.values(), rotation=45)
    plt.yticks(tick_marks, model.names.values())
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{class_report}")

    with open(os.path.join(save_path, 'classification_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['train_acc'], label='Training Accuracy')
    plt.plot(results['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()

    plt.subplot(1, 2, 2)
    plt.plot(results['train_loss'], label='Training Loss')
    plt.plot(results['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

    model.save(os.path.join(save_path, 'model_final.pt'))


def main():
    # Step 3: Preprocess Image Data and split into train/val/test
    split_dataset(dataset_path, 'images')

    # Step 4: Setup YAML file for YOLOv8
    setup_yaml(yaml_file, dataset_path)

    # Step 5: Train YOLOv8 Model
    model = train_yolo_model(yaml_file, model_save_path)

    # Step 6: Evaluate YOLOv8 Model and Save Results
    evaluate_model(model, yaml_file, evaluation_save_path)


# Run the main function
if __name__ == '__main__':
    main()
