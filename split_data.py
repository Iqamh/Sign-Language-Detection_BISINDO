import os
import cv2

# Paths
dataset_path = 'C:\\Users\\iqbal\\Downloads\\Sign-Language-Detection_BISINDO\\bisindo'

# Step 1: Extract frames from videos and save as images


def extract_frames(video_path, output_folder, frame_per_second=0.25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_per_second)

    video_name = os.path.basename(video_path).split('.')[0]
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            output_path = os.path.join(
                output_folder, f"{video_name}_frame{count}.jpg")
            cv2.imwrite(output_path, frame)
            count += 1

        frame_count += 1

    cap.release()

# Step 2: Process Video Data


def process_video_data(dataset_dir):
    words_dir = os.path.join(dataset_dir, 'Words')
    for word in os.listdir(words_dir):
        word_dir = os.path.join(words_dir, word)
        if os.path.isdir(word_dir):
            for file in os.listdir(word_dir):
                if file.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(word_dir, file)
                    extract_frames(video_path, word_dir)


# Step 3: Create label files (.txt)
def create_labels(dataset_dir):
    # Process Alphabets
    alphabets_dir = os.path.join(dataset_dir, 'Alphabets')
    for letter in os.listdir(alphabets_dir):
        letter_dir = os.path.join(alphabets_dir, letter)
        if os.path.isdir(letter_dir):
            # 'A' = 0, 'B' = 1, ..., 'Z' = 25
            class_id = ord(letter.upper()) - 65
            images = [f for f in os.listdir(
                letter_dir) if f.endswith(('.jpg', '.png'))]

            for img in images:
                label_path = os.path.join(letter_dir, img.replace(
                    '.jpg', '.txt').replace('.png', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"{class_id}\n")

    # Process Words
    words_dir = os.path.join(dataset_dir, 'Words')
    current_class_id = 26  # Start from 26 after letters
    for word in os.listdir(words_dir):
        word_dir = os.path.join(words_dir, word)
        if os.path.isdir(word_dir):
            images = [f for f in os.listdir(
                word_dir) if f.endswith(('.jpg', '.png'))]

            for img in images:
                label_path = os.path.join(word_dir, img.replace(
                    '.jpg', '.txt').replace('.png', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"{current_class_id}\n")

            current_class_id += 1


def main():
    # Step 1: Process Video Data
    process_video_data(dataset_path)

    # Step 2: Create label files (.txt)
    create_labels(dataset_path)


# Run the main function
if __name__ == '__main__':
    main()
