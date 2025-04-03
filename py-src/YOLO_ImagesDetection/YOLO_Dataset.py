import os

def manage_dataset_images(path):
    # Check if the directory exists
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return
    else:
        print(f"Directory {path} exists.")
        train_path = os.path.join(path, "train")
        test_path = os.path.join(path, "test")
        valid_path = os.path.join(path, "valid")

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)

        images_train_path = os.path.join(train_path, "images")
        images_test_path = os.path.join(test_path, "images")
        images_valid_path = os.path.join(valid_path, "images")

        os.makedirs(images_train_path, exist_ok=True)
        os.makedirs(images_test_path, exist_ok=True)
        os.makedirs(images_valid_path, exist_ok=True)


        image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        train_count = int(len(image_files) * 0.7)

        for i, image in enumerate(image_files):
            src_path = os.path.join(path, image)
            if i < train_count:
                dest_path = os.path.join(images_train_path, image)
            elif i < train_count + int(len(image_files) * 0.2):
                dest_path = os.path.join(images_valid_path, image)
            else:
                dest_path = os.path.join(images_test_path, image)
            
            # Move the file to the appropriate directory
            os.rename(src_path, dest_path)
            print(f"Moved {image} to {dest_path}")


def manage_dataset_labels(path):
    # Check if the directory exists
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        return
    else:
        print(f"Directory {path} exists.")

        images_path = os.path.join(path, "test", "images")
        labels_path = os.path.join(path, "test", "labels")

        image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

        for image in image_files:
            image_name, _ = os.path.splitext(image)
            label_file = f"{image_name}.txt"
            src_label_path = os.path.join(path, label_file)
            dest_label_path = os.path.join(labels_path, label_file)
            # Move the corresponding label file if it exists
            if os.path.exists(src_label_path):
                os.rename(src_label_path, dest_label_path)
                print(f"Moved {label_file} to {dest_label_path}")


manage_dataset_labels("D:/Enzo/RoboflowDataset/images")
            
