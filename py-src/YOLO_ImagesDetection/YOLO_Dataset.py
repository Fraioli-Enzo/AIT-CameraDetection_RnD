import os

# Define old-to-new class mapping
class_mapping = {3: 0, 6: 1}  # old index -> new index

labels_dir = "D:/Enzo/datasets/FDDv2/valid/labels"
new_labels_dir = "D:/Enzo/datasets/FDDv2/valid/filtered_labels"

os.makedirs(new_labels_dir, exist_ok=True)

for file_name in os.listdir(labels_dir):
    file_path = os.path.join(labels_dir, file_name)
    new_file_path = os.path.join(new_labels_dir, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])

        if class_id in class_mapping:
            parts[0] = str(class_mapping[class_id])  # Update class index
            new_lines.append(" ".join(parts))

    # Save the new annotation file
    if new_lines:
        with open(new_file_path, "w") as f:
            f.write("\n".join(new_lines))
