import os
import random
import shutil

# Set the source directory and test directory
source_dir = 'dataset'
test_dir = 'dataset_test'

test_dir_zero = 'dataset_test/0'
test_dir_one = 'dataset_test/1'


# Create the test directory if it doesn't exist
if not os.path.exists(test_dir_zero):
    os.makedirs(os.path.join(test_dir, "0"))
if not os.path.exists(test_dir_one):
    os.makedirs(os.path.join(test_dir, "1"))

# Function to move a percentage of files from source to test directory
def move_files(src_dir, test_dir, percentage):
    files = os.listdir(src_dir)
    num_files_to_move = int(len(files) * percentage)

    # Randomly select files to move
    files_to_move = random.sample(files, num_files_to_move)

    for file in files_to_move:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(test_dir, file)
        shutil.move(src_path, dst_path)
        print(f"Moved: {file}")

# Move 20% of files from each "1" folder
move_files(os.path.join(source_dir, "1"), os.path.join(test_dir, "1"), 0.2)

# Move 20% of files from each "0" folder
move_files(os.path.join(source_dir, "0"), os.path.join(test_dir, "0"), 0.2)

