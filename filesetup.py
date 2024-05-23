import os
import random


def delete_files(directory, num_files_to_delete):
    files = os.listdir(directory)
    files_to_delete = random.sample(files, num_files_to_delete)

    for file_name in files_to_delete:
        file_path = os.path.join(directory, file_name)
        os.remove(file_path)


# Example usage:
directory_path = 'images3/images/train/sad'
num_files_to_delete = 900  # Specify the number of files you want to delete
delete_files(directory_path, num_files_to_delete)