import os


def validate_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def remove_files(*files):
    for file in files:
        if file and os.path.exists(file) and not os.path.isdir(file):
            os.remove(file)