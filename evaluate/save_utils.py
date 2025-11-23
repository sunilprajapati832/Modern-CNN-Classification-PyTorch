import os


def create_result_dir(base, model_name):
    path = os.path.join(base, model_name)
    os.makedirs(path, exist_ok=True)
    return path
