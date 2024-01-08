import os

folder_path = "results/img2img-samples" 

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        index = filename.find("-")
        if index != -1:
            new_filename = filename[:index] + "-corrected.jpg"

            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)

            os.rename(old_filepath, new_filepath)
            print(f"Rename {filename} into {new_filename}")