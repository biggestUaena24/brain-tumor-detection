import os
import cv2

from pathlib import Path
from PIL import Image


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]  # the root directory is 2 levels up

    input_filepath = os.path.join(project_root, "data/raw/brain_tumor")
    output_filepath = os.path.join(project_root, "data/processed/brain_tumor")

    input_dir = Path(input_filepath)
    output_dir = Path(output_filepath)

    if not input_dir.exists():
        raise ValueError(f"Path: {input_filepath} does not exists.")

    if not output_dir.exists():
        output_dir.mkdir()

    classes = ["no", "yes"]

    for cl in classes:
        input_folder = os.path.join(input_dir.resolve(), cl)

        # create output input_folder
        output_folder = Path(os.path.join(output_dir.resolve(), cl))
        if not output_folder.exists():
            output_folder.mkdir()

        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            img_resized = img.resize((224, 224), Image.LANCZOS)
            img_resized.save(os.path.join(
                output_folder.resolve(), img_name), format="JPEG")


if __name__ == '__main__':
    main()
