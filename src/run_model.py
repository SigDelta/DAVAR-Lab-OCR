import json
from pathlib import Path

import click
import cv2
from tqdm import tqdm

from model.model_wrapper import LGPMAModel
from model.utils import visualize_result


@click.command()
@click.option("-i", "--input_dir", type=Path, default="samples")
@click.option("-o", "--output_dir", type=Path, default="results")
@click.option(
    "--config_path",
    type=str,
    default="src/configs/lgpma/configs/lgpma_pub.py",
)
@click.option(
    "--checkpoint_path", type=str, default="models/lgpma/maskrcnn-lgpma-pub-e12-pub.pth"
)
def main(input_dir: Path, output_dir: Path, config_path: str, checkpoint_path: str):
    # loading model from config file and pth file
    model = LGPMAModel(config_path=config_path, checkpoint_path=checkpoint_path)

    # generate prediction of html and save result to savepath
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    input_file_paths = list(input_dir.glob("*"))
    for file_path in tqdm(input_file_paths):
        img = cv2.imread(str(file_path))

        result = model(img)
        result["img_shape"] = img.shape
        result["file_path"] = file_path.name
        results.append(model(file_path))

        if result["status"] == "success":
            visualize_result(
                img=img,
                result=result,
                output_path=output_dir / f"{file_path.stem}_vis.jpg",
            )

    json_path = output_dir / "preds.json"
    with json_path.open("w+") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
