import os
import tempfile

import click
import yaml

import train_finetune


@click.command()
@click.option("-p", "--config_path", default="Configs/config.yml", type=str)
def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_params = config.setdefault("model_params", {})
    cde = model_params.setdefault("cde", {})
    cde["enabled"] = True
    config["mocha_cde_only"] = True

    fd, tmp_path = tempfile.mkstemp(suffix=".yml", prefix="mocha_finetune_")
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        train_finetune.main(["--config_path", tmp_path], standalone_mode=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
