import logging

def setup_logger(log_path="outputs/logs/train.log"):
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)

    logging.getLogger("").addHandler(console)
