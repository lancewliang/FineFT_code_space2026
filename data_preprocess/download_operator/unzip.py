import nest_asyncio
import logging
import os, tarfile
import gzip
import shutil

nest_asyncio.apply()
logging.basicConfig(level=logging.DEBUG)

# function used by default if not provided via options
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_store_path",
    type=str,
    default="./DOWNLOAD_DATASET/binance-futures/BTCUSDT/trades",
    help="the name of the exchange",
)


class unziper:
    def __init__(self, root_store_path):
        self.root_store_path = root_store_path
        path_list = os.listdir(self.root_store_path)
        self.path_list = [
            os.path.join(self.root_store_path, path) for path in path_list
        ]
        self.gz_file_list = [i for i in self.path_list if i.endswith(".gz")]

    def untar(self):

        for gz_file in self.gz_file_list:
            if os.path.exists(gz_file[:-3]):
                os.remove(gz_file)
            else:
                with gzip.open(
                    gz_file,
                    "rb",
                ) as f_in:
                    with open(
                        gz_file[:-3],
                        "wb",
                    ) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(
                    os.path.join(
                        gz_file,
                    )
                )

    def run(self):
        self.untar()


if __name__ == "__main__":
    args = parser.parse_args()
    unziper(args.root_store_path).run()
