import nest_asyncio
from tardis_dev import datasets, get_exchange_details
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
    default="./DOWNLOAD_DATASET",
    help="the name of the exchange",
)
parser.add_argument(
    "--exchange",
    type=str,
    default="binance-futures",
    help="the name of the exchange",
)
parser.add_argument(
    "--symbols",
    type=str,
    default="BTCUSDT",
    help="the name of the symbols",
)
parser.add_argument(
    "--start_date", type=str, default="2023-04-01", help="start_date of the downloading"
)
parser.add_argument(
    "--end_date", type=str, default="2023-10-18", help="end_date of the downloading"
)

parser.add_argument(
    "--data_types",
    type=str,
    default="book_snapshot_5",
    help="the name of the data_types",
)
args = parser.parse_args()


def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


def untar(fname, dirs):
    """
    解压tar.gz文件
    :param fname: 压缩文件名
    :param dirs: 解压后的存放路径
    :return: bool
    """
    try:
        t = tarfile.open(fname)
        t.extractall(path=dirs)
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    # returns data available at https://api.tardis.dev/v1/exchanges/deribit
    deribit_details = get_exchange_details(args.exchange)
    # print(deribit_details)
    download_dirs = os.path.join(
        args.root_store_path,
        "{}/{}/{}".format(args.exchange, args.symbols, args.data_types),
    )

    datasets.download(
        # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
        exchange=args.exchange,
        # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
        # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
        # data_types=['trades', 'book_snapshot_5',],
        data_types=[
            args.data_types,
        ],
        # change date ranges as needed to fetch full month or year for example
        from_date=args.start_date,
        # to date is non inclusive
        to_date=args.end_date,
        # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
        symbols=[
            args.symbols,
        ],
        # (optional) your API key to get access to non sample data as well
        api_key="replace with your api key, get it from https://tardis.dev/",
        # (optional) path where data will be downloaded into, default dir is './datasets'
        download_dir=download_dirs,
        # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
        # get_filename=default_file_name,
        # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
        # get_filename=file_name_nested,
    )
    download_dir = download_dirs
    gz_file_list = os.listdir(download_dir)
    gz_file_list = [i for i in gz_file_list if i.endswith(".gz")]
    for gz_file in gz_file_list:
        with gzip.open(
            os.path.join(
                download_dir,
                gz_file,
            ),
            "rb",
        ) as f_in:
            with open(
                os.path.join(
                    download_dir,
                    gz_file[:-3],
                ),
                "wb",
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(
            os.path.join(
                download_dir,
                gz_file,
            )
        )
