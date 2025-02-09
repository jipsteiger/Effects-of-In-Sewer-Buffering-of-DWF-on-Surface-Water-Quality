import logging
import os
import sys

import requests

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))


"""
(Adjusted) Code from KNMI dataplatform docs, at:
https://developer.dataplatform.knmi.nl/open-data-api#example-last
https://dataplatform.knmi.nl/dataset/rad-opera-hourly-rainfall-accumulation-euradclim-2-0
https://dataplatform.knmi.nl/dataset/harmonie-arome-cy43-p2a-1-0
"""


class OpenDataAPI:
    def __init__(self, api_token: str):
        self.base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        self.headers = {"Authorization": api_token}

    def __get_data(self, url, params=None):
        return requests.get(url, headers=self.headers, params=params).json()

    def list_files(self, dataset_name: str, dataset_version: str, params: dict):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files",
            params=params,
        )

    def get_file_url(self, dataset_name: str, dataset_version: str, file_name: str):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        )


def download_file_from_temporary_download_url(download_url, folder, filename):
    try:
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Create the full file path
        file_path = os.path.join(folder, filename)

        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.exception("Unable to download file using download URL")
        sys.exit(1)

    logger.info(f"Successfully downloaded dataset file to {filename}")


def get_api_key(file_name="precipitation/api_key.txt"):
    with open(file_name, "r") as file:
        api_key = file.read()
    return api_key


def main(
    dataset_name="harmonie_arome_cy43_p4a",
    dataset_version="1.0",
    data_folder="data/precipitation/harmonie",
):
    api_key = get_api_key()
    dataset_name = "RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM"
    dataset_version = "2.0"
    logger.info(f"Fetching latest file of {dataset_name} version {dataset_version}")

    api = OpenDataAPI(api_token=api_key)

    # sort the files in descending order and only retrieve the first file
    params = {"maxKeys": 1, "orderBy": "created", "sorting": "desc"}
    response = api.list_files(dataset_name, dataset_version, params)
    if "error" in response:
        logger.error(f"Unable to retrieve list of files: {response['error']}")
        sys.exit(1)

    latest_file = response["files"][0].get("filename")
    logger.info(f"Latest file is: {latest_file}")

    # fetch the download url and download the file
    response = api.get_file_url(dataset_name, dataset_version, latest_file)
    download_file_from_temporary_download_url(
        response["temporaryDownloadUrl"], r"data/precipitation/harmonie", latest_file
    )


if __name__ == "__main__":
    main(
        dataset_name="harmonie_arome_cy43_p4a",
        dataset_version="1.0",
        data_folder="data/precipitation/harmonie",
    )
    main(
        dataset_name="RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM",
        dataset_version="2.0",
        data_folder="data/precipitation/euradclim",
    )
