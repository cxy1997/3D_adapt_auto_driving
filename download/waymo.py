import os
from .utils import download_gcloud, unzip

license_page = "https://waymo.com/open/licensing/"
note = f"Please accept the license agreement at {license_page}. It may take up to 2 business days to be granted access."


split_list = ["training", "validation"]


def get_download_link(split):
    return f"waymo_open_dataset_v_1_0_0_individual_files/{split}"


def download_waymo(dst, delete_zips=True):
    # Download raw data
    for split in split_list:
        remote_folder = get_download_link(split)
        local_folder = download_gcloud(remote_folder, dst, note)
        for zip_file in filter(lambda x: x.endswith(".tar"), os.listdir(local_folder)):
            unzip(os.path.join(local_folder, zip_file), delete_zips=delete_zips)

    print(f"Waymo dataset has been downloaded to {dst}")


if __name__ == "__main__":
    download_waymo("/tmp/waymo")
