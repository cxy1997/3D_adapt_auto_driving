from .utils import download_url, unzip

split_list = ["train1", "train2", "train3", "train4", "val", "test"]


# Download Argoverse v1.1
# https://www.argoverse.org/data.html#download-link
def get_download_link(split):
    return f"https://s3.amazonaws.com/argoai-argoverse/tracking_{split}_v1.1.tar.gz"


def download_argo(dst, delete_zips=True):
    # Download raw data
    for split in split_list:
        remote_file = get_download_link(split)
        local_file = download_url(remote_file, dst)
        unzip(local_file, delete_zips=delete_zips)

    print(f"Argoverse dataset has been downloaded to {dst}")


if __name__ == "__main__":
    download_argo("/tmp/argo/", delete_zips=False)
