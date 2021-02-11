import os
from .utils import download_url, unzip, download_gdrive

# http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
image_2 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
image_3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip"
velodyne = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
calib = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
label = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

# https://github.com/kujason/avod
train = "14v045QtiTo7rz4WA7SiBx4Ge29M1fdkx"
val = "1FKeWeDJlQLqNB6KjQuUQpdlVe31aUy5t"
trainval = "1r2M_XnBQ533Je_DFiiolmQrnstcKghcR"


def download_kitti(dst, delete_zips=True):
    # Download raw data
    file_list = [image_2, image_3, velodyne, calib, label]
    for remote_file in file_list:
        local_file = download_url(remote_file, dst)
        unzip(local_file, delete_zips=delete_zips)

    # Prepare split files
    for token in [train, val, trainval]:
        download_gdrive(token, dst)
    with open(os.path.join(dst, "test.txt"), "w") as f:
        test_list = list(map(lambda x: f"{x:06d}", range(7518)))
        f.write("\n".join(test_list))

    print(f"KITTI dataset has been downloaded to {dst}")


if __name__ == "__main__":
    download_kitti("/tmp/kitti/", delete_zips=False)
