import os
from subprocess import Popen, PIPE
from shutil import which
import multiprocessing
import webbrowser


def exec(cmd):
    return Popen(cmd, shell=True, stdout=PIPE).stdout.read().decode("utf-8")


def is_tool(name):
    return which(name) is not None


def download_url(src, dst):
    if "." in dst:  # dst is a file
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    else:           # dst is a folder
        os.makedirs(dst, exist_ok=True)
        dst = os.path.join(dst, os.path.basename(src))

    if os.path.isfile(dst):
        return dst
    elif is_tool("axel"):
        command = f"axel -n {multiprocessing.cpu_count()} {src} -o {dst}"
    else:
        command = f"wget {src} -o {dst}"

    print(f">>> {command}")
    os.system(command)
    return dst


def download_gdrive(token, dst):
    os.makedirs(dst, exist_ok=True)
    if is_tool("gdrive"):
        command = f"gdrive download {token} --path {dst}"
    else:
        if not os.path.isfile("/tmp/gdrive-linux-x64"):
            os.system("wget https://github.com/gdrive-org/gdrive/releases/download/2.1.0/gdrive-linux-x64 -d /tmp --no-check-certificate")
        os.system("chmod +x /tmp/gdrive-linux-x64")
        command = f"/tmp/gdrive-linux-x64 download {token} --path {dst}"
    print(f">>> {command}")
    os.system(command)


def unzip(file, delete_zips=True):
    if file.endswith(".zip"):
        command = f"unzip -n {file} -d {os.path.dirname(file)}"
    elif file.endswith(".tar.gz"):
        command = f"tar -xzf {file} -C {os.path.dirname(file)}"
    elif file.endswith(".tar"):
        command = f"tar -xf {file} -C {os.path.dirname(file)}"
    else:
        raise NotImplementedError
    print(f">>> {command}")
    os.system(command)

    if delete_zips:
        os.remove(file)


def download_gcloud(src, dst, note):
    if "." in dst:
        dst = os.path.dirname(dst)
    os.makedirs(dst, exist_ok=True)

    if is_tool("gsutil"):
        gsutil = f"gsutil"
    else:
        gsutil = os.path.expanduser('~/google-cloud-sdk/bin/gsutil')
        if not os.path.isfile(gsutil):
            print(f">>> curl https://sdk.cloud.google.com | bash")
            os.system("curl https://sdk.cloud.google.com | bash")

            print(f">>> {os.path.expanduser('~/google-cloud-sdk/bin/gcloud')} init")
            os.system(f"{os.path.expanduser('~/google-cloud-sdk/bin/gcloud')} init")

    invalid = exec(f"{gsutil} ls gs://{src}") == ""
    if invalid:
        print(note)
    else:
        print(f">>> {gsutil} -m cp -r gs://{src} {dst}")
        os.system(f"{gsutil} -m cp -r gs://{src} {dst}")
    return os.path.join(dst, os.path.basename(src))


def open_browser(url):
    webbrowser.open(url, new=2, autoraise=True)
