import requests
import zipfile
import io
import os
import shutil
import mimetypes
import requests
import zipfile
import io
import os
import shutil
import mimetypes
from urllib.parse import urlparse, unquote

def download_dataset(data_url, save_path=None, force_download=False):
    """
    This function downloads a dataset from a URL. It can handle both zip files
    (directories) and single files, detecting the type automatically.
    It extracts zip files and stores single files in the specified path.
    
    :param data_url: URL of the file to download
    :param save_path: Full path where the file should be saved. If None, it will
                      save to 'data/[filename]'
    :param force_download: If True, overwrites existing files
    """
    if 'dl=0' in data_url:
        data_url = data_url.replace('dl=0', 'dl=1')

    # Determine the save directory and filename
    if save_path:
        save_dir, filename = os.path.split(save_path)
    else:
        save_dir = 'data'
        filename = None

    if os.path.exists(save_path) and not force_download:
        print(f"File already exists at '{save_path}'. Skipping download.")
        return

    os.makedirs(save_dir, exist_ok=True)

    print("Downloading data...")
    response = requests.get(data_url)

    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        
        if content_type == 'application/zip' or data_url.endswith('.zip'):
            # It's a zip file
            if not filename:
                filename = 'downloaded_archive.zip'
            zip_path = os.path.join(save_dir, filename)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(save_dir)
            os.remove(zip_path)  # Remove the zip file after extraction
            print(f"Zip file successfully downloaded and extracted to '{save_dir}'")
        else:
            # It's a single file
            if not filename:
                # Try to get filename from Content-Disposition header
                content_disposition = response.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                else:
                    # Use the last part of the path from the URL, without query parameters
                    url_path = unquote(urlparse(data_url).path)
                    filename = os.path.basename(url_path)
                    if not filename:
                        # If still no filename, use the mime type to generate one
                        ext = mimetypes.guess_extension(content_type) or ''
                        filename = f"downloaded_file{ext}"

            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"File successfully downloaded and saved to '{file_path}'")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
        
        
# import requests
# import zipfile
# import io
# import os
# import shutil

# def download_dataset(data_url, download_directory='data/', force_download=False):
#     """
#     This function downloads a data directory from a URL 
#     as a zip file and then extracts and stores the results
#     in a local directory. Checks if the data already exists,
#     modifies the URL if for a Dropbox folder, can force download, etc.
#     """
#     if 'dl=0' in data_url:
#         data_url = data_url.replace('dl=0', 'dl=1')

#     if os.path.exists(download_directory) and os.listdir(download_directory) and not force_download:
#         print(f"Data already exists in '{download_directory}'. Skipping download.")
#         return

#     if force_download and os.path.exists(download_directory):
#         shutil.rmtree(download_directory)
#         print(f"Directory '{download_directory}' removed due to force_download=True.")

#     os.makedirs(download_directory, exist_ok=True)

#     print("Downloading data...")
#     response = requests.get(data_url)

#     if response.status_code == 200:
#         with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#             z.extractall(download_directory)
#         print(f"Data successfully downloaded and extracted to '{download_directory}'")
#     else:
#         print(f"Failed to download data. Status code: {response.status_code}")
