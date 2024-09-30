import os
from glob import glob
from itertools import chain, combinations
from multiprocessing import Process
from pathlib import Path
from typing import Any, List, Optional, Tuple
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from functools import partial
from joblib import Parallel, cpu_count, delayed
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from spacy.util import minibatch

from .paths import root

def read_directory_csvs(directory):
    """
    This function expects a posix path, for example as returned by one of the paths from the dcss paths module,
    like russian_troll_tweets_path if you run:

        from dcss.paths import russian_troll_tweets_path

    It returns a Pandas dataframe with all the various csv files concatenated into one object.
    """
    files = os.listdir(directory)
    files = [f for f in files if '.csv' in f]

    df = pd.concat((pd.read_csv(directory / file, encoding='utf-8', low_memory=False)
                    for file in files),
                   ignore_index=False)
    return df




def save_to_file(q, file_name):
    with open(file_name, 'w') as out:
        while True:
            val = q.get()
            if val is None: break
            for speech in val:
                out.write('\n'.join(speech))

def mp_disk(items, function, file_name, q, *args):
    cpu = cpu_count()
    batch_size = 32
    partitions = minibatch(items, size=batch_size)
    p = Process(target = save_to_file, args = (q, file_name))
    p.start()
    Parallel(n_jobs=cpu, max_nbytes=None)(delayed(function)(v, *args) for v in partitions) #executes the function on each batch
    q.put(None)
    p.join()


def list_files(rootdir, extension):
    """
    This utility function returns a list of paths to files of a given type, e.g. all csv files in a nested directory structure.
    """
    PATH = rootdir
    EXT = f'*.{extension}'
    files = [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))]
    return files

class IterSents(object):
    """
    Gensim can operate on one file at a time to prevent memory issues.
    This class is a simple iterator that will provide the data to
    Word2Vec one at a time.
    """
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

def mp(items, function, *args, **keywords):
    """Applies a function to a list or dict of items, using multiprocessing.

    This is a convenience function for generalized multiprocessing of any
    function that deals with a list or dictionary of items. The functions
    passed to `mp` must accept the list of items to be processed at the end
    of their function call, with optional arguments first. *args can be any
    number of optional arguments accepted by the function that will be
    multiprocessed. On Windows, functions must be defined outside of the
    current python file and imported, to avoid infinite recursion.
    """
    if isinstance(items, list) == False:
        print("Items must be a list")
        return

    if len(items) < 1:
        print("List of items was empty")
        return

    cpu = cpu_count()

    batch_size = 32
    partitions = minibatch(items, size=batch_size)
    executor = Parallel(n_jobs=cpu,
                        backend="multiprocessing",
                        prefer="processes")
    do = delayed(partial(function, *args, **keywords))
    tasks = (do(batch) for batch in partitions)
    temp = executor(tasks)

    results = list(chain(*temp))

    return results


def sparse_groupby(groups, sparse_m, vocabulary):
    grouper = LabelBinarizer(sparse_output=True)
    grouped_m = grouper.fit_transform(groups).T.dot(sparse_m)

    df = pd.DataFrame.sparse.from_spmatrix(grouped_m)
    df.columns = vocabulary
    df.index = grouper.classes_

    return df



def load_api_key(key: str, env_path: Path = Path.cwd()) -> Optional[str]:
    """
    Assumes you have a .env file in the root directory.
    Should be added to .gitignore, of course.
    """
    load_dotenv(env_path / ".env")
    api_key = os.getenv(key)
    return api_key


def load_api_key_list(
    key_names: List[str], env_path: Path = root
) -> List[Optional[str]]:
    """
    Assumes you have a .env file in the root directory with the api keys on new lines.
    Should be added to .gitignore, of course.
    """
    load_dotenv(env_path / ".env")
    keys: List[Optional[str]] = []
    for key in key_names:
        api_key = os.getenv(key)
        keys.append(api_key)
    return keys


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


import ast
import glob
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.logging import RichHandler

from .paths import root


def set_torch_device():
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        device_properties: Any = torch.cuda.get_device_properties(device)
        vram = device_properties.total_memory // (1024**2)
        logging.info(
            f"Set device to {device} with {vram}MB (~ {np.round(vram/1024)}GB) of VRAM"
        )
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
        logging.info(f"Set device to {device}")
    else:
        device = torch.device("cpu")
        logging.info(f"Set device to {device}")
    return device


def load_api_key(key: str, env_path: Path = Path.cwd()) -> Optional[str]:
    """
    Assumes you have a .env file in the root directory.
    Should be added to .gitignore, of course.
    """
    load_dotenv(env_path / ".env")
    api_key = os.getenv(key)
    return api_key


def load_api_key_list(
    key_names: List[str], env_path: Path = root
) -> List[Optional[str]]:
    """
    Assumes you have a .env file in the root directory with the api keys on new lines.
    Should be added to .gitignore, of course.
    """
    load_dotenv(env_path / ".env")
    keys: List[Optional[str]] = []
    for key in key_names:
        api_key = os.getenv(key)
        keys.append(api_key)
    return keys


def initialize_logger(logging_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, logging_level),
        format="%(asctime)s\n%(message)s",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger("rich")
    return logger


def save_json(data: Any, file_path: str) -> None:
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(
            f"An error occurred while saving data to {file_path}: {e}")


def get_fpaths_and_fnames(dir: str, ftype: str = "json") -> List[Tuple[Path, str]]:
    directory = Path(dir)
    files = glob.glob(str(directory / f"*.{ftype}"))
    fpaths_fnames = [(Path(file), Path(file).stem) for file in files]
    return fpaths_fnames


def strings_to_lists(series: Any) -> Any:
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def lists_to_strings(series: Any, sep: str = ", ") -> Any:
    """
    If the lists data you want as a string is stored as a string
    You need to convert it to lists first, then back to the string you want... :/
    """
    series = strings_to_lists(series)
    return series.apply(lambda x: sep.join(x) if isinstance(x, list) else x)


def run_in_conda(script: str, conda_env_name: str = "gt") -> None:
    conda_script = script

    command = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {conda_env_name} && "
        f"python {conda_script} && "
        "conda deactivate"
    )

    logging.info(f"Executing command: {command}")

    try:
        process = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info(
            f"Successfully executed '{conda_script}' in conda env '{conda_env_name}'\n"
            f"Output:\n{process.stdout.decode()}"
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Failed to execute '{conda_script}' in conda env '{conda_env_name}'\n"
            f"Error:\n{e.stderr.decode()}"
        )


def markdown_table(df: pd.DataFrame, filepath: str = None, indexed=False) -> str:
    """
    Convert a pandas DataFrame to a markdown table.

    Parameters:
    df (pd.DataFrame): The DataFrame to convert to markdown.
    filepath (str, optional): The path where the markdown file should be saved.
    Defaults to None.
    indexed (bool, optional): Whether to include the DataFrame index in the markdown
    table.
    Defaults to False.

    Returns:
    str: The markdown formatted table as a string.
    """
    pd.set_option("display.float_format", lambda x: "%.0f" % x)

    md = df.to_markdown(
        index=indexed
    )  # Convert the DataFrame to markdown with or without index

    if filepath is not None:
        with open(filepath, "w") as file:
            file.write(
                md
            )  # Write the markdown string to the file if a filepath is provided

    return md  # Return the markdown string


def estimate_meters_from_rssi(df, rssi_col, A=-40, n=2):
    """
    A = -40  # RSSI value at 1 meter distance
    n = 2    # Path-loss exponent
    """
    estimated_meters = 10 ** ((A - df[rssi_col]) / (10 * n))
    return estimated_meters


def update_quarto_variables(new_key, new_value, path="_variables.yml"):
    with open(path, "r") as file:
        quarto_variables = yaml.safe_load(file)

    # add a new key-value pair or update an existing key
    quarto_variables[new_key] = new_value

    with open(path, "w") as file:
        yaml.dump(quarto_variables, file, default_flow_style=False)
