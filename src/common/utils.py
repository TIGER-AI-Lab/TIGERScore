import random
import os
import numpy as np
import torch
import argparse
import hashlib
import requests
import time
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

USER_AGENT = get_datasets_user_agent()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def empty2None(x):
    if x == '':
        return None
    elif isinstance(x, str):
        return x
    else:
        raise argparse.ArgumentTypeError('String value expected.')

def empty2Noneint(x):
    if x == '':
        return None
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return int(x)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')

def empty2zero(x):
    if x == '':
        return 0
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return int(x)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')



def generate_hash_code(text):
    if text is None:
        return None
    # Convert the text to bytes and create a hash object
    hash_object = hashlib.sha256(text.encode())

    # Get the hexadecimal representation of the hash code
    hex_code = hash_object.hexdigest()

    # Return the first 16 digits of the hexadecimal code
    return hex_code[:16]

def fetch_single_image(image_url, timeout=None, retries=2):
    if os.path.exists(image_url):
        # fetch from local
        try:
            image = Image.open(image_url).convert("RGB")
        except Exception as e:
            if retries > 0:
                time.sleep(3)
                return fetch_single_image(image_url, timeout=timeout, retries=retries - 1)
    else:
        # fetch from url
        try:
            r = requests.get(image_url, timeout=timeout, stream=True, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            image = Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            if retries > 0:
                time.sleep(3) # Wait 3 seconds before retrying
                return fetch_single_image(image_url, timeout=timeout, retries=retries - 1)
            else:
                print(f"Failed to fetch image from {image_url} after {retries} retries")
                raise e
    return image

def fetch_images(image_urls, num_threads, timeout=None, retries=2):
    """
    Fetch images from a list of URLs in parallel.
    Args:
        image_urls (list): List of image URLs.
        num_threads (int): Number of threads to use.
        timeout (int, optional): Timeout for the request. Defaults to None.
        retries (int, optional): Number of retries. Defaults to 0.
    Returns:
        list: List of PIL images.
    """
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        images = list(
            tqdm(
            executor.map(fetch_single_image_with_args, image_urls), 
            total=len(image_urls),
            desc="Fetching images")
        )
    print("Fetched {} images".format(len(images)))
    return images
