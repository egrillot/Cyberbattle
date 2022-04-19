

import requests
import json
import tqdm

url = 'https://github.com/mitre/cti/tree/master/enterprise-attack/enterprise-attack.json'

def download_mitre_attacks():

    r = requests.get(url, stream=True)
    print(r.headers.get('content-length'))
    file_name = 'recent_dl/enterprise-attack.txt'
    data = r.read()
    file_size = len(data)
    buffer = 8192
    file_size_dl = 0

    while file_size_dl != file_size:

        buffer = min(buffer, file_size - file_size_dl)
        break