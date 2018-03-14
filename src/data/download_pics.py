import os
from os.path import join
import time
from urllib import request
from urllib.error import HTTPError

from src.utils.paths import PATH_DATA


def url_to_fname(url, out_dir):
    headshot = url.split('/')[-1]
    fname = os.path.join(out_dir, headshot)
    return fname


if __name__ == '__main__':
    url_file = join(PATH_DATA, 'img_urls.txt')
    out_dir = join(PATH_DATA, 'celeb_id_raw')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    urls = []
    with open(url_file, 'r') as f:
        for line in f:
            url = line.strip()
            urls.append(url)

    errors = []
    time_start = time.time()
    for c, url in enumerate(urls):
        fname = url_to_fname(url, out_dir)

        if os.path.exists(fname):
            continue

        print(f'[{c}/{len(urls)}] Downloading {url}...', end=' ')
        try:
            with open(fname, 'wb') as f:
                f.write(request.urlopen(url).read())
            print('Success!')
        except HTTPError:
            print('Fail!')
            errors.append(url)
    print(f'Errors: {len(errors)}')
    print(f'Time: {time.time() - time_start}')
    with open(join(PATH_DATA, 'celeb_id_raw_errors.txt'), 'w') as fout:
        fout.write('\n'.join(errors))
