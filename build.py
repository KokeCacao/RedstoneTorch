import base64
import gzip
from pathlib import Path


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def build_script():
    # list all your packages and files here
    # use '/' for sub-packages
    to_encode = None
    packages = ['submission', 'albumentations', 'albumentations/augmentations', 'albumentations/core', 'albumentations/imgaug', 'albumentations/pytorch', 'albumentations/torch', 'utils']
    files = ['setup.py', 'reproduceability.py']
    for package in packages:
        if to_encode is None: to_encode = list(Path(package).glob('*.py'))
        else: to_encode = to_encode + list(Path(package).glob('*.py'))
    for f in files:
        to_encode = to_encode + [Path(f)]

    file_data = {str(path): encode_file(path) for path in to_encode}
    template = Path('script_template.py').read_text('utf8')
    Path('build/script.py').write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')


if __name__ == '__main__':
    build_script()
