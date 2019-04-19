from setuptools import setup

# list all your packages here
# use '.' for sub-packages
setup(
    name='submission',
    packages=['submission', 'albumentations', 'albumentations.augmentations', 'albumentations.core', 'albumentations.imgaug', 'albumentations.pytorch', 'albumentations.torch', 'utils'],
)
