from setuptools import setup, find_packages


setup(
  name = 'threefiner',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  entry_points={
    # CLI tools
    'console_scripts': [
      'threefiner = threefiner.cli:main'
    ],
  },
  version = '0.1.2',
  license='MIT',
  description = 'Threefiner: a text-guided mesh refiner',
  author = 'kiui',
  author_email = 'ashawkey1999@gmail.com',
  long_description=open("readme.md", encoding="utf-8").read(),
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/3DTopia/threefiner',
  keywords = [
    'generative mesh refinement',
  ],
  install_requires=[
    'tyro',
    'tqdm',
    'rich',
    'ninja',
    'numpy',
    'pandas',
    'matplotlib',
    'opencv-python',
    'imageio',
    'imageio-ffmpeg',
    'scipy',
    'scikit-learn',
    'torch',
    'einops',
    'huggingface_hub',
    'diffusers',
    'accelerate',
    'transformers',
    "sentencepiece", # required by deepfloyd-if T5 encoder
    'plyfile',
    'pygltflib',
    'xatlas',
    'trimesh',
    'PyMCubes',
    'pymeshlab',
    "pysdf",
    "diso",
    "envlight",
    'dearpygui',
    'kiui >= 0.2.1',
  ],
  classifiers=[
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
