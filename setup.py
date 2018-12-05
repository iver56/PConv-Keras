from setuptools import setup

setup(
    name='keras_pconv',
    version='1.0',
    packages=['keras_pconv', ],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
        'Keras>=2.2.0',
        'matplotlib>=2.2.0',
        'numpy>=1.13.3',
        'opencv-python>=3.4.1.15',
        'Pillow>=5.1.0',
    ],
    extras_require={
        "tf": ["tensorflow>=1.8.0"],
        "tf_gpu": ["tensorflow-gpu>=1.8.0"],
    }
)
