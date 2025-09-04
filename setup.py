from setuptools import find_packages, setup

setup(
    name="gan_synthesis",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "gan_synthesis": ["mask_vae_models/*", "model_utils/*"]  # include model weights/configs
    },
)
