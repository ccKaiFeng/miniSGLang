from setuptools import setup, find_packages

# setup.py 是 Python 包安装脚本。
# 在仓库根目录执行 `pip install -e .` 后，当前源码目录会以 editable
# 方式安装到环境中，之后其他脚本就能 `import zipcache`。

VERSION = "0.0.1"
DESCRIPTION = "ZIPCACHE"
LONG_DESCRIPTION = "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification"

# Setting up
# find_packages() 会自动查找 zipcache/ 下面的 Python 包。
setup(
    name="zipcache",
    version=VERSION,
    author="Yefei He",
    author_email="billhe@zju.edu.cn",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "AI"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
)
