sudo apt-get updata
sudo apt-get install -y python3-pip

sudo pip3 install -i https://mirrors.aliyun.com/pypi/simple numpy pandas scipy scikit-learn matplotlib
sudo pip3 install -i https://mirrors.aliyun.com/pypi/simple  tensorflow tsfresh xgboost

安裝tslearn之前需要裝:llvm -> llvmlite -> numba -> tslearn