Install casacore

sudo apt update
sudo apt install -y cmake g++ libboost-all-dev libcfitsio-dev libfftw3-dev \
    libgsl-dev libhdf5-dev libreadline-dev libwcslib-dev casacore-data

cd ~
git clone https://github.com/casacore/casacore.git
cd casacore
mkdir build && cd build
cmake .. -DBUILD_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install

ls -l /usr/local/lib | grep casa

cd casacore_c++

g++ -o read_ms read_ms.cpp -I/usr/local/include -L/usr/local/lib -lcasa_casa -lcasa_tables -lboost_system -lboost_filesystem


