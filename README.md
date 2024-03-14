# 342pr1
Placeholder repository for CENG342's first project. </br>
https://github.com/He-Is-HaZaRdOuS/342pr1 </br>

# About
This project aims to use MPI to speed up image processing algorithms, specifically focusing on edge detection.

# Disclaimer!
CMake does not recognize non-english characters in the build path. </br>
If your computer's file structure contains any non-english characters then you won't be able to compile the program. </br>
Please place the project folder somewhere such that the absolute path to it won't contain any non-english characters. </br>

## Installing CMake

##### Option 1
Install CMake from your distribution's package manager

##### Option 2
Open a terminal window and execute the following commands line by line </br>

```bash
 version=3.28
 build=1
 limit=3.20
 result=$(echo "$version >= $limit" | bc -l)
 os=$([ "$result" == 1 ] && echo "linux" || echo "Linux")
 mkdir ~/temp
 cd ~/temp
 wget https://cmake.org/files/v$version/cmake-$version.$build-$os-x86_64.sh
 sudo mkdir /opt/cmake
 sudo sh cmake-$version.$build-$os-x86_64.sh --prefix=/opt/cmake #(Type "y" to accept the license agreement and type "n" to forego installing inside the subdirectory)
 cmake --version #(expected output is "cmake version 3.28.1") 
```

## Installing MPI
##### Option 1
Install OpenMPI from your distribution's package manager

##### Option 2
https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html

## Compilation
open a terminal window and cd into the project folder </br>

#### (Release)
```bash
 mkdir build-release
 cd build-release
 cmake -DCMAKE_BUILD_TYPE=Release ..
 cmake --build .
```

#### (Debug)
```bash
 mkdir build-debug
 cd build-debug
 cmake -DCMAKE_BUILD_TYPE=Debug ..
 cmake --build .
```

the executable will be generated inside the respective build-X folder. </br>

## Running
To run the sequential executable, open a terminal window and type </br>
```bash
./sequential <INPUT> <OUTPUT>
```
To run the parallel executable, open a terminal window and type </br>
```bash
mpirun -n <N> ./parallel <INPUT> <OUTPUT>
```

The following explains the arguments and their format.
* N: Number of cores to allocate to the program
* INPUT: Name of input image file
* OUTPUT: Name of output image file
</br>
