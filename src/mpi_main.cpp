/**
 *
 * CENG342 Project-1
 *
 * Edge Detection
 *
 * Usage:  main <input.jpg> <output.jpg>
 *
 * @group_id 06
 * @author  Emre
 * @author  Firat
 * @author  Yasin
 * @author  Yousif
 *
 * @version 1.0, 02 March 2024
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "mpi.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1

int main(int argc,char* argv[])
{
    if(argc != 3){
        std::cerr << "Invalid number of arguments, aborting...";
        exit(1);
    }

    MPI_Init(&argc,&argv);
    int width, height, bpp;

    /* Prepend path  to input and output filenames */
    std::string input = RESOURCES_PATH;
    std::string output = PARALLEL_OUTPUT_PATH;
    input = input + argv[1];
    output = output + argv[2];

    // Reading the image in grey colors
    uint8_t* input_image = stbi_load(input.c_str(), &width, &height, &bpp, CHANNEL_NUM);

    printf("Width: %d  Height: %d \n",width,height);
    printf("Input: %s , Output: %s  \n",input.c_str(),output.c_str());

    // start the timer
    double time1= MPI_Wtime();

    //seq_edgeDetection(input_image,width, height,...);

    double time2= MPI_Wtime();
    printf("Elapsed time: %lf \n",time2-time1);

    // Storing the image
    stbi_write_jpg(output.c_str(), width, height, CHANNEL_NUM, input_image, 100);
    stbi_image_free(input_image);

    MPI_Finalize();
    return 0;
}
