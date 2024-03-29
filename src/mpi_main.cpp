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
#define KERNEL_DIMENSION 3
#define THRESHOLD 40
#define USE_THRESHOLD 1

//Do not use global variables

void par_edgeDetection(uint8_t *input_image, int width, int height, int zone, int start, int rank);
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION], float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]);

int main(int argc,char* argv[]) {
    if(argc != 3){
        std::cerr << "Invalid number of arguments, aborting...";
        exit(1);
    }

    int m_rank, comm_sz;
    int width, height, bpp;
    int buf_size;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    std::string input;
    std::string output;
    uint8_t *input_image = nullptr;

    if(m_rank == 0) {
        /* Prepend path to input and output filenames */
        input = RESOURCES_PATH;
        output = PARALLEL_OUTPUT_PATH;
        input = input + argv[1];
        output = output + argv[2];

        /* Read image in grayscale */
        input_image = stbi_load(input.c_str(), &width, &height, &bpp, CHANNEL_NUM);

        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + input + "\"\n";
            std::cerr << "Aborting...\n";
            exit(1);
        }

        //stbi_write_jpg("z.c_str()", width, height, CHANNEL_NUM, input_image, 100);

        buf_size = width * height;

        printf("Width: %d  Height: %d  BPP: %d \n",width, height, bpp);
        printf("Input: %s , Output: %s  \n",input.c_str(), output.c_str());
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const int zone = height / comm_sz;

    //std::cout << "Process: " << m_rank << " width: " << width << " height: " << height << " zone:" << zone << " buf: " << buf_size << std::endl;

    uint8_t *temp_out = (uint8_t*) malloc( width * zone * sizeof(uint8_t)); // NOLINT(*-use-auto)

    MPI_Barrier(MPI_COMM_WORLD);
    // start the timer
    double time1= MPI_Wtime();

    MPI_Scatter(input_image, zone * width, MPI_UINT8_T, temp_out, zone * width, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    par_edgeDetection(temp_out, width, height, zone, 0, m_rank);

    uint8_t *out = nullptr;
    if(m_rank == 0) {
         out = (uint8_t*) malloc(width * height * sizeof(uint8_t)); // NOLINT(*-use-auto)
    }

    MPI_Gather(temp_out, zone * width, MPI_UINT8_T, out, zone * width, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    double time2= MPI_Wtime();
    printf("Elapsed time: %lf \n",time2-time1);
    if(m_rank == 0) {
        // Storing the image
        stbi_write_jpg(output.c_str(), width, height, CHANNEL_NUM, out, 100);
        stbi_image_free(input_image);
    }

    free(temp_out);
    free(out);

    MPI_Finalize();
    return 0;
}

/* Apply Sobel's Operator  */
void par_edgeDetection(uint8_t *input_image, int width, int height, int zone, int start, int rank) {
    std::string z = std::to_string(rank) + ".jpg";
    z = PARALLEL_OUTPUT_PATH + z;
    //stbi_write_jpg(z.c_str(), width, zone, CHANNEL_NUM, input_image, 100);
    /* Declare Kernels */
    float sobelX[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {-1, 0, 1},{-2, 0, 2},{-1, 0, 1} };
    float sobelY[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {-1, -2, -1},{0, 0, 0},{1, 2, 1} };
    [[maybe_unused]] float gaussianBlur[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {1, 2, 1},{2, 4, 2},{1, 2, 1} };
    [[maybe_unused]] float boxBlur[KERNEL_DIMENSION][KERNEL_DIMENSION] = { {1, 1, 1},{1, 1, 1},{1, 1, 1} };

    /* Declare helper variables */
    float slider[KERNEL_DIMENSION][KERNEL_DIMENSION];
    float gx = 0;
    float gy = 0;

    /* Allocate temporary memory to construct final image */
    uint8_t *output_image = (uint8_t*) malloc(width * zone * sizeof(uint8_t)); // NOLINT(*-use-auto)
    //uint8_t *output_image_blur_u8 = (uint8_t*) malloc(width * height * sizeof(uint8_t)); // NOLINT(*-use-auto)
    //uint8_t *gradientX_u8 = (uint8_t*) malloc(width * height * sizeof(uint8_t)); // NOLINT(*-use-auto)
    //uint8_t *gradientY_u8 = (uint8_t*) malloc(width * height * sizeof(uint8_t)); // NOLINT(*-use-auto)
    //uint8_t *angle_u8 = (uint8_t*) malloc(width * height * sizeof(uint8_t)); // NOLINT(*-use-auto)

    /* Iterate through all pixels */
	for(int y = start; y < zone; ++y) {
	    //std::cout << "Entered Sobel from index: " << start << " till index: " << start + zone << std::endl;
		for(int x = 0; x < width; ++x) {
            for(int wy = 0; wy < KERNEL_DIMENSION; ++wy) {
                for(int wx = 0; wx < KERNEL_DIMENSION; ++wx) {
                    /* Duplicate opposite edge values if on barrier pixels */
                    int xIndex = (x + wx - 1);
                    int yIndex = (y + wy - 1);
                    if(xIndex < 0)
                        xIndex = -xIndex;
                    if(yIndex < 0)
                        yIndex = -yIndex;
                    if(xIndex >= width)
                        xIndex = xIndex - KERNEL_DIMENSION - 1;
                    if(yIndex >= zone)
                        yIndex = yIndex - KERNEL_DIMENSION - 1;

                    /* Build up moving window */
                    slider[wy][wx] = input_image[xIndex + yIndex * width];
                }
            }

            /* Convolve moving window with kernels (Sobel X and Y gradient) */
            gx = convolve(slider, sobelX);
            gy = convolve(slider, sobelY);
            //gradientX_u8[x + y * width] = (uint8_t) gx;
            //gradientY_u8[x + y * width] = (uint8_t) gy;
            float magnitude = sqrt(gx * gx + gy * gy);

#if USE_THRESHOLD
            /* Clamp down color values if */
            output_image[x + y * width] = magnitude > THRESHOLD ? 255 : 0;
#else
            /* Use whatever value outputted from square root */
            output_image[x + y * width] = (uint8_t) magnitude;
#endif
            //angle_u8[x + y * width] = (uint8_t) atan2(gy, gx);
		}
	}

    /* copy zone to input array */
    for(int y = start; y < zone; ++y) {
        for(int x = 0; x < width; ++x) {
            input_image[x + y * width] = output_image[x + y * width];
        }
    }
/*
    std::string z = std::to_string(start) + ".jpg";
    z = PARALLEL_OUTPUT_PATH + z;
    stbi_write_jpg(z.c_str(), width, zone, CHANNEL_NUM, input_image, 100);
    */
/*
    stbi_write_jpg(SEQUENTIAL_OUTPUT_PATH "sobel.jpg", width, height, CHANNEL_NUM, output_image_u8, 100);
    stbi_write_jpg(SEQUENTIAL_OUTPUT_PATH "gradientX.jpg", width, height, CHANNEL_NUM, gradientX_u8, 100);
    stbi_write_jpg(SEQUENTIAL_OUTPUT_PATH "gradientY.jpg", width, height, CHANNEL_NUM, gradientY_u8, 100);
    stbi_write_jpg(SEQUENTIAL_OUTPUT_PATH "angle.jpg", width, height, CHANNEL_NUM, angle_u8, 100);
*/

    /* De-allocate dynamic arrays */
    free(output_image);
    //free(output_image_blur_u8);
    //free(gradientX_u8);
    //free(gradientY_u8);
    //free(angle_u8);

}

/* Convolve slider across kernel (multiply and sum values of two arrays) */
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION], float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]) {
    float sum = 0;
    for(int y = 0; y < KERNEL_DIMENSION; ++y) {
        for(int x = 0; x < KERNEL_DIMENSION; ++x) {
            sum = sum + slider[x][y] * kernel[x][y];
        }
    }

    return sum;
}
