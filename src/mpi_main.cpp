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

#include <unistd.h>
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

void par_edgeDetection(uint8_t *input_image, int width, int height, int rank, int comm_sz);
float convolve(float slider[KERNEL_DIMENSION][KERNEL_DIMENSION], float kernel[KERNEL_DIMENSION][KERNEL_DIMENSION]);

int main(int argc,char* argv[]) {
    if(argc != 3){
        std::cerr << "Invalid number of arguments, aborting...";
        exit(1);
    }

    MPI_Init(&argc,&argv);
    int m_rank, comm_sz, width, height, bpp, buf_size;
    std::string input, output;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    uint8_t *input_image = nullptr;

    if(m_rank == 0) {
        /* Prepend path to input and output filenames */
        input = RESOURCES_PATH;
        output = PARALLEL_OUTPUT_PATH;
        input = input + argv[1];
        output = output + argv[2];

        /* Read image in grayscale */
        input_image = stbi_load(input.c_str(), &width, &height, &bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + input + "\"\n";
            std::cerr << "Aborting...\n";
            exit(1);
        }

        buf_size = width * height;
        printf("Width: %d  Height: %d  BPP: %d \n",width, height, bpp);
        printf("Input: %s , Output: %s  \n",input.c_str(), output.c_str());
    }

    /* Broadcast updated variables to other processes */
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    /* Partition work among processes */
    const int zone = height / comm_sz;
    const int wh = width * height;
    const int zw = zone * width;

    /* Allocate a temporary output buffer for each process */
    uint8_t *temp_out = (uint8_t*) malloc( (width * (zone + 2)) * sizeof(uint8_t)); // NOLINT(*-use-auto)

    /* Map zones to processes */
    if(m_rank == 0) {
        /* memcpy 0 - zone+2 to process 0 from input */
        memcpy(temp_out, input_image, ((zone + 2)* width) * sizeof(uint8_t));

        for(int dest = 1; dest < comm_sz; ++dest) {
            if(dest != comm_sz - 1) {
                /* Send zone-1 - zone+1 to intermediate processes */
                MPI_Send(&input_image[(zw * dest) - width], (zone + 2) * width, MPI_UINT8_T, dest, 1, MPI_COMM_WORLD);
            }
            else {
                /* Send zone-2 - zone to last process */
                MPI_Send(&input_image[(zw * dest) - (width * 2)], (zone + 2) * width, MPI_UINT8_T, dest, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        MPI_Recv(temp_out, (zone + 2) * width, MPI_UINT8_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Start the timer */
    double time1= MPI_Wtime();

    /* Apply Sobel's operator */
    par_edgeDetection(temp_out, width, zone + 2, m_rank, comm_sz);

    /* Synchronize and stop timer */
    MPI_Barrier(MPI_COMM_WORLD);
    double time2= MPI_Wtime();

    uint8_t *out = nullptr;
    if(m_rank == 0) {
         out = (uint8_t*) malloc(wh * sizeof(uint8_t)); // NOLINT(*-use-auto)
    }

    /* Collect sub-solutions into one buffer on process 0 */
    MPI_Gather(temp_out, zw, MPI_UINT8_T, out, zw, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    if(m_rank == 0) {
        printf("Elapsed time: %lf \n",time2-time1);
        /* Write image to disk */
        stbi_write_jpg(output.c_str(), width, height, CHANNEL_NUM, out, 100);
        stbi_image_free(input_image);
    }

    /* Let go of heap memory */
    free(temp_out);
    if(m_rank == 0) {
        /* Prepend path to input and output filenames */
        std::string par_input = PARALLEL_OUTPUT_PATH;
        std::string seq_input = SEQUENTIAL_OUTPUT_PATH;
        seq_input = seq_input + "seq.jpg";
        par_input = par_input + "par.jpg";
        uint8_t *seq_img;
        uint8_t *par_img;
        int seq_width, seq_height, seq_bpp;
        int par_width, par_height, par_bpp;

        /* Read image in grayscale */
        seq_img = stbi_load(seq_input.c_str(), &seq_width, &seq_height, &seq_bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + seq_input + "\"\n";
            std::cerr << "Aborting...\n";
            exit(1);
        }

        par_img = stbi_load(par_input.c_str(), &par_width, &par_height, &par_bpp, CHANNEL_NUM);

        /* If image could not be opened, Abort */
        if(stbi_failure_reason()) {
            std::cerr << stbi_failure_reason() << " \"" + par_input + "\"\n";
            std::cerr << "Aborting...\n";
            exit(1);
        }

        /* Make sure sequential and parallel outputs are the same */
        int err_cnt = 0;
        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                if(par_img[x + y * width] != seq_img[x + y * width]) {
                    ++err_cnt;
                }
            }
        }
        if(err_cnt == 0)
            std::cout << "Sequential and Parallel images are identical\n";
        else
            std::cout << err_cnt << " pixels are mismatched\n";

        free(out);
    }

    MPI_Finalize();
    return 0;
}

/* Apply Sobel's Operator  */
void par_edgeDetection(uint8_t *input_image, int width, int height, int rank, int comm_sz) {

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
    uint8_t *output_image = (uint8_t*) malloc(width * (height) * sizeof(uint8_t)); // NOLINT(*-use-auto)

    /* Iterate through all pixels */
	for(int y = 0 ; y < height; ++y) {
	    //std::cout << "Entered Sobel from index: " << y << " till index: " << height - 1 << std::endl;
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
                    if(yIndex >= height)
                        yIndex = yIndex - KERNEL_DIMENSION - 1;

                    /* Build up moving window */
                    slider[wy][wx] = input_image[xIndex + yIndex * width];
                }
            }

            /* Convolve moving window with kernels (Sobel X and Y gradient) */
            gx = convolve(slider, sobelX);
            gy = convolve(slider, sobelY);
            float magnitude = sqrt(gx * gx + gy * gy);

#if USE_THRESHOLD
            /* Clamp down color values if */
            output_image[x + (y) * width] = magnitude > THRESHOLD ? 255 : 0;
#else
            /* Use whatever value outputted from square root */
            output_image[x + y * width] = (uint8_t) magnitude;
#endif
            //angle_u8[x + y * width] = (uint8_t) atan2(gy, gx);
		}
	}

    int offset;
    if(rank == 0) {
        offset = 0;
    }
    else if (rank != comm_sz - 1) {
        offset = 1;
    }
    else {
        offset = 2;
    }

    /* copy zone to input array */
    for(int y = 0 ; y < height - 2; ++y) {
        for(int x = 0; x < width; ++x) {
            input_image[x + (y) * width] = output_image[x + (y + offset) * width];
        }
    }

    //stbi_write_jpg((std::to_string(rank) + ".jpg").c_str(), width, height - 2, CHANNEL_NUM, input_image, 100);

    /* De-allocate dynamic arrays */
    free(output_image);
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
