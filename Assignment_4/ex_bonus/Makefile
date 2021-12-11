
exbonus.opencl: exercise_bonus.c
	nvcc -DUSE_OPENCL -O3 -lOpenCL -o exbonus.opencl exercise_bonus.c
	pgcc -DUSE_ACC -acc -ta=tesla:cc3x,nvidia,lineinfo -Minfo=accel -O3 -o exbonus.openacc exercise_bonus.c

.PHONY: all
all: exbonus.opencl exbonus.openacc
