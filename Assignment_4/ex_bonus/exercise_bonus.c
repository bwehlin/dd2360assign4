// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

#ifdef USE_OPENCL
// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) { fprintf(stderr,"Error on line %d: %s\n",__LINE__, clGetErrorString(err)); return EXIT_FAILURE; }
#endif

#ifndef bool
#define bool int
#define true 1
#define false 0
#endif


#ifdef USE_OPENCL
// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);
#endif

typedef struct
{
  float pos[3];
  float vel[3];
} Particle;

void print_time(const char* op, struct timeval* start, struct timeval* end)
{
  float ms = (float)(end->tv_sec - start->tv_sec) * 1e3f + (float)(end->tv_usec - start->tv_usec) / 1e3f;
  printf("%s took %.3f ms", op, ms);
}

#ifdef USE_OPENCL
const char *lorentz_kernel = R"~( 
  typedef struct
  {
    float pos[3];
    float vel[3];
  } Particle;

  __kernel
  void lorentz(__global Particle* particles, float dt, int n)
  {
    float sigma = 10.f;
    float rho = 28.f;
    float beta = 8.f/3.f;

    int idx = get_global_id(0);
    if (idx >= n)
    {
      return;
    }

    __global Particle* p = &particles[idx];

    p->vel[0] = sigma * (p->pos[1] - p->pos[0]);
    p->vel[1] = p->pos[0] * (rho - p->pos[2]) - p->pos[1];
    p->vel[2] = p->pos[0] * p->pos[1] - beta * p->pos[2];

    for (int dim = 0; dim < 3; ++dim)
    {
      p->pos[dim] = p->pos[dim] +  dt * p->vel[dim]; 
    }
  }
)~";
#endif


void lorentz_cpu(Particle* particles, float dt, int n_part, int n_iter)
{
  float sigma = 10.f;
  float rho = 28.f;
  float beta = 8.f/3.f;

#pragma acc data copy(particles[0:n_part])

  for (int j = 0; j < n_iter; ++j)
  {
#ifdef USE_ACC
#pragma acc parallel loop
#endif
    for (int i = 0; i < n_part; ++i)
    {
      Particle* p = &particles[i];

      p->vel[0] = sigma * (p->pos[1] - p->pos[0]);
      p->vel[1] = p->pos[0] * (rho - p->pos[2]) - p->pos[1];
      p->vel[2] = p->pos[0] * p->pos[1] - beta * p->pos[2];

      for (int dim = 0; dim < 3; ++dim)
      {
        p->pos[dim] = p->pos[dim] +  dt * p->vel[dim]; 
      }
    }
  }
}

Particle* create_particles(size_t n)
{
  Particle* arr = (Particle*) malloc(sizeof(Particle) * n);
  if (!arr)
  {
    return NULL;
  }
  for (size_t i = 0ul; i < n; ++i)
  {
    for (size_t dim = 0; dim < 3; ++dim)
    {
      arr[i].pos[dim] = (float)(rand()) / (float)(RAND_MAX);
    }
  }
  return arr;
}

#ifdef USE_OPENCL
int gpu_main(int argc, char** args)
{
  int n_particles = atoi(args[1]);
  int n_iter = atoi(args[2]);
  int blocksz = atoi(args[3]);
  int write_output = atoi(args[4]);
 
  int transfer_data = argc == 6 ? atoi(args[5]) : 0;

  printf("Running with %d particles over %d iterations...\n", n_particles, n_iter);

  Particle* particles = create_particles(n_particles);
  if (!particles)
  {
    printf("Out of memory!\n");
    return EXIT_FAILURE;
  }

  float dt = 0.001f;

  cl_platform_id * platforms; cl_uint     n_platform;

  // Find OpenCL Platforms
  cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
  err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

  // Find and sort devices
  cl_device_id *device_list; cl_uint n_devices;
  err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
  device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
  err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);
  
  // Create and initialize an OpenCL context
  cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

  // Create a command queue
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err); 

  /* Insert your own code here */
 
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&lorentz_kernel, NULL, &err);CHK_ERROR(err);
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

  if (err != CL_SUCCESS)
  {
    size_t logSz;
    err = clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSz);CHK_ERROR(err);
    char* buildLog = (char*)malloc(logSz*sizeof(char));
    err = clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, logSz, buildLog, NULL);CHK_ERROR(err);
    printf("BUILD FAILED! Build log: %s", buildLog);
    return EXIT_FAILURE;
  }

  cl_kernel kernel = clCreateKernel(program, "lorentz", &err);CHK_ERROR(err);
  cl_mem particles_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, n_particles*sizeof(Particle), NULL, &err);CHK_ERROR(err);

  cl_int n_dev = n_particles;
  cl_float dt_dev = dt;


  err = clEnqueueWriteBuffer(cmd_queue, particles_dev, CL_TRUE, 0, n_particles*sizeof(Particle), particles, 0, NULL, NULL);CHK_ERROR(err);

  FILE* fl = NULL;
  
  if (write_output)
  {
    fl = fopen("sims.csv", "w");
    if (!fl)
    {
      printf("Error: %s\n", strerror(errno));
      return EXIT_FAILURE;
    }
  }

  struct timeval begin = {0}, end = {0};
  gettimeofday(&begin, NULL);
  for (size_t i = 0; i < n_iter; ++i)
  {
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&particles_dev);
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_float), (void*)&dt_dev);
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&n_dev);

    size_t n_workitem[1] = {((n_particles+blocksz-1)/blocksz)*blocksz};
    size_t workgroup_size[1] = {blocksz};
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, n_workitem, workgroup_size, 0, NULL, NULL);CHK_ERROR(err);

    if (write_output || transfer_data)
    {
      err = clEnqueueReadBuffer(cmd_queue, particles_dev, CL_TRUE, 0, n_particles*sizeof(Particle), particles, 0, NULL, NULL);

      err = clFlush(cmd_queue);CHK_ERROR(err);
      err = clFinish(cmd_queue);CHK_ERROR(err);
    }

    if (write_output)
    {
      fprintf(fl, "%f", (float)i * dt);
      for (int j = 0; j < 3; ++j)
      {
        Particle* part = &particles[j];
        fprintf(fl, ", %f, %f, %f", part->pos[0], part->pos[1], part->pos[2]);
      }
      fprintf(fl, "\n");
    }
  }
  err = clFinish(cmd_queue);CHK_ERROR(err);
  gettimeofday(&end, NULL);
  print_time("GPU", &begin, &end);

  // Finally, release all that we have allocated.
  err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
  err = clReleaseContext(context);CHK_ERROR(err);

  err = clReleaseMemObject(particles_dev);CHK_ERROR(err);

  free(platforms);
  free(device_list);
  free(particles);

  return EXIT_SUCCESS;
}
#endif

int cpu_main(int argc, char** args)
{
  int n_particles = atoi(args[1]);
  int n_iter = atoi(args[2]);

  int use_acc = argc == 4;

  float dt = 0.001f;

  Particle* particles = create_particles(n_particles);

  struct timeval begin, end;
  gettimeofday(&begin, NULL);
  lorentz_cpu(particles, dt, n_particles, n_iter);
  gettimeofday(&end, NULL);

  print_time("CPU", &begin, &end);

  free(particles);
}

int main(int argc, char **argv)
{
  if (argc == 3)
  {
    return cpu_main(argc, argv);
  }
#ifdef USE_OPENCL
  else if (argc == 5 || argc == 6)
  {
    return gpu_main(argc, argv);
  }
#endif
  else
  {
    printf("usage: %s n_particles n_iterations [block_sz write_output]\n", argv[0]);
    return EXIT_FAILURE;
  }
}

#ifdef USE_OPENCL

// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}

#endif

