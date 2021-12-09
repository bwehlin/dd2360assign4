
#define MAX_I 100
#include <stdio.h>

int main(int argc, char** argv)
{
  float* asdf;
#pragma acc parallel loop
  for (int i = 0; i < MAX_I; ++i)
  {
    asdf[i] = i;
  }
}

