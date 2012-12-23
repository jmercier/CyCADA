extern "C"
__global__ void fill(unsigned int *a, unsigned int value, unsigned int size)
{
    const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= size)
        return;

    a[index] = value;
}
