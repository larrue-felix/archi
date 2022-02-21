#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

double rnorm(float *U, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += sqrt(U[i]);
    }
    return sum;
}

void fill_array(float *buf, int n)
{
    for (int i = 0; i < n; i++)
    {
        buf[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    float U[N] = {0.0};
    fill_array(U, N);
    float res = rnorm(U, 5);
    printf("Res is %lf", res);
}