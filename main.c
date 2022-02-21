#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // for timing
#include <immintrin.h>

#define N 1000000

double now()
{
    // Retourne l'heure actuelle en secondes
    struct timeval t;
    double f_t;
    gettimeofday(&t, NULL);
    f_t = t.tv_usec;
    f_t = f_t / 1000000.0;
    f_t += t.tv_sec;
    return f_t;
}

void fill_array(float *buf, int n)
{
    for (int i = 0; i < n; i++)
    {
        buf[i] = (float)rand() / RAND_MAX;
    }
}

double rnorm(float *U, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += sqrt(U[i]);
    }
    return sum;
}

double vect_rnorm(float *a, int n)
{
    double sum;
    __m256 *ptr = (__m256 *)a;

    int nb_iters = n / 8;
    for (int i = 0; i < nb_iters; i++, ptr++, a += 8)
    {
        _mm256_store_ps(a, _mm256_sqrt_ps(*ptr));
        for (int j = 0; j < 8; j++)
        {
            sum += a[j];
        }
    }
    return sum;
}

void copy_array(float *origin, float *copy, int n)
{
    for (int i = 0; i < N; i++)
    {
        copy[i] = origin[i];
    }
}

int main()
{
    float U[N] = {0.0};
    double res_seq, res_vect;
    double time_seq, time_vect;

    fill_array(U, N);

    float U_copy_vect[N] __attribute__((aligned(32)));
    copy_array(U, U_copy_vect, N);

    time_seq = now();
    res_seq = rnorm(U, N);
    time_seq = now() - time_seq;

    time_vect = now();
    res_vect = vect_rnorm(U_copy_vect, N);
    time_vect = now() - time_vect;

    printf("VALEURS\n");
    printf("Sequentiel (scalaire: %lf vectoriel: %lf) Parallèle (nb_thread: ... scalaire ... vectoriel ...)\n", res_seq, res_vect);
    printf("TEMPS D'EXECUTION\n");
    printf("Sequentiel (scalaire: %lf vectoriel: %lf) Parallèle (nb_thread: ... scalaire ... vectoriel ...)\n", time_seq, time_vect);
    printf("Accélération (vectoriel: ... multithread: ... vectoriel + multithread ...)\n");
}