#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // for timing

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
    double res_seq;
    double time_seq;

    time_seq = now();
    fill_array(U, N);
    time_seq = now() - time_seq;

    res_seq = rnorm(U, 5);
    printf("VALEURS\n");
    printf("Sequentiel (scalaire: %lf vectoriel: ...) Parallèle (nb_thread: ... scalaire ... vectoriel ...)\n", res_seq);
    printf("TEMPS D'EXECUTION\n");
    printf("Sequentiel (scalaire: %lf vectoriel: ...) Parallèle (nb_thread: ... scalaire ... vectoriel ...)\n", time_seq);
    printf("Accélération (vectoriel: ... multithread: ... vectoriel + multithread ...)\n");
}