#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // for timing
#include <immintrin.h>

#define N 1000000
#define NUM_THREADS 2

double res = 0;
pthread_mutex_t mutexsum;

struct thread_data
{
    int thread_id;
    int start_index;
    int end_index;
    float *arr;
};

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

void *thread_seq(void *threadarg)
{
    struct thread_data *my_data;
    my_data = (struct thread_data *)threadarg;
    double result;
    int start_index = my_data->start_index;
    int end_index = my_data->end_index;
    float *U = my_data->arr;

    for (int i = start_index; i < end_index; i++)
    {
        result += sqrt(U[i]);
    }
    pthread_mutex_lock(&mutexsum);
    res += result;
    pthread_mutex_unlock(&mutexsum);
}

double rnormPar(float *U, int n, int nb_threads, int mode)
{
    pthread_t thread[NUM_THREADS];
    pthread_attr_t attr;
    int rc;
    long t;
    void *status;
    double s;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    struct thread_data thread_data_array[nb_threads];
    for (int t = 0; t < nb_threads; t++)
    {
        struct thread_data data;
        thread_data_array[t].thread_id = t;
        thread_data_array[t].start_index = t * (n / nb_threads);
        thread_data_array[t].end_index = (t + 1) * (n / nb_threads);
        thread_data_array[t].arr = U;

        rc = pthread_create(&thread[t], &attr, thread_seq, (void *)&thread_data_array[t]);
        if (rc)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    pthread_attr_destroy(&attr);
    for (t = 0; t < nb_threads; t++)
    {
        rc = pthread_join(thread[t], &status);
        if (rc)
        {
            printf("ERROR: pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    s = res;
    return s;
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
    double res_seq, res_vect, res_thread;
    double time_seq, time_vect, time_thread;

    fill_array(U, N);

    float U_copy_vect[N] __attribute__((aligned(32)));
    copy_array(U, U_copy_vect, N);

    time_seq = now();
    res_seq = rnorm(U, N);
    time_seq = now() - time_seq;

    time_vect = now();
    res_vect = vect_rnorm(U_copy_vect, N);
    time_vect = now() - time_vect;

    time_thread = now();
    res_thread = rnormPar(U, N, NUM_THREADS, 0);
    time_thread = now() - time_thread;

    printf("VALEURS\n");
    printf("Sequentiel (scalaire: %lf vectoriel: %lf) Parallèle (nb_thread: %i scalaire %lf vectoriel ...)\n", res_seq, res_vect, NUM_THREADS, res_thread);
    printf("TEMPS D'EXECUTION\n");
    printf("Sequentiel (scalaire: %lf vectoriel: %lf) Parallèle (nb_thread: %i scalaire %lf vectoriel ...)\n", time_seq, time_vect, NUM_THREADS, time_thread);
    printf("Accélération (vectoriel: ... multithread: ... vectoriel + multithread ...)\n");
}