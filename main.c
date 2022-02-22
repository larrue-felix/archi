#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // for timing
#include <immintrin.h>
#include <stdint.h>
#include <inttypes.h>

#define N 1048576
#define NUM_THREADS 2

double g_res_thread = 0;
pthread_mutex_t mutexsum_thread;

double g_res_thread_vect = 0;
pthread_mutex_t mutexsum_thread_vect;

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

float sum8(__m256 x)
{
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

double vect_rnorm(float *a, int n)
{
    double sum;
    __m256 *ptr = (__m256 *)a;

    int nb_iters = n / 8;
    for (int i = 0; i < nb_iters; i++, ptr++)
    {
        sum += sum8(_mm256_sqrt_ps(*ptr));
    }
    return sum;
}

void *thread_seq(void *threadarg)
{
    struct thread_data *my_data;
    my_data = (struct thread_data *)threadarg;
    double result;
    uintptr_t thread_id = my_data->thread_id;
    int start_index = my_data->start_index;
    int end_index = my_data->end_index;
    float *U = my_data->arr;

    for (int i = start_index; i < end_index; i++)
    {
        result += sqrt(U[i]);
    }
    pthread_mutex_lock(&mutexsum_thread);
    g_res_thread += result;
    pthread_mutex_unlock(&mutexsum_thread);

    pthread_exit((void *)thread_id);
}

void *thread_vect(void *threadarg)
{
    struct thread_data *my_data;
    my_data = (struct thread_data *)threadarg;
    double result;
    uintptr_t thread_id = my_data->thread_id;
    int start_index = my_data->start_index;
    int end_index = my_data->end_index;
    float *U = my_data->arr;

    __m256 *ptr = (__m256 *)U;

    for (int i = start_index; i < end_index; i += 8, ptr++)
    {
        result += sum8(_mm256_sqrt_ps(*ptr));
    }
    pthread_mutex_lock(&mutexsum_thread_vect);
    g_res_thread_vect += result;
    pthread_mutex_unlock(&mutexsum_thread_vect);

    pthread_exit((void *)thread_id);
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
    void *thread_routine;

    thread_routine = (mode == 1) ? thread_vect : thread_seq;
    for (int t = 0; t < nb_threads; t++)
    {
        struct thread_data data;
        thread_data_array[t].thread_id = t;
        thread_data_array[t].start_index = t * (n / nb_threads);
        thread_data_array[t].end_index = (t + 1) * (n / nb_threads);
        thread_data_array[t].arr = U;

        rc = pthread_create(&thread[t], &attr, thread_routine, (void *)&thread_data_array[t]);
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

    s = (mode == 1) ? g_res_thread_vect : g_res_thread;
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
    float *U, *U_copy_vect, *U_copy_thread_vect;
    U = malloc(N * sizeof(float));
    U_copy_vect = aligned_alloc(32, N * sizeof(float));
    U_copy_thread_vect = aligned_alloc(32, N * sizeof(float));

    double res_seq, res_vect, res_thread, res_thread_vect;
    double time_seq, time_vect, time_thread, time_thread_vect;
    double acc_vect, acc_thread, acc_thread_vect;

    fill_array(U, N);

    copy_array(U, U_copy_vect, N);
    copy_array(U, U_copy_thread_vect, N);

    time_seq = now();
    res_seq = rnorm(U, N);
    time_seq = now() - time_seq;

    time_vect = now();
    res_vect = vect_rnorm(U_copy_vect, N);
    time_vect = now() - time_vect;

    time_thread = now();
    res_thread = rnormPar(U, N, NUM_THREADS, 0);
    time_thread = now() - time_thread;

    time_thread_vect = now();
    res_thread_vect = rnormPar(U_copy_thread_vect, N, NUM_THREADS, 1);
    time_thread_vect = now() - time_thread_vect;

    acc_vect = (time_seq / time_vect);
    acc_thread = (time_seq / time_thread);
    acc_thread_vect = (time_seq / time_thread_vect);

    printf("VALEURS\n");
    printf("Sequentiel (scalaire: %lf vectoriel: %lf) Parallèle (nb_thread: %i scalaire %lf vectoriel %lf)\n", res_seq, res_vect, NUM_THREADS, res_thread, res_thread_vect);
    printf("TEMPS D'EXECUTION\n");
    printf("Sequentiel (scalaire: %lf vectoriel: %lf) Parallèle (nb_thread: %i scalaire %lf vectoriel %lf)\n", time_seq, time_vect, NUM_THREADS, time_thread, time_thread_vect);
    printf("Accélération (vectoriel: %lf multithread: %lf vectoriel + multithread %lf)\n", acc_vect, acc_thread, acc_thread_vect);
}