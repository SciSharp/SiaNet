#ifndef CALCU_PTHREAD_H
#define CALCU_PTHREAD_H

//#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <error.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <sys/mman.h>
#include <float.h>

#define IM2COL_MAX_CPU_NUMBER 4
#define MAX_CPU_NUMBER 8
#define THREAD_STATUS_SLEEP 2
#define THREAD_STATUS_WAKEUP 4
#define MB
#define WMB
#define YIELDING    sched_yield()

#define BUFFER_SIZE (32 << 20)
#define OUT_OFFSSET (4 << 20)
#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)

extern float* g_buff;

typedef struct {
	void(*routine)(void *, int);
	int  position;
	void* args;
} queue_t;

typedef struct {
	queue_t * volatile queue  __attribute__((aligned(16)));
	volatile long status;
	pthread_mutex_t lock;
	pthread_cond_t wakeup;
} sub_thread_status_t;

void all_sub_pthread_exec(queue_t *queue_Q, int exec_thread_num);
void sub_pthread_init(void);
void sub_pthread_exit(void);
#endif
