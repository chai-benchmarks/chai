/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#ifndef _PARTITIONER_H_
#define _PARTITIONER_H_

#ifndef _OPENCL_COMPILER_
#include <iostream>
#ifdef OCL_2_0
#include <atomic>
#endif
#endif

#define STATIC_PARTITIONING 0
#define DYNAMIC_PARTITIONING 1

typedef struct Partitioner {

    int n_tasks;
    int cut;
#ifdef OCL_2_0
    int strategy;
#endif

} Partitioner;

#ifndef _OPENCL_COMPILER_

inline Partitioner partitioner_create(int n_tasks, float alpha) {
    Partitioner p;
    p.n_tasks = n_tasks;
    if(alpha >= 0.0 && alpha <= 1.0) {
        p.cut = p.n_tasks * alpha;
#ifdef OCL_2_0
        p.strategy = STATIC_PARTITIONING;
#endif
    } else {
#ifdef OCL_2_0
        p.strategy = DYNAMIC_PARTITIONING;
#else
        std::cerr << "Illegal value! Alpha must be between 0 and 1!" << std::endl;
        exit(-1);
#endif
    }
    return p;
}

inline int cpu_first(const Partitioner *p, int id
#ifdef OCL_2_0
    ,
    std::atomic_int *worklist) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return worklist->fetch_add(1);
    } else
#else
    ) {
#endif
    {
        return id;
    }
}

inline int cpu_next(const Partitioner *p, int old, int numCPUThreads
#ifdef OCL_2_0
    ,
    std::atomic_int *worklist) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return worklist->fetch_add(1);
    } else
#else
    ) {
#endif
    {
        return old + numCPUThreads;
    }
}

inline bool cpu_more(const Partitioner *p, int old) {
#ifdef OCL_2_0
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return (old < p->n_tasks);
    } else
#endif
    {
        return (old < p->cut);
    }
}

#else

inline int gpu_first(const Partitioner *p, int id, __local int *tmp
#ifdef OCL_2_0
    ,
    __global atomic_int *worklist) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        if(get_local_id(1) == 0 && get_local_id(0) == 0) {
            tmp[0] = atomic_fetch_add(worklist, 1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        return tmp[0];
    } else
#else
    ) {
#endif
    {
        return p->cut + id;
    }
}

inline int gpu_next(const Partitioner *p, int old, int numGPUGroups, __local int *tmp
#ifdef OCL_2_0
    ,
    __global atomic_int *worklist) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        if(get_local_id(1) == 0 && get_local_id(0) == 0) {
            tmp[0] = atomic_fetch_add(worklist, 1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        return tmp[0];
    } else
#else
    ) {
#endif
    {
        return old + numGPUGroups;
    }
}

inline bool gpu_more(const Partitioner *p, int old) { return (old < p->n_tasks); }

#endif

#endif
