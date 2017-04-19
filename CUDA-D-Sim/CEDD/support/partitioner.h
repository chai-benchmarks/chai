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

#include <atomic>
#include <iostream>

#define STATIC_PARTITIONING 0
#define DYNAMIC_PARTITIONING 1

// Partitioner definition -----------------------------------------------------

typedef struct CoarseGrainPartitioner {

    int n_tasks;
    int strategy;
    union {
        int cut;                    // Used for static partitioning
        std::atomic_int *worklist;  // Used for dynamic partitioning
    };
    int cpu_current;
    int gpu_current;

} CoarseGrainPartitioner;

// Create a partitioner -------------------------------------------------------

inline CoarseGrainPartitioner partitioner_create(int n_tasks, float alpha, std::atomic_int *worklist) {
    CoarseGrainPartitioner p;
    p.n_tasks = n_tasks;
    if(alpha >= 0.0 && alpha <= 1.0) {
        p.strategy = STATIC_PARTITIONING;
        p.cut      = p.n_tasks * alpha;
    } else {
        p.strategy = DYNAMIC_PARTITIONING;
        p.worklist = worklist;
    }
    return p;
}

// Partitioner iterators: first() ---------------------------------------------

inline int cpu_first(CoarseGrainPartitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->cpu_current = p->worklist->fetch_add(1);
    } else {
        p->cpu_current = 0;
    }
    return p->cpu_current;
}

inline int gpu_first(CoarseGrainPartitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->gpu_current = p->worklist->fetch_add(1);
    } else {
        p->gpu_current = p->cut;
    }
    return p->gpu_current;
}

// Partitioner iterators: more() ----------------------------------------------

inline bool cpu_more(CoarseGrainPartitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return p->cpu_current < p->n_tasks;
    } else {
        return p->cpu_current < p->cut;
    }
}

inline bool gpu_more(CoarseGrainPartitioner *p) {
    return p->gpu_current < p->n_tasks;
}

// Partitioner iterators: next() ----------------------------------------------

inline int cpu_next(CoarseGrainPartitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->cpu_current = p->worklist->fetch_add(1);
    } else {
        p->cpu_current += 1;
    }
    return p->cpu_current;
}

inline int gpu_next(CoarseGrainPartitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->gpu_current = p->worklist->fetch_add(1);
    } else {
        p->gpu_current += 1;
    }
    return p->gpu_current;
}


#endif
