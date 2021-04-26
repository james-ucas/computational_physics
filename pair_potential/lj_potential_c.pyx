import cython
import numpy as np

DTYPE = np.float


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_energy_c(double [::1,] x, double sigma=1.0, double epsilon=1.0, int dimensions=3):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double energy = 0.0
    cdef double rija, r2, r6

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            for d in range(dimensions):
                r2 += (x[dimensions*i+d]-x[dimensions*j+d])*(x[dimensions*i+d]-x[dimensions*j+d])
            r6 = 1.0/r2/r2/r2
            energy += 4.0*epsilon*r6*(r6 - 1.0)
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_gradient_c(double [::1,] x, double sigma=1.0, double epsilon=1.0, int dimensions=3):
    cdef Py_ssize_t i, j, d
    cdef int particles = x.size//dimensions
    cdef double rija, r2, r6, gij, gia

    gradient = np.zeros_like(x)
    cdef double[::1] gradient_view = gradient

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            gij = 0.0
            for d in range(dimensions):
                rija = x[dimensions*i+d]-x[dimensions*j+d]
                r2 += rija*rija
            r6 = 1.0/r2/r2/r2
            gij = 24.0*epsilon*r6*(1.0-2.0*r6)/r2
            for d in range(dimensions):
                gia = gij*(x[dimensions*i+d]-x[dimensions*j+d])
                gradient_view[dimensions*i+d] += gia
                gradient_view[dimensions*j+d] -= gia

    return gradient
