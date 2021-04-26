import numpy as np
import cython

DTYPE = np.float


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_energy_c(double [:,::1,] x, double sigma, double epsilon):
    cdef Py_ssize_t i, j, d
    cdef int particles, dimensions
    cdef double energy = 0.0
    cdef double r2, r6
    particles = x.shape[0]
    dimensions = x.shape[1]

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            for d in range(dimensions):
                r2 += (x[i,d]-x[j,d])*(x[i,d]-x[j,d])
            r6 = 1.0/r2/r2/r2
            energy += 4.0*epsilon*r6*(r6 - 1.0)
    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lj_gradient_c(double [:,::1,] x, double sigma, double epsilon):
    cdef Py_ssize_t i, j, d
    cdef int particles, dimensions
    cdef double gij
    cdef double r2, r6
    particles = x.shape[0]
    dimensions = x.shape[1]
    gradient = np.zeros_like(x)
    cdef double[:,::1] gradient_view = gradient

    for i in range(particles):
        for j in range(i+1,particles):
            r2 = 0.0
            gij = 0.0
            for d in range(dimensions):
                r2 += (x[i,d]-x[j,d])*(x[i,d]-x[j,d])
            r6 = 1.0/r2/r2/r2
            gij = 24.0*epsilon*r6*(1.0-2.0*r6)/r2
            for d in range(dimensions):
                gradient[i, d] += gij*(x[i,d]-x[j,d])
                gradient[j, d] -= gij*(x[i,d]-x[j,d])

    return gradient
