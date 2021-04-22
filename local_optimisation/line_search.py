def line_search(f, x, p, gx, c=0.5, t=0.5, max_alpha=1):
    """
    Performs a backtracking line search based on the Armijo–Goldstein condition

    :param f: the objective function
    :type f: callable
    :param x: the point from which to start the line search
    :type x: numpy.ndarray
    :param p: the search direction, unit vector
    :type p: numpy.ndarray
    :param gx: the gradient vector at x
    :type gx: numpy.ndarray
    :param c: scales the threshold above which the step is rejected
    :type c: float
    :param t: scales the step upon rejection
    :type t: float
    :param max_alpha: maximum allowed value of alpha
    :type max_alpha: float
    :return: alpha, the size of a good step to take in the direction p
    :rtype: float

    """
    m = p @ gx

    fx = f(x)
    test = 1e-10

    #  this is a horrible hack to fix a bug in HEF line searches
    #  ensures a decreasing search direction but should be unnecessary
    a = f(x + test * p) < fx + c * test * m
    b = f(x - test * p) < fx + c * test * m
    if b and not a:
        p *= -1

    alpha = max_alpha
    while f(x + alpha * p) > fx + c * alpha * m:
        alpha *= t

    return alpha
