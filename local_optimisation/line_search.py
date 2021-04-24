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
    if m >= 0:
        raise ValueError(f"no function decrease guaranteed in search direction p (p @ gx = {p @ gx})")

    fx = f(x)
    alpha = max_alpha
    while f(x + alpha * p) > fx + c * alpha * m:
        alpha *= t

    return alpha
