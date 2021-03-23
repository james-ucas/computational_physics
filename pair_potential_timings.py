from time import time

import numpy as np


def make_random_configuration(atoms):
    return np.random.rand(atoms * 3).reshape(atoms, 3)


def time_to_calculate_energy(pair_potential_function, potential_function, configuration, args=()):
    t0 = time()
    _ = pair_potential_function(configuration, potential_function, args)
    return time() - t0


def main():
    import matplotlib.pyplot as plt
    from pair_potential import slow_pair_potential, faster_pair_potential, fast_pair_potential
    from pair_potential.lj_potential import lennard_jones_potential, vectorised_lennard_jones_potential

    atom_counts = [10, 20, 50, 100, 200, 500, 1000]
    configurations = [make_random_configuration(atoms) for atoms in atom_counts]
    pair_potential_functions = [slow_pair_potential, faster_pair_potential, fast_pair_potential]
    potential_functions = [lennard_jones_potential, lennard_jones_potential, vectorised_lennard_jones_potential]
    labels = ['slow', 'faster', 'fast']

    arguments = zip(pair_potential_functions, potential_functions)
    timings = np.array([time_to_calculate_energy(ppf, pf, x)
                        for (ppf, pf) in arguments
                        for x in configurations],
                       dtype=float).reshape(3, -1)
    fmt = 'timings for {:n} particles:\nslow\t{:.2f} s\nfaster\t{:.2f} s\nfast\t{:.2f} s'
    print(fmt.format(atom_counts[-1], *timings[:, -1]))

    fig, ax = plt.subplots(1, 1)
    for label, timing in zip(labels, timings):
        ax.plot(atom_counts, timing, 'o-', label=label)
    ax.legend()
    ax.set_xlabel('number of atoms')
    ax.set_ylabel('run time/ s')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('figures/pair_potential_timings.pdf')


if __name__ == '__main__':
    main()
