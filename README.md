$`\Phi`$ID -- Integrated Information Decomposition
=============================================

This library contains Octave (v5.0.0) and Matlab (>R2016a) functions to compute
Integrated Information Decomposition ($`\Phi`$ID) in continuous and discrete data
using Barrett's Minimum Mutual Info (MMI) redundancy function. This code implements
the measures used for further analysis in:

* Luppi, Mediano, Rosas, Holland, Fryer, O'Brien, Rowe, Menon, Bor and
  Stamatakis (2022). A synergistic core for human brain evolution and
  cognition.
  [Nature Neuroscience] (preprint DOI: https://www.biorxiv.org/content/10.1101/2020.09.22.308981v1)

For details details about Integrated Information Decomposition, we refer the reader 
to the original methodological paper:

* Mediano*, Rosas*, Luppi, Carhart-Harris, Bor, Seth and Barrett (2021). Beyond integrated
  information: A taxonomy of information dynamics phenomena. arXiv:2109.13186

If you haven't read this paper, or the Williams & Beer paper below, you are
strongly encouraged to do so before using this software.


Usage
-----

This library implements PhiID in bipartite systems. 
To compute $`\Phi`$ID on the time-delayed mutual information (TDMI) of a time 
series dataframe X with 2 dimensions (channels/regions) and 1000 samples (timepoints)
(in this example, random timeseries are used for X), run:

```octave
X = randn([2, 1000]);
atoms = PhiIDFull(X);
```

The resulting `struct` contains all the PhiID atoms for the input data (see
`help PhiIDFull.m` for details on atom names). The same usage applies for the discrete estimateor, 
which is called with the function PhiIDFullDiscrete.

For the paper by Luppi et al., the atom that corresponds to (persistent) 
synergy is termed sts, and the atom that corresponds to (persistent) redundancy
is termed rtr.

Given BOLD signals with N regions by T time-points, called my_BOLD, an N-by-N matrix of synergy
between each pair of regions A and B can be obtained by filling the cell corresponding to 
(A,B) with the value of atoms.sts obtained from running the PhiIDFull command
on the BOLD signals from regions A and B. The same principle applies for 
obtaining a matrix of redundancy, using atoms.rtr as the value.
Note that as mentioned above, both atoms are computed together as part of the 
Integrated Information Decomposition performed by the PhiIDFull command.


```octave
for row = 1:size(my_BOLD,1)
    for col = 1:size(my_BOLD,1)

        atoms = PhiIDFull([my_BOLD(row,:); my_BOLD(col, :)]);
        synergy_mat(row,col) = atoms.sts;
        redundancy_mat(row,col) = atoms.rtr;

    end
end
```

The redundancy-to-synergy gradient can then be obtained as follows:

```octave
gradient = floor(tiedrank(mean(synergy_mat))) - floor(tiedrank(mean(redundancy_mat)));
```

Note that the TDMI is by definition the sum of all atoms. 
TDMI-normalised values can be obtained by dividing the quantity of interest by this value.

In general, some atoms can be negative, but certain combinations of them are guaranteed to
be non-negative, e.g. transfer entropy:

```octave
TE = atoms.xtr + atoms.str + atoms.xty + atoms.sty
```


Most of the functions in this library depend on the Java Information Dynamics
Toolbox (JIDT) by Lizier, Ozdemir & Mediano:

https://www.github.com/jlizier/jidt

For convenience, an up-to-date JIDT jar is distributed with this code.

Feature requests and bug reports are warmly welcome. Email Pedro Mediano (see
email in the paper above) for any questions or comments.


Further reading
---------------

* Williams PL, Beer RD (2010). Nonnegative decomposition of multivariate
  information. arXiv:1004.2515.

* Barrett AB (2015). Exploration of synergistic and redundant information
  sharing in static and dynamical Gaussian systems. Physical Review E.


(C) Pedro Mediano, Fernando Rosas and Andrea Luppi, 2019-23

