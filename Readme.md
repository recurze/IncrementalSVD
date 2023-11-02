## Motivation

As part of a school assignment, I was writing [RobustPCA](https://en.wikipedia.org/wiki/Robust_principal_component_analysis) to separate the background (low-rank, still) and the player (sparse, moving) from a simple basketball video. As one would expect, SVD was the most time consuming part of a iteration. The video matrix (M, m pixels x n frames) was thin (I'm jealous) and long. So, I decided to write this routine to see if I can speed up the process. In the end, I decided to simply throw more cpus at `scipy.linalg.svd`.

## IncrementalSVD

I implement [this research paper](https://www.merl.com/publications/docs/TR2002-24.pdf) here.

## TODO

Speed comparison and some graphs maybe.
