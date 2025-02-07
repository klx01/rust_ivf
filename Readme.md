Trying to implement a vector index myself, for learning purposes.

Uses portable_simd feature, so can only run on nightly.

Implemented a simple version for flat search and ivf_flat index. Things left to try:

* optimise more
* check if better algorithms can be used
* try to use gpu
* quantized IVF
* reimplement in other languages


What was already optimized:
* my kmeans++ initialisation had a problem - each iteration of the loop was getting progressively slower (because i was calculating distances from each picked centroid, so everytime i picked a new centroid, it would become slower). There was no actual need to calculate all of them again and again - that was fixed (at the cost of allocating one more array)
* for both clustering and searching, pretty much all of the time is spent calculating distances. Rewriting this part to use simd made everything literally 10 times faster
* clusters were originally arrays of indices, and clustering gave a smaller speed boost that it should be (i.e. for nlist=64 and nprobe=16 search speedup was not 4 times, but less than 3 times). And after rewriting distances to use simd it actually became slower than flat search. So now clustering rearranges the initial array, and clusters are ranges within that initial array. This now has an expected speed boost (i.e. for nlist=64 and nprobe=16 search speedup is actually 4 times)