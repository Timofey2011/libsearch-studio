A Bloom filter is a compact probabilistic data structure that answers set
membership with no false negatives and a tunable false-positive rate. It stores
a bit array of m bits; inserting an element sets k bit positions chosen by k
independent hash functions.

To query, the same k hash functions are applied: if any addressed bit is zero
the element is definitely absent, and if all are one the element is probably
present. False positives arise when unrelated insertions happen to set all k
positions.

Choosing the number of hash functions is a trade-off: the optimal k is
(m divided by n) times the natural logarithm of two, where n is the expected
number of elements. Too few hash functions waste the bit array's resolution;
too many saturate the array quickly and inflate the false-positive rate.

Counting Bloom filters replace bits with small counters so deletions become
possible, at several times the memory. Blocked Bloom filters pack each
element's bits into one cache line for speed on modern CPUs.
