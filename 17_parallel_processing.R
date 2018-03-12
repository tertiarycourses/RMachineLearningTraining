
#With the integration of parallelMap into mlr, it becomes easy to activate the parallel computing capabilities
#already supported by mlr. parallelMap supports all major parallelization backends:
#All you have to do is select a backend by calling one of the parallelStart* functions.
#The first loop mlr encounters which is marked as parallel executable will be
#automatically parallelized. It is good practice to call parallelStop at the end of your script.

library(parallel)
library(parallelMap)
parallelStartSocket(detectCores()-1)


parallelMap:: # look at the functions

## eg > resampling ; hyperparamaeter tuning

parallelStop()
#> Stopped parallelization. All cleaned up.