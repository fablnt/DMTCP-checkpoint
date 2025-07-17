The files in this folder contain the output of simulations performed on the Leonardo cluster:

- ```LAUNCH1.out```: P3alphaRecommender results without importing problematic libraries from [Recommender_import_list.py](https://github.com/fablnt/DMTCP-checkpoint/blob/master/src/Recommender_import_list.py)
- ```DMTCP1.out```: P3alphaRecommender results using DMTCP without importing problematic libraries from [Recommender_import_list.py](https://github.com/fablnt/DMTCP-checkpoint/blob/master/src/Recommender_import_list.py)
- ```LAUNCH2.out```: P3alphaRecommender results importing problematic libraries from [Recommender_import_list.py](https://github.com/fablnt/DMTCP-checkpoint/blob/master/src/Recommender_import_list.py)
- ```DMTCP2.out```: P3alphaRecommender results using DMTCP importing problematic libraries from [Recommender_import_list.py](https://github.com/fablnt/DMTCP-checkpoint/blob/master/src/Recommender_import_list.py)
- ```LAUNCH3.out```: MatrixFactorization_BPR_Cython_Recommender results without importing problematic libraries from [Recommender_import_list.py](https://github.com/fablnt/DMTCP-checkpoint/blob/master/src/Recommender_import_list.py)
- ```DMTCP3.out```: MatrixFactorization_BPR_Cython_Recommender results using DMTCP without importing problematic libraries from [Recommender_import_list.py](https://github.com/fablnt/DMTCP-checkpoint/blob/master/src/Recommender_import_list.py)

The directories in this folder are the outputs produced by some tests done with ```checkpoint.sh```.
- ```output_test0```:  Simple test of a counter, checkpointing successful as expected.
- ```output_run_test_recommender_successful```: Test of P3AlphaRecommender model checkpointed successfully: the checkpoint has been done during the initial setup of the problem. 
- ```output_run_test_recommender_failed```: Test of P3AlphaRecommender model checkpointed failed: the checkpoint has been done during the training.
