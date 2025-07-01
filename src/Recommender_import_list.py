#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19

@author: Maurizio Ferrari Dacrema
"""



######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects



######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.NegHOSLIM import NegHOSLIMRecommender, NegHOSLIMElasticNetRecommender, NegHOSLIMLSQR
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.GraphBased.LightGCNRecommender import LightGCNRecommender
from Recommenders.GraphBased.INMORecommender import INMORecommender
from Recommenders.GraphBased.GraphFilterCFRecommender import GraphFilterCFRecommender, GraphFilterCF_W_Recommender
from Recommenders.GraphBased.ItemRankRecommender import ItemRankRecommender, ItemRankSVDRecommender, ItemRankInferenceRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_WARP_Cython, MatrixFactorization_SVDpp_Cython, MatrixFactorization_AsySVD_Cython
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask as MultVAERecommender_PyTorch


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender





# print("Importing Non-Personalized Recommenders...")
# from Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
# print("Non-Personalized Recommenders imported successfully.")

# print("Importing UserKNN CF...")
# from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
# print("UserKNN CF imported.")

# print("Importing ItemKNN CF...")
# from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
# print("ItemKNN CF imported.")

# print("Importing NegHOSLIM Recommenders...")
# from Recommenders.SLIM.NegHOSLIM import NegHOSLIMRecommender, NegHOSLIMElasticNetRecommender, NegHOSLIMLSQR
# print("NegHOSLIM Recommenders imported.")

# print("Importing SLIM BPR Cython...")
# from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
# print("SLIM BPR Cython imported.")

# print("Importing SLIM ElasticNet...")
# from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
# print("SLIM ElasticNet imported.")

# print("Importing P3alpha...")
# from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
# print("P3alpha imported.")

# print("Importing RP3beta...")
# from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
# print("RP3beta imported.")

# """
# print("Importing LightGCN...")
# from Recommenders.GraphBased.LightGCNRecommender import LightGCNRecommender
# print("LightGCN imported.")


# print("Importing INMO Recommender...")
# from Recommenders.GraphBased.INMORecommender import INMORecommender
# print("INMO Recommender imported.")
# """

# print("Importing GraphFilterCF...")
# from Recommenders.GraphBased.GraphFilterCFRecommender import GraphFilterCFRecommender, GraphFilterCF_W_Recommender
# print("GraphFilterCF imported.")

# print("Importing ItemRank Recommenders...")
# from Recommenders.GraphBased.ItemRankRecommender import ItemRankRecommender, ItemRankSVDRecommender, ItemRankInferenceRecommender
# print("ItemRank Recommenders imported.")

# print("Importing Matrix Factorization (Cython)...")
# from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_WARP_Cython, MatrixFactorization_SVDpp_Cython, MatrixFactorization_AsySVD_Cython
# print("Matrix Factorization (Cython) imported.")

# print("Importing PureSVD...")
# from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
# print("PureSVD imported.")

# print("Importing IALS Recommender...")
# from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
# print("IALS Recommender imported.")

# print("Importing NMF Recommender...")
# from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
# print("NMF Recommender imported.")

# print("Importing EASE_R...")
# from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
# print("EASE_R imported.")

# """
# print("Importing LightFM CF...")
# from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
# print("LightFM CF imported.")
# """

# print("Importing MultVAE...")
# from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender
# print("MultVAE imported.")

# """
# print("Importing MultVAE PyTorch...")
# from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask as MultVAERecommender_PyTorch
# print("MultVAE PyTorch imported.")
# """
# print("Importing ItemKNNCBF...")
# from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
# print("ItemKNNCBF imported.")

# print("Importing UserKNNCBF...")
# from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
# print("UserKNNCBF imported.")

# print("Importing ItemKNN CFCBF Hybrid...")
# from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
# print("ItemKNN CFCBF Hybrid imported.")

# print("Importing UserKNN CFCBF Hybrid...")
# from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
# print("UserKNN CFCBF Hybrid imported.")

# """
# print("Importing LightFM Hybrid...")
# from Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender
# print("LightFM Hybrid imported.")
# """