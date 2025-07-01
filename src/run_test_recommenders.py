
import sys
import traceback
import os
import shutil
import numpy as np

print(">>> Script started", flush=True)
print(">>> __name__ =", __name__)
print(">>> sys.argv[0] =", sys.argv[0])
sys.stdout.flush()

print(">>> Starting imports...", flush=True)

try:
    from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
    print(">>> Imported BaseCBFRecommender", flush=True)
except Exception as e:
    print(">>> Import BaseCBFRecommender failed:", e, flush=True)

try:
    from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
    print(">>> Imported Evaluator", flush=True)
except Exception as e:
    print(">>> Import Evaluator failed:", e, flush=True)

try:
    from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
    print(">>> Imported Movielens1MReader", flush=True)
except Exception as e:
    print(">>> Import Movielens1MReader failed:", e, flush=True)

try:
    from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
    print(">>> Imported DataSplitter_leave_k_out", flush=True)
except Exception as e:
    print(">>> Import DataSplitter_leave_k_out failed:", e, flush=True)

try:
    from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
    print(">>> Imported Incremental_Training_Early_Stopping", flush=True)
except Exception as e:
    print(">>> Import Incremental_Training_Early_Stopping failed:", e, flush=True)

try:
    from Recommenders.Recommender_import_list import *
    print(">>> Imported Recommender_import_list", flush=True)
except Exception as e:
    print(">>> Import Recommender_import_list failed:", e, flush=True)

print("ending imports")



# Not importing multiprocessing
# import multiprocessing

def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()

def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object

def run_recommender(recommender_class, log_file):
    temp_save_file_folder = "./result_experiments/__temp_model/{}/".format(recommender_class.RECOMMENDER_NAME)
    os.makedirs(temp_save_file_folder, exist_ok=True)

    try:
        dataset_object = Movielens1MReader()

        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_out_value=2)
        dataSplitter.load_data()
        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        ICM_all = dataSplitter.get_loaded_ICM_dict()["ICM_genres"]
        UCM_all = dataSplitter.get_loaded_UCM_dict()["UCM_all"]

        write_log_string(log_file, "On Recommender {}\n".format(recommender_class))

        recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
        earlystopping_keywargs = {
            "epochs": 2000,
            "validation_every_n": 5,
            "stop_on_validation": True,
            "evaluator_object": evaluator,
            "lower_validations_allowed": 50,
            "validation_metric": "NDCG",
        }

        fit_params = earlystopping_keywargs if isinstance(recommender_object, Incremental_Training_Early_Stopping) else {}
        recommender_object.fit(**fit_params)

        write_log_string(log_file, "Fit OK, ")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
        results_df, results_run_string = evaluator.evaluateRecommender(recommender_object)
        write_log_string(log_file, "EvaluatorHoldout OK, ")

        evaluator = EvaluatorNegativeItemSample(URM_test, URM_train, [5], exclude_seen=True)
        _, _ = evaluator.evaluateRecommender(recommender_object)
        write_log_string(log_file, "EvaluatorNegativeItemSample OK, ")

        items_to_compute_not_sorted = np.random.randint(0, URM_train.shape[1], size=300)
        items_to_compute_sorted = np.sort(items_to_compute_not_sorted)
        for user_id in range(URM_train.shape[0]):
            recommender_object.recommend(user_id, cutoff=50, items_to_compute=items_to_compute_sorted, return_scores=True)
            recommender_object.recommend(user_id, cutoff=50, items_to_compute=items_to_compute_not_sorted, return_scores=True)

        write_log_string(log_file, "items_to_compute in the right order OK, ")

        recommender_object.save_model(temp_save_file_folder, file_name="temp_model")
        write_log_string(log_file, "save_model OK, ")

        recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
        recommender_object.load_model(temp_save_file_folder, file_name="temp_model")

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)
        result_df_load, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

        print(results_run_string)
        print(results_run_string_2)

        write_log_string(log_file, "load_model OK, ")

        from Recommenders.DataIO import DataIO
        dataIO = DataIO(temp_save_file_folder)
        _ = dataIO.load_data("temp_model.zip")

        shutil.rmtree(temp_save_file_folder, ignore_errors=True)

        write_log_string(log_file, " PASS\n")
        write_log_string(log_file, results_run_string + "\n\n")

    except Exception as e:
        print("On Recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()
        log_file.write("On Recommender {} Exception {}\n\n\n".format(recommender_class, str(e)))
        log_file.flush()


def main():
    import time
    print(">>> inside main()")
    print(">>> __name__ =", __name__)
    print(">>> sys.argv[0] =", sys.argv[0])
    print(">>> PID =", os.getpid())
    print(">>> Parent PID =", os.getppid())
    print(">>> ENV PATH =", os.environ.get("PATH"))
    print(">>> ENV LD_PRELOAD =", os.environ.get("LD_PRELOAD"))
    print(">>> Starting at", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    log_file_name = "./result_experiments/run_test_recommender.txt"
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)

    recommender_class = MatrixFactorization_BPR_Cython_Recommender

    with open(log_file_name, "w") as log_file:
        run_recommender(recommender_class, log_file)

    print(">>> Script finished", flush=True)


print(">>> entering __main__", flush=True)
main()