import pickle
import time
from pprint import pprint

best_trial = None
while True:
    try:
        with open(
            "/home/tsentker/Documents/projects/varreg-on-crack/hyperopt_results_all_cases_small_tau.pkl",
            "rb",
        ) as f:
            d = pickle.load(f)

            if best_trial != d.best_trial:
                print("*" * 80)
                print(d.best_trial["result"])
                pprint(d.best_trial["misc"]["vals"])

                best_trial = d.best_trial
    except:
        pass

    time.sleep(100)
