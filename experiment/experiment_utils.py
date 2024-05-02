import openml
import warnings

import numpy as np
from naiveautoml.commons import get_scoring_name, build_scorer
import sklearn

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]

    X = df.drop(columns=[ds.default_target_attribute]).values
    y = df[ds.default_target_attribute].values

    return X, y

class LCMValidator:

    def __init__(self, instance, n_splits=5):
        self.instance = instance
        self.n_splits = n_splits

    def __call__(self, pl, X, y, scorings, errors="message"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            try:

                n=np.floor(np.sqrt(y.shape[0]))
                
                def run(job):

                    results = []
                    m=0
                    for i in range(4,n+1):
                        m=i
                        if i**(2)>len(X)*0.8:
                            Xp = X
                            yp = y
                        else:
                            Xp, _, yp, _ = sklearn.model_selection.train_test_split(X, y, train_size=i**2)
                        
                        if self.instance.task_type == "classification":
                            splitter = sklearn.model_selection.StratifiedKFold(
                                n_splits=self.n_splits,
                                random_state=None,
                                shuffle=True
                            )
                        elif self.instance.task_type:
                            splitter = sklearn.model_selection.KFold(n_splits=self.n_splits, random_state=None, shuffle=True)
                        scores = {get_scoring_name(scoring): [] for scoring in scorings}
                        for train_index, test_index in splitter.split(Xp, yp):

                            X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
                            y_train = y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index]
                            X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]
                            y_test = y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index]

                            pl_copy = sklearn.base.clone(pl)
                            pl_copy.fit(X_train, y_train)

                            for scoring in scorings:
                                scorer = build_scorer(scoring)
                                try:
                                    score = scorer(pl_copy, X_test, y_test)
                                except KeyboardInterrupt:
                                    raise
                                except Exception:
                                    score = np.nan
                                    if errors == "message":
                                        self.instance.logger.info(
                                            f"Observed exception in validation of pipeline {pl_copy}. Placing nan."
                                        )
                                    else:
                                        raise

                                scores[get_scoring_name(scoring)].append(score)
                        average = scores.mean()
                        results.append(average)# Simulation of iteration
                        job.record(budget=i**2, objective=average)

                        # Check if the evaluation should be stopped
                        # Calling stopper.stop() under the hood
                        if job.stopped():
                            break
                    
                    return {"objective": average, "metadata": {"i_stopped": m}}

                stopper = SuccessiveHalvingStopper(min_steps=4, max_steps=n)
                problem = HpProblem()
                problem.add_hyperparameter((0,1), "x")
                stopper = LCModelStopper(min_steps=1, max_steps=1000, lc_model="mmf4")
                search =  CBO(problem, run, stopper=stopper, log_dir="multi-fidelity-exp")
                results = search.search(max_evals=1)
                score = results["objective"][0]

                if not np.isnan(score) and score > self.r:
                    self.r = score

                results_at_highest_anchor = elcm.df[elcm.df["anchor"] == np.max(elcm.df["anchor"])].mean(
                    numeric_only=True)
                results = {
                    s: results_at_highest_anchor[f"score_test_{s}"] if not np.isnan(score) else np.nan for s in scorings
                }
                evaluation_histories = {s: elc if not np.isnan(score) else {} for s in scorings}

                # return the object itself, so that it can be overwritten in the pool (necessary because of pynisher)
                return results, evaluation_histories
            except KeyboardInterrupt:
                raise
            except Exception:
                if errors == "message":
                    self.instance.logger.info(f"Observed exception in validation of pipeline {pl}.")
                else:
                    raise
            return {s: np.nan for s in scorings}, {s: {} for s in scorings}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None, None
            else:
                raise

class LCMValidator:

    def __init__(self, instance, train_size=0.8):
        self.r = -np.inf
        self.instance = instance
        self.train_size = train_size

    def __call__(self, pl, X, y, scorings, errors="message"):
        warnings.filterwarnings('ignore', module='sklearn')
        warnings.filterwarnings('ignore', module='numpy')
        try:
            if not isinstance(scorings, list):
                scorings = [scorings]

            try:
                score, score_est, elc, elcm = lccv(
                    pl,
                    X,
                    y,
                    r=self.r,
                    base_scoring=scorings[0],
                    additional_scorings=scorings[1:],
                    target_anchor=self.train_size,
                    max_evaluations=5
                )
                if not np.isnan(score) and score > self.r:
                    self.r = score

                results_at_highest_anchor = elcm.df[elcm.df["anchor"] == np.max(elcm.df["anchor"])].mean(
                    numeric_only=True)
                results = {
                    s: results_at_highest_anchor[f"score_test_{s}"] if not np.isnan(score) else np.nan for s in scorings
                }
                evaluation_histories = {s: elc if not np.isnan(score) else {} for s in scorings}

                # return the object itself, so that it can be overwritten in the pool (necessary because of pynisher)
                return results, evaluation_histories
            except KeyboardInterrupt:
                raise
            except Exception:
                if errors == "message":
                    self.instance.logger.info(f"Observed exception in validation of pipeline {pl}.")
                else:
                    raise
            return {s: np.nan for s in scorings}, {s: {} for s in scorings}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if errors in ["message", "ignore"]:
                if errors == "message":
                    self.instance.logger.error(f"Observed an error: {e}")
                return None, None
            else:
                raise