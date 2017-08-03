lvl1_params = [
{'n_estimators': 300,
                'max_features': 10,
                'min_samples_leaf': 10}
]

rf_grid = { "rf__n_estimators"       : [25, 50]}
            #    "criterion"          : ["gini", "entropy"],
            #    "rf__max_depth"          : [1, 5, 10],
            #    "rf__min_samples_split"  : [2, 4] ,
            #    "rf__max_features"       : ["sqrt", "log2", None],
            # #    "oob_score"          : [True, False],
            #    "rf__min_samples_leaf"   : [1, 5, 10]}
