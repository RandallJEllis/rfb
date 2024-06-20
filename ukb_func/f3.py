from sklearn.metrics import fbeta_score


def f3_metric(
        X_val,
        y_val,
        estimator,
        labels,
        X_train,
        y_train,
        weight_val=None,
        weight_train=None,
        config=None,
        groups_val=None,
        groups_train=None,
        ):

    import time

    start = time.time()

    y_pred = estimator.predict(X_val)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = 1 - fbeta_score(y_val, y_pred, beta=3)

    y_pred = estimator.predict(X_train)
    train_loss = 1 - fbeta_score(y_train, y_pred, beta=3)

    alpha = 0.25
    return val_loss * (1 + alpha) - alpha * train_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "pred_time": pred_time,
    }
