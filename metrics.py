def accuracy_score(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    return (y_true == y_pred).mean()
