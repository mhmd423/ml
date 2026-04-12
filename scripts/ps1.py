import matplotlib.pyplot as plt

from src.metrics import accuracy_score
from src.models import LogisticRegression
from src.utils import load_data


def main():
    try:
        X_train, y_train = load_data(problem_number=1, subset_number=1, dataset_type="train")
        X_valid, y_valid = load_data(problem_number=1, subset_number=1, dataset_type="valid")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Dataset not found. Expected files under "
            "'data/cs229-2018-autumn/problem-sets/PS1/data/'."
        ) from exc

    model = LogisticRegression().fit(
        X_train,
        y_train,
        method="newton_method",
        standardize=True,
        num_iterations=1000,
    )

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    train_acc = accuracy_score(y_train, train_pred)
    valid_acc = accuracy_score(y_valid, valid_pred)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {valid_acc:.4f}")

    fig = model.visualize(X_train, y_train, contour_levels=20, more_info=True)
    plt.show()
    return fig


if __name__ == "__main__":
    main()
