"""
This Python file is an example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.
Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script
produces should be the same.
The test CSV file has the same columns as the training file
"csc311_project_training_202609.csv", except that it may not have
the `label` column.
Here's an example of how your script may be used in our test file:
    from pred import predict_all
    predict_all("example_test_set.csv")
"""
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "results" / "final_model.joblib"
sys.path.insert(0, str(ROOT))


def predict_all(filename):
    """Make predictions for the data in filename"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {MODEL_PATH}. Run "
            "python models/train_final_model.py first."
        )
    artifact = joblib.load(MODEL_PATH)
    df = pd.read_csv(filename, dtype=str, keep_default_na=False)
    X = artifact["preprocessor"].transform(df)
    return artifact["model"].predict(X).tolist()


if __name__ == "__main__":
    print(predict_all(sys.argv[1])[:10])
