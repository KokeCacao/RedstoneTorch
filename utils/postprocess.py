import pandas as pd

def sort_id(dir_sample_submission_csv, dir_target_csv, predicted_col, id_col_left, id_col_right):
    """

    :param dir_sample_submission_csv: .../sample_submission.csv
    :param dir_target_csv: .../submission.csv
    :param predicted_col: Predicted
    :param id_col_left: Id
    :param id_col_right: Id

    """
    f1 = pd.read_csv(dir_sample_submission_csv)
    f1.drop(predicted_col, axis=1, inplace=True)
    f2 = pd.read_csv(dir_target_csv)
    f1 = f1.merge(f2, left_on=id_col_left, right_on=id_col_right, how='outer')
    f1.to_csv(dir_target_csv, index=False)