import datetime
import pandas as pd

def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)


def data_from_df(df):
    data_by_class = {}
    oos_data_by_class = {}
    classes = df.columns
    for c in classes:
        examples = [example.lower() for example in df[c].to_list() if isinstance(example, str)]
        if c == "OOS":
            oos_data_by_class[c] = examples
        else:
            data_by_class[c] = examples

    return data_by_class, oos_data_by_class

if __name__ == "__main__":
    with open ("data/salesforce/sales.tsv", encoding="utf-8", mode="r") as f:
        df = pd.read_csv(f, sep="\t")
    data_by_class, oos_data_by_class = data_from_df(df)
