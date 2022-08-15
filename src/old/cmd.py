# from src.runner import Runner
# import pandas as pd
#
# if __name__ == "__main__":
#     max_sequence = 128
#     classes = 4
#     runner = Runner(max_sequence, classes)
#
#     df_full = pd.read_csv("../dataset/ecommerceDataset.csv", header=None)
#     df_full = df_full.dropna()
#
#     sequences = df_full[1]
#     labels = df_full[0]
#
#     runner.train(sequences, labels, 2)