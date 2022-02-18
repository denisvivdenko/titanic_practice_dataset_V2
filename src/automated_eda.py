import pandas as  pd
import sweetviz as sv

df = pd.read_csv("src\\datasets\\train.csv")
advert_report = sv.analyze(df)
advert_report.show_html('src\\train_dataset_eda.html')
