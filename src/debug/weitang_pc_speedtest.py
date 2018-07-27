from src.experiment_dataset.dataset_experiment_2018_7_13 import AcousticEmissionDataSet_13_7_2018


user_drive_input = input('Enter Drive: ')

AEdata = AcousticEmissionDataSet_13_7_2018(drive=user_drive_input)
data1, data2 = AEdata.leak_noleak()


