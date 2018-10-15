

class AcousticEmissionDataSet_3_10_2018:
    '''
        The sensor position are (-4.5, -2, 2, 5, 8, 10)m
        The leak position is always at 0m, at 10mm diameter
    '''
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.lcp_dataset_filename = self.drive + 