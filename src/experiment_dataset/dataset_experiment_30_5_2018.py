
path = 'F:/Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/2m/PLB'

# for Pencil Lead Break data, the sensor position are (-2, -1, 22, 23)m
class AcousticEmissionDataSet_30_5_2018:
    def __init__(self, drive):
        self.drive = drive + ':/'
        self.path_0m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/0m/PLB/'
        self.path_2m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/2m/PLB/'
        self.path_4m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/4m/PLB/'
        self.path_6m_plb = self.drive + 'Experiment_30_5_2018/test1_-2,-1,22,23m/PLB, Hammer/6m/PLB/'

    def PLB_4_sensor(self):
        