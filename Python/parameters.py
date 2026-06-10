class Params:
    def __init__(self, data_path = "Datasets/Field Dataset"):
        self.data_path = data_path
        # self.raw_data = None
        self.test_data_external = None
        self.features = None
        self.labels = None
        self.feature_names = []
        self.sampling_rate = 50
