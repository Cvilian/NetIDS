import os
from utils import *

class NetData:

    def __init__(self):
        self.x, self.y = self.getData()

    def getrawData(self):
        x_raw = pd.read_csv("../dataset/flow_data.csv", dtype='unicode')

        if not os.path.isfile("../dataset/flow_label.csv") :
            y_raw = pd.DataFrame()
            y_raw['label'] = raw['host_name'].apply(lambda x: labeling(x))
            y_raw.to_csv("../dataset/flow_label.csv", index=False)
            return x_raw, y_raw
        
        y_raw = pd.read_csv("../dataset/flow_label.csv", dtype='unicode')
        return x_raw, y_raw


    def getData(self):
        x_raw, y_raw =  self.getrawData()
        x = x_raw.replace({'-':0})

        x = x.drop(columns=['src', 'dst', 'host_name', 'stream_no.', 'proto'])
        
        for col in ['ip_dscp_fwd', 'ip_dscp_rev'] :
            x[col] = x[col].astype('category').cat.codes

        y = y_raw['label'].map({'N':0, 'Y':1}.get)

        x = x.to_numpy()
        y = y.to_numpy()
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.int64)
        return x, y

    def normalize(self, x):
        x_tan = np.tanh(x)
        #nom_x = (x_log - x_log.min(0)) / (x_log.ptp(0) + 1e-9)
        nom_x = (x_tan - x_tan.mean(0)) / (x_tan.std(0) + 1e-5)
        return nom_x
            
    def getInput(self):
        return self.normalize(self.x), self.y
    



