import os
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from fire import Fire
from pipeline.Reader import Reader

class History(Reader):

    def __init__(self,
                 datapath="../data/brunch/read"):
        super(History, self).__init__()

        self.datapath = datapath

    def read(self):
        files = sorted(os.listdir(self.datapath))

        ids = []
        history = []
        timestamp = []
        session_idx = []

        for idx, file in tqdm(enumerate(files), total=len(files)):
            path = os.path.join(self.datapath, file)
            user_ids, session = self.readfile(filepath=path)

            assert len(user_ids) == len(session), "The number of user ids and its session doesn't match"

            tokens = file.split("_")
            time = datetime.strptime(tokens[0], '%Y%m%d%H')
            times = [time] * len(user_ids)
            indices = [idx] * len(user_ids)

            ids.extend(user_ids)
            history.extend(session)
            timestamp.extend(times)
            session_idx.extend(indices)

        df = pd.DataFrame({"id" : ids, "history" : history, "timestamp" : timestamp, "session" : session_idx})
        df.to_parquet("../data/brunch/session.parquet")

    def readfile(self,
             filepath):
        ids = []
        history = []
        with open(filepath) as fp:
            lines = [line for line in fp]

            for line in lines:
                tokens = line.split(" ")
                tokens.pop()
                if tokens[0][0] != '#':
                    continue
                user_id = tokens[0]
                session = []

                tokens = tokens[1:]
                for elem in tokens:
                    if elem[0] != '@':
                        continue
                    session.append(elem)

                ids.append(user_id)
                history.append(session)

        return ids, history

    def write(self):
        pass


if __name__ == "__main__":
    History().read()