from util import Encoder
from util import Decoder
from util import test
from tensorlization import *

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = ["train", "test"]
    select_mode = mode[0]
    train_data_path = "./data"
    test_data_path = "./data"

    if select_mode == "train":
        data_path = train_data_path
        train(data_path)

    if select_mode == "test":
        data_path = test_data_path
        test(data_path)



