import torch
import torch.nn.functional as F
import numpy as np
import os
from prepare import *
from network import *
from dataset import *
import sys
from pytorch_lightning import Trainer

msa_file='./example/T1001.a3m'
model_dir='models'
output_file = 'output'
trainset_dir = os.path.join('training_set','feature')
device = 'cuda'if torch.cuda.is_available()else 'cpu'

def predict(file_dir):
    feature=feature_deal(file_dir) # [1,526,n,n]
    net=trRosettaNetworkModule()
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(model_dir,'epoch=1-step=1971.ckpt')),False)
    output = net(feature)
    output = [i.cpu().detach().numpy() for i in output]
    output_dict = dict(zip([ 'phi','theta', 'omega', 'dist'], output))
    np.savez_compressed(os.path.join(output_file,'out.npz'), **output_dict)  # 保存npz文件
    print('saved')


def train(trainset_dir):
    data=trRosettaDataModule(batch_size=1,data_dir=trainset_dir)
    model=trRosettaNetworkModule()
    gpus=torch.cuda.device_count()
    print("using %s gpus"%(gpus))
    trainer=Trainer(gpus=gpus,min_epochs=2,max_epochs=2,precision=16)
    trainer.fit(model,data)
    print('done')
    # trainer.test(model,data)

if __name__ == '__main__':
    arg1=sys.argv[1]
    if arg1 == "predict":
        predict(msa_file)
    elif arg1 == "train":
        train(trainset_dir)
