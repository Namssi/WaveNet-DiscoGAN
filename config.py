import argparse

rser = argparse.ArgumentParser(description='PyTorch WaveNet + DiscoGAN')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
args = parser.parse_args()
batch_size = args.batch_size

############arguments assighment: TODO use config file#########################
in_out_size = 256
res_size=512
layer_size = 10
stack_size = 2#args.stack_size
rfield_size = int(np.sum([2 ** i for i in range(0, layer_size)] * stack_size))

data_dir_g1 = '/home/jn1664/DGM/DiscoGAN/data_piano'
data_dir_g2 = '/home/jn1664/DGM/DiscoGAN/data_glock'
seed_dir_g1 = data_dir_g1 + '/A_Grand.wav'
seed_dir_g2 = data_dir_g2 + '/A_Glockenspiel.wav'
model_file_path = 'model/'
source_name = 'piano_grand'
target_name = 'glockenspiel'

sample_size = 10000
train_steps = 1000000
##################################################################################

