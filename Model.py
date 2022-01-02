import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01, weight_decay=self.config.weight_decay)
        ### YOUR CODE HERE

    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            samples_taken = 0
            ### YOUR CODE HERE

            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                prev_batch = i
                new_train_x = np.zeros((self.config.batch_size,3,32,32))
                if(samples_taken+self.config.batch_size - samples_taken >= self.config.batch_size):
                    batch_train_x = curr_x_train[samples_taken:samples_taken+self.config.batch_size,:]
                    batch_train_y = curr_y_train[samples_taken:samples_taken+self.config.batch_size]
                else:
                    batch_train_x = curr_x_train[curr_x_train.shape[0]-samples_taken:,:]
                    batch_train_y = curr_y_train[curr_y_train.shape[0]-samples_taken:]
                for j in range(batch_train_x.shape[0]):
                    new_train_x[j,:,:,:] = parse_record(batch_train_x[j], True)

                samples_taken = self.config.batch_size*(i+1)

                #one_hot_y = np.zeros((batch_train_y.size, batch_train_y.max()+1))
                #one_hot_y[np.arange(batch_train_y.size),batch_train_y] = 1

                new_train_x, batch_train_y = torch.Tensor(new_train_x).cuda(), torch.tensor(batch_train_y).cuda()
                logps = self.network(new_train_x)
                batch_train_y = batch_train_y.to(dtype=torch.int64)
                loss = self.criterion(logps, batch_train_y)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                new_image = parse_record(x[i], False)
                new_image = np.expand_dims(new_image, axis=0) # expands dimension from Tensor of dimensions 3 to Tensor of dimaneions 4
                processed_image = torch.tensor(new_image).cuda()
                processed_image = processed_image.type(torch.cuda.FloatTensor)
                logps = self.network(processed_image)
                ps = torch.exp(logps)
                top_p, top_class = torch.topk(ps, 1)
                preds.append(int(top_class[0]))
                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))

    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
