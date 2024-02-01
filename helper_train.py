import torch.nn as nn
import torchvision
import torch
import time
import random
import csv
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt



class Trainer():
    def __init__(self, model: object, hyperParams, DataLoders):
        
        self.model  = model.cuda()
        self.learning_rate = hyperParams.get('LEARNING_RATE', 1e-3)
        self.n_epoch =  hyperParams.get('NUM_EPOCHS', 50)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.DataLoders = DataLoders
        self.train_data = {}

        
        self.save_dir = Path(f'models')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_name = 'handwritten_digit_recognition_model.pth'


        
    def train(self):
    
        epoch_losses = []
        start_time = time.time() 
        self.model.train()
        for i_epoch in range(self.n_epoch):
            epoch_loss = 0.0
            n = 0
            for (features, targets) in self.DataLoders['train']:
                
                # Zero grad of the optimizer
                self.optimizer.zero_grad()
                # load on GPU
                features, targets = features.cuda(), targets.cuda()
                outputs = self.model(features)
                loss = self.loss_fn(outputs, targets)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                epoch_loss += float(loss)
                n += 1
                self.optimizer.step()
            epoch_loss = epoch_loss / n
            epoch_losses.append(epoch_loss)
            
            print(f'[{i_epoch+1}/{self.n_epoch}] Train-loss: {epoch_loss:.3f}')
            
            elapsed = (time.time() - start_time)/60
            if i_epoch % 4 == 0:
                print(f'Time elapsed: {elapsed:.2f} min')

        elapsed = (time.time() - start_time)/60
        print(f'Total Training Time: {elapsed:.2f} min')
        self.train_data['epoch_train_losses'] = epoch_losses
    
        fig=plt.figure(figsize=(8, 8))
        plt.plot(epoch_losses, 'y', label = 'Training Loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training loss / epoch')
        plt.legend()
        plt.show()
        figname = f'training_loss.png'
        fig.savefig(self.save_dir / figname, dpi=fig.dpi)
        


        
 


   
        
    def save_model(self):
        file_name = f'handwritten_digit_recognition_model.pth'
        torch.save(self.model.state_dict(), self.save_dir / file_name)
        
        csv_name =  f'handwritten_digit_recognition_model_results.csv'
        csv_path = self.save_dir / csv_name
        
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Type", "Key", "Value"])  # Write header row
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            for data_type, data_dict in [
                ("train_data", self.train_data),
            ]:
                for key, value in data_dict.items():
                    writer.writerow([data_type, key, value])
                    
    def load_model(self, version="v1"):
        file_name = f'handwritten_digit_recognition_model.pth'
        self.model.load_state_dict(torch.load(self.save_dir / file_name))




        
        

            


                
                

            
        
