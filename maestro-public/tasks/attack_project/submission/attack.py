from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        epsilon = 0.2,
        min_val = 0,
        max_val = 1
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilon = 0.1
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        #size_original_images = original_images.shape
        original_images = original_images.to(self.device)
        original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images
        
        # -------------------- TODO ------------------ #

        # Write your attack function here

        # Write your attack function here

        # 1- loop range needs to be updated
        # 2- parameter c, k and learning rate needs to be updated
        
        
        # w =  torch.from_numpy(np.zeros(shape=size_original_images))
        # print("AAAAAAAAAAAAAAAAAAAAA",type(np.zeros(5)[0,0]))
        w =  torch.zeros_like(original_images)
        w = w.to(self.device)
        w.requires_grad = True
        optimizer = torch.optim.Adam([w],lr=0.1)
        k = torch.tensor([0.1])
        k = k.to(self.device)
        
        
    # all_c =  10**np.arange(-2,2.1,0.1)  
    # for c in all_c: 
    #     c = c   
        c = 0.059
        for i in range(10):
           #print(w[0,0,0,0].requires_grad)
           
           optimizer.zero_grad()
           data_grad = self.vm.get_batch_input_gradient(original_images, labels)
           perturbed_image = 0.5 * (torch.tanh(w) + 1) 

           output = self.vm.get_batch_output(perturbed_image)

           target_probs = output[:,target_label]

           output[:,target_label] = 0

          
           f = torch.max(torch.max(output,dim=1)[0] - target_probs,-k)
           
           loss = torch.norm(perturbed_image - original_images,p = 2) +  c * f
           old_w = w 
           loss.backward()

           
           optimizer.step()
        #    print("FOR",i)
        
        # print("END OF FOR")

        #------------evaluator---------
#         students_submission_path = "submission"
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("device: ", device)
#         dataset_configs = {
#             "name": "CIFAR10",
#             "binary": True,
#             "dataset_path": "datasets/CIFAR10/student",
#             "student_train_number": 10000,
#             "student_val_number": 1000,
#             "student_test_number": 100,
#                             }
#         dataset = get_dataset(dataset_configs) 
#         defense_list = [
#                 "defender",
#                     ]
#         target_label = 1
#         results = evaluate_attack(defense_list, students_submission_path, dataset,  device, target_label)


#         print("RRRRRRRRRRRRR",results)
    
        

        # ------------------ END TODO ---------------- #

        adv_outputs = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return np.squeeze(perturbed_image.cpu().detach().numpy()), correct
