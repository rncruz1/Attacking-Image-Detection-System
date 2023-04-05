"""
Author: c0ldstudy
2022-03-29 13:27:24
"""
import importlib.util
import torch
import time
import numpy as np

class VirtualModel:
    def __init__(self, device, defender) -> None:
        self.device = device
        self.defender = defender
        # self.model = defender.model
        self.predict_queries = 0
        self.gradient_queries = 0
        # pass

    def _set(self, status):
        if status == "train":
            self.defender.model.train()
            return self
        elif status == "eval":
            self.defender.model.eval()
            return self
        else:
            raise Exception("model status is set wrongly.")

    def _to(self, device):
        self.defender.model.to(device)
        return self

    def get_batch_output(self, batch):
        self.predict_queries += batch.shape[0]
        output = self.defender.get_batch_output(batch)
        return output

    def get_batch_input_gradient(self, batch, labels):
        self.gradient_queries += batch.shape[0]
        return self.defender.get_batch_input_gradient(batch, labels)

    def reset_stats(self):
        self.predict_queries = 0
        self.gradient_queries = 0


class EvaluatePair:
    def __init__(self, attack_path, defense_path, dataset, device=None) -> None:
        self.device = device
        self.attack_path = attack_path
        self.defense_path = defense_path
        self.dataset = dataset
        self.defender = self.load_defense_predict(defense_path, self.device)
        self.attacker = self.load_attack(attack_path)

    def load_defense_predict(self, defense_path, device):
        spec = importlib.util.spec_from_file_location('predict', defense_path + "/predict.py")
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)
        defender_predict = predict_module.Prediction(device, defense_path)
        defender = VirtualModel(self.device, defender_predict)
        return defender

    def load_attack(self, attack_path):
        spec = importlib.util.spec_from_file_location('attack', attack_path + '/attack.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # for attack methods evaluator, the Attack class name should be fixed
        attacker = foo.Attack(self.defender, self.device, attack_path)
        return attacker

    def adv_generator(self, testset, attack_method, target_label=None):
        distance = []
        n_success_attack = 0
        perturbed_images = []
        for img, labels in testset:
            if target_label != None and target_label == labels:
                print("label ", labels)
                print("target_label ", target_label)
                continue
            perturbed_data, success = attack_method(img, [labels], target_label)
            perturbed_images.append((labels, perturbed_data))
            delta_data = img - perturbed_data[0]
            distance.append(np.linalg.norm(delta_data))
            n_success_attack += success
        return distance, perturbed_images, n_success_attack

    def evaluate(self, target_label = None):
        # trainset = self.dataset['train']
        # valset = self.dataset['val']
        testset = self.dataset['test'] # TorchVisionDataset
        # testset = self.dataset['server_test'] # TorchVisionDataset

        adv_images = []
        gt_labels = []
        start_time = time.perf_counter()
        (distance, perturbed_images, n_success_attack) = self.adv_generator(testset, self.attacker.attack, target_label)
        run_time = time.perf_counter() - start_time
        # n_success_attack_list.append(n_success_attack)
        for image in perturbed_images:
            adv_images.append(image[1])
            gt_labels.append(image[0])
        adv_images = (torch.tensor(np.array(adv_images)).type(torch.FloatTensor))
        gt_labels = torch.tensor(gt_labels)

        adv_dataset = torch.utils.data.TensorDataset(adv_images, gt_labels)
        adv_testloader = torch.utils.data.DataLoader(adv_dataset, batch_size=100, shuffle=True, num_workers=10)  # raw data
        targeted_correct = 0
        untargeted_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in adv_testloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.defender.get_batch_output(inputs)
                _, predicted = torch.max(outputs, 1)
                if target_label != None:
                    targeted_correct += (predicted != target_label).sum().item()
                    targeted_correct += (predicted == torch.full(predicted.shape, -1, dtype=torch.int).to(self.device)).sum().item()

                total += labels.size(0)
                untargeted_correct += (predicted == labels).sum().item()
                untargeted_correct += (predicted == torch.full(predicted.shape, -1, dtype=torch.int).to(self.device)).sum().item()
        targeted_adv_acc = 100 * targeted_correct / total
        untargeted_adv_acc = 100 * untargeted_correct / total
        # print("query:", self.defender.predict_queries, self.defender.gradient_queries)
        return {"targeted_adv_acc": targeted_adv_acc, "untargeted_adv_acc": untargeted_adv_acc, "run_time": run_time, "distance": np.mean(distance), "predict_queries": self.defender.predict_queries/total, "gradient_queries": self.defender.gradient_queries/total}
    
    def raw_evaluate(self):
        # 3 evaluate on original data
        testset = self.dataset['test'] # TorchVisionDataset
        # raw data result
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=True, num_workers=10
        )  # raw data
        correct = 0
        total = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                output = self.defender.get_batch_output(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        run_time = time.perf_counter() - start_time

        print("Accuracy of the network on the images: %.3f %%" % (100 * correct / total))
        raw_acc = 100 * correct / total
        return {"raw_acc": raw_acc}
