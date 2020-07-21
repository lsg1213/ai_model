import torch, pdb

class self_training():
    def __init__(self, teachermodel):
        self.prob = []
        self.model = teachermodel
        self.batch_size = 256
        self.score = []

    def get_max(self, data):
        return torch.max(data,dim=-1)

    def get_samples(self, unlabeled_data, number, device=torch.device('cpu')):
        self.model.eval()
        self.model.to(device)
        unlabeled_data_loader = torch.utils.data.DataLoader(unlabeled_data, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for i, items in enumerate(unlabeled_data_loader):
                inputs = items[0].to(device)
                output = self.model(inputs)
                score = self.get_max(output)[0]
                for index, res in enumerate(output):
                    if len(res.shape) == 1:
                        res = torch.unsqueeze(res, 0)
                    self.prob.append([i * self.batch_size + index, res, score[index]])
        
        self.prob = sorted(self.prob, key=lambda x: x[2],reverse=True)
        outputs = self.prob[:number]
        sample_index = [i[0] for i in outputs]
        samples = torch.utils.data.Subset(unlabeled_data, indices=sample_index)
        new_unlabeled_index = []
        for i in range(len(unlabeled_data)):
            if not (i in sample_index):
                new_unlabeled_index.append(i)
        new_unlabeled_dataset = torch.utils.data.Subset(unlabeled_data, indices=new_unlabeled_index)
        
        return new_unlabeled_dataset, samples