#!/usr/bin/env python
# coding: utf-8

import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from create_network import MTANDeepLabv3, MTLDeepLabv3
from utils import *
from misc import genWeights

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter


class trainerDense:

    def __init__(self,
                mode='none',
                port='none',
                network='split',
                weight='equal',
                grad_method='none',
                gpu=0,
                with_noise=False,  # Cambiado 'store_true' a False
                autol_init=0.1,
                autol_lr=1e-4,
                task='all',
                dataset='nyuv2',
                seed = 42, 
                total_epoch = 200):

        self.mode = mode
        self.port = port
        self.network = network
        self.weight = weight
        self.grad_method = grad_method
        self.gpu = gpu
        self.with_noise = with_noise  # Cambiado 'store_true' a False
        self.autol_init = autol_init
        self.autol_lr = autol_lr
        self.task = task
        self.dataset = dataset
        self.seed = int(seed)
        self.device = None
        self.train_tasks = None
   
        self.total_epoch = total_epoch 


    def initialize(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # create logging folder to store training weights and losses
        if not os.path.exists('logging'):
            os.makedirs('logging')


        # Crear un directorio para los registros de TensorBoard
        self.log_dir = 'tensorboard_logs'  # Puedes cambiar este nombre según tus preferencias
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


        # define model, optimiser and scheduler
        self.device = torch.device("cuda:{}".format(int(self.gpu)) if torch.cuda.is_available() else "cpu")
        if self.with_noise:
            self.train_tasks = create_task_flags('all', self.dataset, with_noise=True)
        else:
            self.train_tasks = create_task_flags('all', self.dataset, with_noise=False)

        self.pri_tasks = create_task_flags(self.task, self.dataset, with_noise=False)

        train_tasks_str = ''.join(task.title() + ' + ' for task in self.train_tasks.keys())[:-3]
        pri_tasks_str = ''.join(task.title() + ' + ' for task in self.pri_tasks.keys())[:-3]

        print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
            .format(self.dataset.title(), train_tasks_str, pri_tasks_str, self.network.upper()))
        print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
            .format(self.weight.title(), self.grad_method.upper()))

        if self.network == 'split':
            self.model = MTLDeepLabv3(self.train_tasks).to(self.device)
        elif self.network == 'mtan':
            self.model = MTANDeepLabv3(self.train_tasks).to(self.device)


    def choose_task_weighting(self, weight):

        if weight == 'uncert':
            logsigma = torch.tensor([-0.7] * len(self.train_tasks), requires_grad=True, device=self.device)
            self.params = list(self.model.parameters()) + [logsigma]
            self.logsigma_ls = np.zeros([self.total_epoch, len(self.train_tasks)], dtype=np.float32)

        if weight in ['dwa', 'equal']:
            self.T = 2.0  # temperatura utilizada en dwa
            self.lambda_weight = np.ones([self.total_epoch, len(self.train_tasks)])
            self.params = self.model.parameters()

        if weight == 'autol':
            self.params = self.model.parameters()
            self.autol = AutoLambda(self.model, self.device, self.train_tasks, self.pri_tasks, self.autol_init)
            self.meta_weight_ls = np.zeros([self.total_epoch, len(self.train_tasks)], dtype=np.float32)
            self.meta_optimizer = optim.Adam([self.autol.meta_weights], lr=self.autol_lr)


        # para probar combinaciones de pesos 
        if weight == 'combinations':
            #numero de pesos
            nw = 21
            #genera pesos
            self.my_weight = genWeights(nw, len(self.train_tasks), self.device)
            self.lambda_weight = np.ones([self.total_epoch, len(self.train_tasks)])
            print('primer lambda ', self.lambda_weight.shape)
            self.params = self.model.parameters()      
            
        self.optimizer = optim.SGD(self.params, lr=0.1, weight_decay=1e-4, momentum=0.9)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_epoch)  

        # una copia de train_loader con diferente orden de datos, utilizada para la actualización meta de Auto-Lambda
        if weight == 'autol':
            self.val_loader = torch.utils.data.DataLoader(
                dataset= self.train_set,
                batch_size = self.batch_size,
                shuffle=True,
                num_workers=4
            )



    def define_dataset(self, dataset):

        if dataset == 'nyuv2':
            dataset_path = 'dataset/nyuv2'
            train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
            test_set = NYUv2(root=dataset_path, train=False)
            self.batch_size = 4
        elif dataset == 'cityscapes':
            dataset_path = 'dataset/cityscapes'
            train_set = CityScapes(root=dataset_path, train=True, augmentation=True)
            test_set = CityScapes(root=dataset_path, train=False)
            self.batch_size = 4

    #def define_data_loader(self):

        self.train_loader = torch.utils.data.DataLoader(
            dataset = train_set,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset = test_set,
            batch_size = self.batch_size,
            shuffle = False
        )    
    
    def apply_gradient_methods(self, grad_method):
    # apply gradient methods
        if grad_method != 'none':
            self.rng = np.random.default_rng()
            self.grad_dims = []
            for mm in self.model.shared_modules():
                for param in mm.parameters():
                    self.grad_dims.append(param.data.numel())
            self.grads = torch.Tensor(sum(self.grad_dims), len(self.train_tasks)).to(self.device)


    def update_weights(self, index):
        if self.weight == 'dwa':
            self.update_weights_dwa(index)
        elif self.weight == 'uncert':
            self.update_weights_uncert(index)
        elif self.weight == 'autol':
            self.update_weights_autol(index)
        elif self.weight == 'combinations':
            self.update_weights_combinations(index)

    def update_weights_dwa(self, index):
        if index == 0 or index == 1:
            self.lambda_weight[index, :] = 1.0
        else:
            w = []
            for i, t in enumerate(self.train_tasks):
                w += [self.train_metric.metric[t][index - 1, 0] / self.train_metric.metric[t][index - 2, 0]]
            w = torch.softmax(torch.tensor(w) / self.T, dim=0)
            self.lambda_weight[index] = len(self.train_tasks) * w.numpy()

    def update_weights_uncert(self, index):
        # Lógica para actualizar pesos basados en incertidumbre
        pass

    def update_weights_autol(self, index):
        # Lógica para actualizar pesos con Auto-Lambda
        pass

    def update_weights_combinations(self, index):
        # Lógica para actualizar pesos en caso de combinaciones de pesos
        pass

    def registra_info_tboard(writer, epoca, hist):
        for (m,v) in hist.items():
            writer.add_scalar(m, v[epoca], epoca)

       

    def train_one_epoch(self, index):
        print('Epoch: ', index)

        self.model.train()
        train_dataset = iter(self.train_loader)

        if self.weight == 'autol':
            val_dataset = iter(self.val_loader)

        for k in range(self.train_batch):
            train_data, train_target = next(train_dataset)
            train_data = train_data.to(self.device)
            train_target = {task_id: train_target[task_id].to(self.device) for task_id in self.train_tasks.keys()}

            # Actualizar meta-pesos con Auto-Lambda
            if self.weight == 'autol':
                val_data, val_target = next(val_dataset)
                val_data = val_data.to(self.device)
                val_target = {task_id: val_target[task_id].to(self.device) for task_id in self.train_tasks.keys()}
                
                self.meta_optimizer.zero_grad()
                self.autol.unrolled_backward(train_data, train_target, val_data, val_target, self.scheduler.get_last_lr()[0], self.optimizer)
                self.meta_optimizer.step()
            
            # Actualizar parámetros de la red multi-tarea con pesos de tareas
            self.optimizer.zero_grad()
            train_pred = self.model(train_data)
            train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(self.train_tasks)]

            train_loss_tmp = [0] * len(self.train_tasks)

            if self.weight in ['equal', 'dwa', 'combinations']:
                train_loss_tmp = [w * train_loss[i] for i, w in enumerate(self.lambda_weight[index])]

            if self.weight == 'uncert':
                train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(self.logsigma)]

            if self.weight == 'autol':
                train_loss_tmp = [w * train_loss[i] for i, w in enumerate(self.autol.meta_weights)]

            loss = sum(train_loss_tmp)

            if self.grad_method == 'none':
                loss.backward()
                self.optimizer.step()

            # Métodos basados en gradientes aplicados aquí
            elif self.grad_method == "graddrop":
                for i in range(len(self.train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    self.grad2vec(self.model, self.grads, self.grad_dims, i)
                    self.model.zero_grad_shared_modules()
                g = self.graddrop(self.grads)
                self.overwrite_grad(self.model, g, self.grad_dims, len(self.train_tasks))
                self.optimizer.step()

            elif self.grad_method == "pcgrad":
                for i in range(len(self.train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    self.grad2vec(self.model, self.grads, self.grad_dims, i)
                    self.model.zero_grad_shared_modules()
                g = self.pcgrad(self.grads, self.rng, len(self.train_tasks))
                self.overwrite_grad(self.model, g, self.grad_dims, len(self.train_tasks))
                self.optimizer.step()

            elif self.grad_method == "cagrad":
                for i in range(len(self.train_tasks)):
                    train_loss_tmp[i].backward(retain_graph=True)
                    self.grad2vec(self.model, self.grads, self.grad_dims, i)
                    self.model.zero_grad_shared_modules()
                g = self.cagrad(self.grads, len(self.train_tasks), 0.4, rescale=1)
                self.overwrite_grad(self.model, g, self.grad_dims, len(self.train_tasks))
                self.optimizer.step()

            self.train_metric.update_metric(train_pred, train_target, train_loss)

        #train_str = self.train_metric.compute_metric()
        #self.train_metric.reset()


    def evaluate_metrics(self, test_dataset):
        self.model.eval()
        with torch.no_grad():
            test_dataset = iter(test_dataset)
            for k in range(self.test_batch):
                test_data, test_target = next(test_dataset)
                test_data = test_data.to(self.device)
                test_target = {task_id: test_target[task_id].to(self.device) for task_id in self.train_tasks.keys()}

                test_pred = self.model(test_data)
                test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(self.train_tasks)]
                self.test_metric.update_metric(test_pred, test_target, test_loss)

            #return test_pred, test_target, test_loss

    def print_and_save_results(self, index, writer, peso_actual):
        train_str = self.train_metric.compute_metric()
        self.train_metric.reset()
        test_str = self.test_metric.compute_metric()
        self.test_metric.reset()

        # Agregar métricas a TensorBoard
        #writer.add_scalar(f'Train/{self.task}_Loss', self.train_metric.get_best_performance(self.task), index)
        #writer.add_scalar(f'Test/{self.task}_Loss', self.test_metric.get_best_performance(self.task), index)


        print('Época {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
            .format(index, train_str, test_str, self.task.title(), self.test_metric.get_best_performance(self.task)))

        if self.weight == 'autol':
            self.meta_weight_ls[index] = self.autol.meta_weights.detach().cpu()
            results_dict = {'train_loss': self.train_metric.metric, 'test_loss': self.test_metric.metric, 'weight': self.meta_weight_ls}
            print(get_weight_str(self.meta_weight_ls[index], self.train_tasks))

        if self.weight in ['dwa', 'equal', 'combinations']:
            results_dict = {'train_loss': self.train_metric.metric, 
                            'test_loss': self.test_metric.metric, 
                            'weight': self.lambda_weight}
            print(get_weight_str(self.lambda_weight[index], self.train_tasks))
            
            # Agregar métricas específicas al SummaryWriter de TensorBoard

          #  for metric_name, metric_values in results_dict.items():
                # Aquí, metric_name sería el nombre de la métrica (por ejemplo, 'train_loss' o 'test_loss')
                # metric_values sería un array con los valores de la métrica en cada época
           # 
           #     for epoch, value in enumerate(metric_values):
                    # Agregar la métrica al SummaryWriter
           #         writer.add_scalar(f'Weighting/{metric_name}', value, epoch)



        if self.weight == 'uncert':
            self.logsigma_ls[index] = self.logsigma.detach().cpu()
            results_dict = {'train_loss': self.train_metric.metric, 'test_loss': self.test_metric.metric, 'weight': self.logsigma_ls}
            print(get_weight_str(1 / (2 * np.exp(self.logsigma_ls[index])), self.train_tasks))

        np.save('logging2/mtl_dense_{}_{}_{}_{}_{}_{}_{}_{}.npy'
                .format(self.network, self.dataset, self.task, self.weight, self.grad_method, 
                        self.seed, peso_actual, self.network), results_dict)  

          

    def plot_performance(self, save = True):
        # Obtén la lista de claves de las métricas
        train_metric_keys = list(self.train_metric.metric.keys())

        # Calcula el número de métricas
        num_metrics = len(train_metric_keys)
        labels = ['Loss', 'task-specific metric', 'All']

        fig, axes = plt.subplots(2, num_metrics, figsize=(16, 8))  # Ajusta el tamaño según tus necesidades

        for j in range(2):

            # Recorre las métricas y crea los gráficos
            for i, key in enumerate(train_metric_keys[:]):
                # Gráfico de métrica de entrenamiento
                if key != 'all':
                    axes[j, i].plot(t.train_metric.metric[key][:, j], label=f'Train {key}')
                    axes[j, i].plot(t.test_metric.metric[key][:, j], label=f'Test {key}')   
                    axes[j, i].legend()  # Agrega una leyenda para cada gráfico         
                else:
                    axes[j, i].plot(t.train_metric.metric['all'])
                    axes[j, i].plot(t.test_metric.metric['all']) 

                axes[j, i].set_xlabel('Epoch')
                axes[j, i].set_ylabel(labels[j])
                axes[j, i].set_title(key)
             #   axes[j, i].legend()  # Agrega una leyenda para cada gráfico
        

            # Ajusta el espaciado entre los subgráficos
            plt.tight_layout()
        if save:    
            plt.savefig('figures2/' + get_weight_str(self.lambda_weight[0], self.train_tasks) + self.network + '_' + self.dataset + '.pdf' )   
            plt.savefig('figures2/' + get_weight_str(self.lambda_weight[0], self.train_tasks) + self.network + '_' + self.dataset + '.png' )     
        # Muestra los subgráficos
        plt.show()
        #plt.close()                



    def train(self):
        if self.weight == 'combinations':
            w = self.my_weight
            
        else:
            w = np.ones([1, 1])
            tensorboard_log_dir = self.log_dir
        
        
        #for j in range(5, 6):
        for j in range(3, 5):
            self.apply_gradient_methods(self.grad_method)
            self.train_batch = len(self.train_loader)
            self.test_batch = len(self.test_loader)
            self.train_metric = TaskMetric(self.train_tasks, self.pri_tasks, self.batch_size, self.total_epoch, self.dataset, include_mtl=True)
            self.test_metric = TaskMetric(self.train_tasks, self.pri_tasks, self.batch_size, self.total_epoch, self.dataset, include_mtl=True)
            
            tensorboard_log_dir = os.path.join(self.log_dir, get_weight_str(self.lambda_weight[0], self.train_tasks))
            writer = SummaryWriter(tensorboard_log_dir)  

            print('\nProbando combinación de pesos: ',  w[:,j].tolist())
            if self.weight == 'combinations':
                self.lambda_weight = np.tile(w[:, j].numpy(), (self.total_epoch, 1))

            for index in range(self.total_epoch):
                self.update_weights(index)
                self.train_one_epoch(index)          

                #train_str = self.train_metric.compute_metric()
                #self.train_metric.reset() 

                
                self.evaluate_metrics(self.test_loader)

                #test_str = self.test_metric.compute_metric()
                #self.test_metric.reset()

                self.scheduler.step()

                peso_actual = w[:,j].tolist()
                self.print_and_save_results(index, writer, peso_actual)
                torch.cuda.empty_cache()  
            guarda_ckpt(ckptpath = '/home/carmen/auto-lambda/checkpoint2/ckpt_' +  get_weight_str(self.lambda_weight[0], self.train_tasks) + self.network +'_' + self.dataset + '.pt', 
                        modelo = self.model, 
                        epoca = self.total_epoch 
                        #opt = 
                        )       
            # Cerrar el SummaryWriter de TensorBoard al final del entrenamiento
            writer.close() 
            self.plot_performance()
            torch.cuda.empty_cache()   
            



t = trainerDense(network='mtan', total_epoch = 200, weight = 'combinations', dataset = 'cityscapes') #autol_lr=3e-5)
t.initialize()
t.choose_task_weighting(weight = 'combinations')
t.define_dataset(dataset = 'cityscapes')
t.train()