Pipeline StageActive PyTorch LayerData Input ShapeData Output ShapeImage Inputx 
(Raw Tensor)[batch_size, 1, 28, 28][batch_size, 1, 28, 28]
Stage 1self.flatten[batch_size, 1, 28, 28][batch_size, 784]
Stage 2nn.Linear(784, 512)[batch_size, 784][batch_size, 512]
Stage 3nn.Linear(512, 512)[batch_size, 512][batch_size, 512]
Stage 4nn.Linear(512, 10)[batch_size, 512][batch_size, 10]
