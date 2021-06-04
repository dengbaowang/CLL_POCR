# CLL_POCR

This is the implemention of our IJCAI'21 paper (Learning from Complementary Labels via Partial-Output Regularization Consistency).

Requirements: 
Python 3.6, 
numpy 1.19, 
Pytorch 1.5, 
torchvision 0.6.

You need to:
1. Download SVHN and CIFAR-10 datasets into './data/'.
2. Run the following demos:
	```
	python main.py --dataset svhn --model lenet --data-dir ./data/svhn/
        python main.py --dataset svhn --model preact --data-dir ./data/svhn/

        python main.py --dataset cifar10 --model preact --data-dir ./data/cifar/
        python main.py --dataset cifar10 --model widenet --data-dir ./data/cifar/
	```

If you have any further questions, please feel free to send an e-mail to: wangdb@seu.edu.cn. Have fun!
