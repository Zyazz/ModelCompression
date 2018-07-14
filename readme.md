## Model Compression Overview
## 1. 动机
深度学习在图像、语音、文本等领域都取得了巨大的成功，推动了一系列智能产品的落地。但深度模型存在着参数众多，训练和 inference 计算量大的不足。目前，基于深度学习的产品大多依靠服务器端运算能力的驱动，非常依赖良好的网络环境。

很多时候，出于响应时间、服务稳定性和隐私方面的考虑，我们更希望将模型部署在本地（如智能手机上）。为此，我们需要解决模型压缩的问题——将模型大小、内存占用、功耗等降低到本地设备能够承受的范围之内。

## 2. 方法
神经网络具有分布式的特点——特征表征和计算都分散于各个层、各个参数。因此，神经网络在结构上天然具有冗余的特点。冗余是神经网络进行压缩的前提。

压缩模型一般可以有几种常见的方法：

#### 2.1 使用小模型
设计小模型
可以直接将模型大小做为约束，在模型结构设计和选择时便加以考虑。对于全连接，使用 bottleneck 是一个有效的手段（如 LSTMP）。Highway，ResNet，DenseNet 等带有 skip connection 结构的模型也被用来设计窄而深的网络，从而减少模型整体参数量和计算量。对 CNN 网络，SqueezeNet 通过引入1 x 1的小卷积核、减少 feature map 数量等方法，在分类精度与 AlexNet 相当的前提下，将模型大小压缩在 1M 以内，而模型大小仅是 Alexnet 的50分之一。

模型小型化
一般而言，相比于小模型，大模型更容易通过训练得到更优的性能。那么，能否用一个较小的模型，“提炼”出训练好的大模型的知识能力，从而使得小模型在特定任务上，达到或接近大模型的精度？Knowledge Distilling(e.g. 1、2)便尝试解决这一问题。knowledge distilling 将大模型的输出做为 soft target 来训练小模型，达到知识“凝练“的效果。实验表明，distilling 方法在 MNIST 及声学建模等任务上有着很好的表现。

#### 2.2 利用稀疏性
我们也可以通过在模型结构上引入稀疏性，从而达到减少模型参数量的效果。

裁剪已有模型
将训练好的模型进行裁剪的方法，至少可以追溯到90年代。 Optimal Brain Damage 和 Optimal Brain Surgeon 通过一阶或二阶的梯度信息，删除不对性能影响不显著的连接，从而压缩模型规模。

学习稀疏结构
稀疏性也可以通过训练获得。更近的一系列工作（[Deep compression](papers/1510.00149.pdf): a、b 、c 及 HashedNets）在控制模型性能的前提下，学习稀疏的模型结构，从而极大的压缩模型规模。

#### 2.3 降低运算精度
不同传统的高性能计算，神经网络对计算精度的要求不高。目前，基本上所有神经网络都采用单精度浮点数进行训练（这在很大程度上决定着 GPU 的架构设计）。已经发布的 NVIDIA Pascal 架构的最大特色便是原生的支持半精度（half float）运算。在服务端，FPGA 等特殊硬件在许多数据中心得到广泛应用，多采用低精度（8 bit）的定点运算。

#### 2.4 参数量化
除了使用低精度浮点运算（float32, float16）外，量化参数是另一种利用简化模型的有效方法。 
将参数量化有如下二个优势： 
* 减少模型大——将 32 或 16 位浮点数量化为 8 位甚至更少位的定点数，能够极大减少模型占用的空间； 
* 加速运算——相比于复杂的浮点运算，量化后的定点运算更容易利用特殊硬件（FPGA，ASIC）进行加速。

上面提到的 Deep Compression 使用不同的位数量化网络。Lin 等的工作，在理论上讨论上，在不损失性能的前提下，CNN 的最优量化策略。此外，还有量化 CNN 和 RNN 权值的相关工作。

参数二值化
量化的极限是二值化，即每一个参数只占用一个 bit。

https://www.cnblogs.com/zhonghuasong/p/7822572.html

### quantization
- [squeezenet](papers/squeezenet.pdf)
- [BinaryConnect: Training Deep Neural Networks with
binary weights during propagations](papers/1511.00363.pdf), Matthieu Courbariaux
- [Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1](papers/binarynet.pdf), Matthieu Courbariaux, 2016
 [pytorch code](https://github.com/DingKe/pytorch_workplace/tree/master/binary)
- [Straight Through Estimator (STE)](papers/ste.pdf),Yoshua Bengio
- [Quantized Neural Networks](papers/1609.07061.pdf),Itay Hubara
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](papers/dorefanet.pdf),Shuchang Zhou, 2016

- [XNOR-Net: ImageNet Classification Using Binary
Convolutional Neural Networks](papers/xnornet.pdf), Mohammad Rastegari, 2016
- [BWN: Ternary weight networks](papers/1605.04711.pdf)
- [TNN Ternary Neural Networks for Resource-Efficient AI Applications](papers/1609.00222.pdf) Hande Alemdar1, 2016
- [Efficient Processing of Deep Neural Networks:
A Tutorial and Survey](papers/1703.09039.pdf)
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](papers/1510.00149.pdf)



- [A Survey of Model Compression and Acceleration
for Deep Neural Networks](papers/1710.09282.pdf)



### cvpr2018
- [Data Distillation: Towards Omni-Supervised Learning](./papers/DataDistillation.pdf)
- [PAD-Net: Multi-Tasks Guided Prediciton-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing](./papers/PAD-Net.pdf)
- [Fast and Accurate Single Image Super-Resolution via Information Distillation Network](./papers/1803.09454.pdf) 


### ICLR2018
- [Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy](./papers/4f34b75b21e3ff9bbf78aac9883cde51e420cb97.pdf)
- [Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks](./papers/kn_gan.pdf)

### NIPS2017
- [Learning Efficient Object Detection Models with Knowledge Distillation](./papers/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf) | [review](http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips30/reviews/483.html)



### Papers
- [Combining labeled and unlabeled data with co-training](papers/cotrain.pdf), A. Blum, T. Mitchell, 1998 
- [Model Compression](./papers/compression.kdd06.pdf), Rich Caruana, 2006
- [Dark knowledge](./papers/darkknowledgehintonppt.pdf), Geoffrey Hinton , OriolVinyals & Jeff Dean, 2014
- [Learning with Pseudo-Ensembles](./papers/1412.4864.pdf), Philip Bachman, Ouais Alsharif, Doina Precup, 2014
- [Distilling the Knowledge in a Neural Network](./papers/knowledge_distillation.pdf), Hinton, J.Dean, 2015
- [Cross Modal Distillation for Supervision Transfer](papers/1507.00448.pdf), Saurabh Gupta, Judy Hoffman, Jitendra Malik, 2015
- [Heterogeneous Knowledge Transfer in Video Emotion Recognition, Attribution and Summarization](./papers/1511.04798.pdf), Baohan Xu, Yanwei Fu, Yu-Gang Jiang, Boyang Li, Leonid Sigal, 2015
- [Distilling Model Knowledge](./papers/DistillingModelKnowledge.pdf), George Papamakarios, 2015
- [Learning Using Privileged Information: Similarity Control and Knowledge Transfer](papers/vapnik15b.pdf), Vladimir Vapnik, Rauf Izmailov, 2015

- [Unifying distillation and privileged information](papers/1511.03643.pdf), David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, Vladimir Vapnik, 2015

- [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](papers/1511.04508.pdf), Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami, 2016
- [Do deep convolutional nets really need to be deep and convolutional?](./papers/do___has_to_be.pdf), Gregor Urban, Krzysztof J. Geras, Samira Ebrahimi Kahou, Ozlem Aslan, Shengjie Wang, Rich Caruana, Abdelrahman Mohamed, Matthai Philipose, Matt Richardson, 2016

- [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](./papers/1612.03928.pdf), Sergey Zagoruyko, Nikos Komodakis, 2016
- [FitNets: Hints for Thin Deep Nets](./papers/fitnet.pdf), Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio, 2015
- [Deep Model Compression: Distilling Knowledge from Noisy Teachers](./papers/noisyteacher.pdf), Bharat Bhusan Sau, Vineeth N. Balasubramanian, 2016
- [Knowledge Distillation for Small-footprint Highway Networks](./papers/1608.00892.pdf), Liang Lu, Michelle Guo, Steve Renals, 2016
- [Sequence-Level Knowledge Distillation](./papers/sequenceKD.pdf), [deeplearning-papernotes](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/seq-knowledge-distillation.md), Yoon Kim, Alexander M. Rush, 2016
- [MobileID: Face Model Compression by Distilling Knowledge from Neurons](./papers/aaai16-face-model-compression.pdf), Ping Luo,  Zhenyao Zhu, Ziwei Liu, Xiaogang Wang and Xiaoou Tang, 2016
- [Recurrent Neural Network Training with Dark Knowledge Transfer](./papers/1505.04630.pdf), Zhiyuan Tang, Dong Wang, Zhiyong Zhang, 2016
- [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://arxiv.org/pdf/1612.03928.pdf), Sergey Zagoruyko, Nikos Komodakis, 2016
- [Adapting Models to Signal Degradation using Distillation](https://arxiv.org/abs/1604.00433), Jong-Chyi Su, Subhransu Maji,2016
- [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/pdf/1703.01780), Antti Tarvainen, Harri Valpola, 2017
- [Data-Free Knowledge Distillation For Deep Neural Networks](papers/1710.07535.pdf), Raphael Gontijo Lopes, Stefano Fenu, 2017 
- [Like What You Like: Knowledge Distill via Neuron Selectivity Transfer](papers/1707.01219.pdf), Zehao Huang, Naiyan Wang, 2017
- [Learning Loss for Knowledge Distillation with Conditional Adversarial Networks](https://arxiv.org/pdf/1709.00513), Zheng Xu, Yen-Chang Hsu, Jiawei Huang, 2017
- [DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer](https://arxiv.org/pdf/1707.01220), Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang, 2017
- [Knowledge Projection for Deep Neural Networks](https://arxiv.org/pdf/1710.09505), Zhi Zhang, Guanghan Ning, Zhihai He, 2017
- [Moonshine: Distilling with Cheap Convolutions](papers/1711.02613.pdf), Elliot J. Crowley, Gavin Gray, Amos Storkey, 2017
- [Local Affine Approximators for Improving Knowledge Transfer](./papers/LLD_2017_paper_28.pdf), Suraj Srinivas and Francois Fleuret, 2017
- [Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model](./papers/6635-best-of-both-worlds-transferring-knowledge-from-discriminative-learning-to-a-generative-visual-dialog-model.pdf), Jiasen Lu1, Anitha Kannan, Jianwei Yang, Devi Parikh, Dhruv Batra 2017
- [Learning Efficient Object Detection Models with Knowledge Distillation](./papers/6676-learning-efficient-object-detection-models-with-knowledge-distillation.pdf), Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, Manmohan Chandraker, 2017
- [Model Distillation with Knowledge Transfer from Face Classification to Alignment and Verification](./papers/1709.02929.pdf), Chong Wang, Xipeng Lan and Yangang Zhang, 2017
- [Learning Transferable Architectures for Scalable Image Recognition](./papers/1707.07012.pdf), Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le, 2017
- [Revisiting knowledge transfer for training object class detectors](./papers/1708.06128.pdf), Jasper Uijlings, Stefan Popov, Vittorio Ferrari, 2017
- [A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning](./papers/Yim_A_Gift_From_CVPR_2017_paper.pdf), Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim, 2017
- [Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net](./papers/1708.04106.pdf), Zihao Liu, Qi Liu, Tao Liu, Yanzhi Wang, Wujie Wen, 2017
- [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/pdf/1712.04440.pdf), Ilija Radosavovic, Piotr Dollár, Ross Girshick, Georgia Gkioxari, Kaiming He, 2017
- [Interpreting Deep Classifiers by Visual Distillation of Dark Knowledge](./papers/1803.04042.pdf), Kai Xu, Dae Hoon Park, Chang Yi, Charles Sutton, 2018
- [Efficient Neural Architecture Search via Parameters Sharing](./papers/1802.03268.pdf), Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean, 2018
- [Transparent Model Distillation](./papers/1801.08640.pdf), Sarah Tan, Rich Caruana, Giles Hooker, Albert Gordo, 2018
- [Defensive Collaborative Multi-task Training - Defending against Adversarial Attack towards Deep Neural Networks](./papers/1803.05123.pdf), Derek Wang, Chaoran Li, Sheng Wen, Yang Xiang, Wanlei Zhou, Surya Nepal, 2018 
- [Deep Co-Training for Semi-Supervised Image Recognition](./papers/1803.05984.pdf), Siyuan Qiao, Wei Shen, Zhishuai Zhang, Bo Wang, Alan Yuille, 2018
- [Feature Distillation: DNN-Oriented JPEG Compression Against Adversarial Examples](./papers/1803.05787.pdf), Zihao Liu, Qi Liu, Tao Liu, Yanzhi Wang, Wujie Wen, 2018
- [Multimodal Recurrent Neural Networks with Information Transfer Layers for Indoor Scene Labeling](./papers/1803.04687.pdf), Abrar H. Abdulnabi, Bing Shuai, Zhen Zuo, Lap-Pui Chau, Gang Wang, 2018
- [Large scale distributed neural network training through online distillation](./papers/1804.03235.pdf), Rohan Anil, Gabriel Pereyra, Alexandre Passos, Robert Ormandi, George E. Dahl, Geoffrey E. Hinton, 2018

***
### Videos
- [Dark knowledge](https://www.youtube.com/watch?v=EK61htlw8hY), Geoffrey Hinton, 2014
- [Model Compression](https://www.youtube.com/watch?v=0WZmuryQdgg), Rich Caruana, 2016
***
### Implementations

## MXNet
- [Bayesian Dark Knowledge](https://github.com/dmlc/mxnet/blob/master/example/bayesian-methods/bdk.ipynb)

## PyTorch
- [Attention Transfer](https://github.com/szagoruyko/attention-transfer)
- [Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model](https://github.com/jiasenlu/visDial.pytorch)
- [Interpreting Deep Classifier by Visual Distillation of Dark Knowledge](https://github.com/xukai92/darksight)
- [A PyTorch implementation for exploring deep and shallow knowledge distillation (KD) experiments with flexibility](https://github.com/peterliht/knowledge-distillation-pytorch)
- [Mean teachers are better role models](https://github.com/CuriousAI/mean-teacher)

## Lua
- [Example for teacher/student-based learning ](https://github.com/hoelzl/Academia)

## Torch
- [Distilling knowledge to specialist ConvNets for clustered classification ](https://github.com/natoromano/specialistnets)
- [Sequence-Level Knowledge Distillation](https://github.com/harvardnlp/seq2seq-attn), [Neural Machine Translation on Android](https://github.com/harvardnlp/nmt-android)
- [cifar.torch distillation](https://github.com/samirasamadi/Distillation)

## Theano
- [FitNets: Hints for Thin Deep Nets](https://github.com/net-titech/distillation/tree/master/FitNets)
- [Transfer knowledge from a large DNN or an ensemble of DNNs into a small DNN](https://github.com/tejasgodambe/knowledge-distillation)

## Lasagne + Theano
- [Experiments-with-Distilling-Knowledge](https://github.com/usholanb/Experiments-with-Distilling-Knowledge)

## Tensorflow
- [Deep Model Compression: Distilling Knowledge from Noisy Teachers](https://github.com/chengshengchan/model_compression)
- [Distillation](https://github.com/suhangpro/distillation)
- [An example application of neural network distillation to MNIST](https://github.com/akamaus/mnist-distill)
- [Data-free Knowledge Distillation for Deep Neural Networks](https://github.com/iRapha/replayed_distillation)
- [Inspired by net2net, network distillation ](https://github.com/luzai/NetworkCompress)
- [Deep Reinforcement Learning, knowledge transfer](https://github.com/arnomoonens/DeepRL/tree/master/agents/knowledgetransfer)
- [Knowledge Distillation using Tensorflow](https://github.com/DushyantaDhyani/kdtf)

## Caffe
- [Face Model Compression by Distilling Knowledge from Neurons](https://github.com/liuziwei7/mobile-id)
- [KnowledgeDistillation Layer (Caffe implementation)](https://github.com/wentianli/knowledge_distillation_caffe)
- [Knowledge distillation, realized in caffe ](https://github.com/smn2010/caffe_kdistill)
- [Cross Modal Distillation for Supervision Transfer](https://github.com/xiaolonw/fast-rcnn-distillation)

## Keras
- [Knowledge distillation with Keras](https://github.com/TropComplique/knowledge-distillation-keras)
- [keras google-vision's distillation ](https://github.com/GINK03/keras-distillation)
- [Distilling the knowledge in a Neural Network](https://github.com/tejasgodambe/knowledge-distillation)