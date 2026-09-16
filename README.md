# Federated Continual Learning based on Knowledge Distillation with Personalized Local and Global Teachers

## Abstract
This study introduces a federated continual learning (FCL)
strategy to mitigate catastrophic forgetting while efficiently incorporat-
ing and adapting to new concepts. The proposed method, called KD-
FCL, is a novel FCL scheme that involves co-distillation at the server
level. The KD-FCL uses two teachers in each training round to distill
knowledge by grouping participating clients based on the similarity of
their class distributions. This scheme simulates the indirect exchange
of knowledge between clients at two levels. At the group level, clients
learn from their peers, and at the global level, they benefit from knowl-
edge available across the entire federation. This ensures that relevant
knowledge is always accessible, enabling the federation to mitigate for-
getting. The proposed FCL strategy is evaluated across several experi-
mental scenarios and benchmarked against two FL techniques (FedAvg
and FedProx) on Fashion-MNIST and CIFAR-10 datasets. In addition,
we compared the performance of KD-FCL with a sample selection (SS)
FCL strategy specially implemented for the evaluation and with the per-
formance of the two strategies applied in combination (SS-KD-FCL). The
results show that KD-FCL, FedAvg, and FedProx perform better than
SS-FCL and SS-KD-FCL in handling forgetting, with KD-FCL perform-
ing best in most experiments.

## Results
You can find the results and figures here: https://fileshot.io/f/727e1ccf7f79e0f8e2157ffbc04fb559#k=KbSufiiXg965BLonTJpEfOqfYkEhhwqCtcb_12HI89s

### Installation

```bash
pip install -r requirements.txt
```

### Individual Subfigures (in `figures/`)


### Model Architecture
- `SimpleNet` is a fully connected network for image classification.
- Input layer: flattened image tensor of shape `(input_channels * input_size * input_size,)`.
- Hidden layer: `Linear(input_dim, 128)` followed by `ReLU()`.
- Output layer: `Linear(128, num_classes)`.
- Default parameters:
  - `input_channels=1`, `input_size=28`, `num_classes=10` for Fashion-MNIST.
  - `input_channels=3`, `input_size=32`, `num_classes=10` for CIFAR-10.
- Loss: `CrossEntropyLoss()`.
- Optimizer: `SGD` with `lr=0.001`.
- Training batch size: `32`.

The model used in our experiments is a simple fully connected neural network, `SimpleNet`. For each input image, the network first flattens the tensor and then applies a single hidden layer with 128 units and ReLU activation. The final layer is a linear classification layer that outputs logits for the target classes. For Fashion-MNIST, the input is treated as a single-channel \(28 x 28\) image; for CIFAR-10, the input is treated as a three-channel \(32 x 32\) image. Training uses standard cross-entropy loss, stochastic gradient descent with a learning rate of 0.001, and a batch size of 32. This compact architecture was chosen to match the federated continual learning setup and focus evaluation on the impact of the proposed knowledge-distillation and personalized teacher mechanisms rather than on model capacity.

#### IEEE LaTeX table

The following table can be copied into an IEEEtran manuscript. The parameter count includes trainable weights and biases.

```latex
% Requires: \usepackage{array,tabularx}
\begin{table}[htbp]
\caption{Model architecture and training parameters.}
\begin{center}
\begin{tabularx}{\columnwidth}{|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|}
\hline
{\bfseries Model} & \multicolumn{3}{|c|}{\textbf{Architecture and Parameters}} \\
\cline{2-4}
{\bfseries Component} & \textbf{\textit{Input}} & \textbf{\textit{Output}} & \textbf{\textit{Trainable parameters}} \\
\hline
Flatten & $1\times28\times28$ or $3\times32\times32$ & 784 or 3,072 & 0 \\
\hline
Linear 1 & 784 or 3,072 & 128 & 101,760 or 393,344 \\
\hline
ReLU & 128 & 128 & 0 \\
\hline
Linear 2 & 128 & 10 & 1,290 \\
\hline
{\bfseries Total} & \multicolumn{2}{|c|}{\textbf{Fashion-MNIST / CIFAR-10}} & \textbf{101,770 / 394,634} \\
\hline
Loss & \multicolumn{3}{|c|}{Cross-entropy} \\
\hline
Optimizer & \multicolumn{3}{|c|}{SGD, learning rate $0.001$} \\
\hline
Training & \multicolumn{3}{|c|}{Batch size 32; one local epoch} \\
\hline
KD parameters & \multicolumn{3}{|c|}{$\lambda_{\mathrm{LL}}=0.9$, $\lambda_{\mathrm{GL}}=0.1$, $T=1.5$, $\lambda_{\mathrm{KD}}=0.05$} \\
\hline
FedProx / cFed & \multicolumn{3}{|c|}{$\mu\in\{\mathrm{disabled},0.01\}$; $\alpha=0.1$, $T_{\mathrm{cFed}}=2.0$} \\
\end{tabularx}
\par\noindent\parbox{\columnwidth}{\footnotesize Parameters include weights and biases; the first value is for Fashion-MNIST and the second for CIFAR-10.}
\label{tab:model-architecture}
\end{center}
\end{table}
```

### Clustering and teacher selection
- Client clustering is performed on client class-distribution probability vectors using `KMeans`.
- The optimal number of clusters `k` is chosen by an elbow method over `k_min=1..k_max`.
- Default clustering hyperparameters:
  - `k_min = 1`
  - `k_max = 10`
  - `n_init = 100`
  - `random_state = 42`
  - `improvement_threshold = 0.1`
- After clustering, each cluster defines a local leader, and the federation also selects a global leader across all clients.
- Local teacher for each client is formed from the current client, the cluster local leader, and a random peer in the same cluster.
- Global teacher for each client is formed from the current client, the global leader, and a random client across the federation.

### Metrics
- `fl_metrics_[mode].txt` - Performance summary
- Accuracy: overall test accuracy of the model after each federated round.
- Client accuracy: average accuracy computed over individual clients, reflecting personalized performance across the federation.
- Global accuracy: accuracy of the aggregated global model evaluated on the full test set.
- Average forgetting accuracy: mean accuracy drop across tasks or rounds relative to the best observed task accuracy.
- Average retention accuracy: mean retained accuracy for previously learned tasks or labels after learning new data.

## License

This project is for research and educational purposes.
