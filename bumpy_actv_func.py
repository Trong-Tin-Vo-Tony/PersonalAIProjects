import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

class Network(nn.Module):
    def __init__(self, *included_modules: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(included_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            x = module(x)
        return x

class BimodalNorm1d(nn.Module):
    """
    Custom Normalization Layer that splits standard normal data into a 
    Bimodal Distribution to feed twin-peak activation functions.
    """
    def __init__(self, num_features, initial_split=1.0, sharpness=10.0):
        super().__init__()
        # Standard BatchNorm to get the z-score (affine=False turns off standard scaling)
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        
        # Learnable Push Distance (d)
        self.d = nn.Parameter(torch.full((1, num_features), initial_split))
        self.sharpness = sharpness
        
        # Learnable Scale (gamma) and Shift (beta) to prevent falling into the outer dead zone
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x):
        # 1. Get standard z-score (centered at 0)
        z = self.bn(x)
        
        # 2. Apply the Bimodal "Pusher" Warp
        x_bimodal = z + self.d * torch.tanh(self.sharpness * z)
        
        # 3. Apply standard scaling and shifting
        return self.gamma * x_bimodal + self.beta

class SplitBatchNorm1d(nn.Module):
    """
    Normalizes the left and right halves of the data independently, 
    centering them at -1 and 1 to feed a twin-peak activation function.
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x):
        global_mean = x.mean(dim=0, keepdim=True)
        
        mask_left = (x < global_mean).float()
        mask_right = (x >= global_mean).float()
        
        count_left = mask_left.sum(dim=0, keepdim=True).clamp(min=1)
        count_right = mask_right.sum(dim=0, keepdim=True).clamp(min=1)
        
        mean_left = (x * mask_left).sum(dim=0, keepdim=True) / count_left
        mean_right = (x * mask_right).sum(dim=0, keepdim=True) / count_right
        
        var_left = (((x - mean_left) ** 2) * mask_left).sum(dim=0, keepdim=True) / count_left
        var_right = (((x - mean_right) ** 2) * mask_right).sum(dim=0, keepdim=True) / count_right
        
        z_left = (x - mean_left) / torch.sqrt(var_left + self.eps)
        z_right = (x - mean_right) / torch.sqrt(var_right + self.eps)
        
        shifted_left = z_left - 1.0
        shifted_right = z_right + 1.0
        
        x_split = (shifted_left * mask_left) + (shifted_right * mask_right)
        
        return self.gamma * x_split + self.beta

class CustomActivationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (8.0 * x**2) * torch.exp(-torch.abs(2.0 * x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        abs_x = torch.abs(x)
        
        # Exact condition to preserve gradient at x=0
        sign_x = torch.where(x < 0, -1.0, 1.0)
        
        term1 = 16.0 * abs_x * (1.0 - abs_x) * torch.exp(-2.0 * abs_x)
        term2 = torch.exp(-abs_x - 1.0)
        term3 = 0.5 * torch.sigmoid(3.0 * abs_x - 15.0)
        
        local_grad = sign_x * (term1 + term2 - term3)
        return grad_output * local_grad

class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return CustomActivationFunc.apply(x)

class BinaryStepWithSigmoidGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save the input for the backward pass (needed for sigmoid gradient)
        ctx.save_for_backward(input)
        
        # Binary Step: 1 if x >= 0, else 0
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        # Calculate the gradient of the sigmoid: sigma(x) * (1 - sigma(x))
        sigmoid_x = torch.sigmoid(input)
        grad_sigmoid = sigmoid_x * (1 - sigmoid_x)
        
        # Chain rule: incoming gradient * local gradient
        return grad_output * grad_sigmoid

class CustomBinaryStep(nn.Module):
    def __init__(self):
        super(CustomBinaryStep, self).__init__()

    def forward(self, x):
        return BinaryStepWithSigmoidGrad.apply(x)
    
class BinaryStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x >= 0).float()

class Swish(nn.Module):
    def __init__(self, beta=torch.Tensor([1.0])):
        super().__init__()
        self.beta = nn.Parameter(beta)
    
    def forward(self, x):
        # return x * torch.sigmoid(self.beta * x)
        return x * torch.sigmoid(x)
    
class ShowOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x)
        return x
    
class CustomAbsoluteFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.abs(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        local_grad = torch.where(x < 0, -1.0, 1.0)

        return grad_output * local_grad
    
class CustomELU(nn.Module):
    def __init__(self, base=torch.e, alpha=1.0):
        super().__init__()
        self.base = base
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * (torch.pow(self.base, x) - 1))

class FGI_ReLU(nn.Module):
    """
    Forward Gradient Injection ReLU (SUGAR methodology).
    Forward Pass: Strict standard ReLU (High Sparsity / Fast Inference).
    Backward Pass: Smooth ELU Surrogate (Prevents Dead Neurons).
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        # Alpha controls how strong the restorative gradient is in the negative zone
        self.alpha = alpha

    def forward(self, x):
        # 1. The Exact Forward Pass we want the network to output
        exact_forward = torch.nn.functional.relu(x)
        
        # 2. The Surrogate function we want the network to learn from
        # surrogate_backward = torch.nn.functional.elu(x, alpha=self.alpha)
        surrogate_backward = CustomELU(base=1.6, alpha=self.alpha)(x)
        
        # 3. The Detach Trick (Straight-Through Estimator)
        # Forward computes: surrogate + exact - surrogate = exact
        # Backward computes: gradient of surrogate
        return surrogate_backward + (exact_forward - surrogate_backward).detach()
    
class Custom_FGI_ReLU(nn.Module):
    """
    Forward Gradient Injection ReLU (SUGAR methodology).
    Forward Pass: Strict standard ReLU (High Sparsity / Fast Inference).
    Backward Pass: Smooth ELU Surrogate (Prevents Dead Neurons).
    """
    def __init__(self, base=torch.e, alpha=1.0, threshold=0.0):
        super().__init__()
        # Alpha controls how strong the restorative gradient is in the negative zone
        self.base = base
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, x):
        # 1. The Exact Forward Pass we want the network to output
        exact_forward = torch.nn.functional.relu(x)
        
        # 2. The Surrogate function we want the network to learn from
        # surrogate_backward = torch.nn.functional.elu(x, alpha=self.alpha)
        surrogate_backward = torch.where(x >= self.threshold, x, self.alpha * torch.pow(self.base, x - self.threshold))
        
        # 3. The Detach Trick (Straight-Through Estimator)
        # Forward computes: surrogate + exact - surrogate = exact
        # Backward computes: gradient of surrogate
        return surrogate_backward + (exact_forward - surrogate_backward).detach()

class CustomAbsolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return CustomAbsoluteFunc.apply(x)
    
class ReLULeakyGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.nn.functional.relu_(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        local_grad = torch.where(x > 0, 1, -0.1)

        return grad_output * local_grad 

class ReLULeakyGrad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ReLULeakyGradFunc.apply(x) 
    
class Print(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x)
        return x

class RepurposedYeoJohnson(nn.Module):
    def __init__(self):
        super().__init__()

    def func_for_positive(self, x):
        return ((x + 1) ** -0.3 - 1) / -0.3
    
    def func_for_negative(self, x):
        return -((-x + 1) ** -0.3 - 1) / -0.3
    
    def forward(self, x):
        return torch.where(x >= 0, self.func_for_positive(x), self.func_for_negative(x))

class SortByMagnitude(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Sort each row by the absolute value of its elements, but keep the original signs
        sorted_indices = torch.argsort(torch.abs(x), dim=1, descending=True)
        sorted_x = torch.gather(x, dim=1, index=sorted_indices)
        return sorted_x

class ShearLayer(nn.Module):
    def __init__(self, shear_factor=0.0):
        super().__init__()

        self.shear_factor = nn.Parameter(torch.tensor(shear_factor))

    def forward(self, x):
        # Apply shear transformation (simplified version)
        return x + self.shear_factor * torch.roll(x, shifts=1, dims=1)
    
class AverageMagnitude(nn.Module):
    def __init__(self):
        super(AverageMagnitude, self).__init__()

    def forward(self, x):
        """
        Calculates the mean of the absolute values of the input tensor.
        
        Args:
            x (torch.Tensor): The input matrix/tensor.
            
        Returns:
            torch.Tensor: A scalar tensor representing the average magnitude.
        """
        # torch.abs ensures all values are positive (magnitude)
        # .mean() computes the average of those values
        return torch.abs(x).mean()

class TensorStatsMonitor(nn.Module):
    def __init__(self, name="MonitorLayer", eps=1e-9):
        """
        An identity layer that prints statistics of the input tensor.
        
        Args:
            name (str): A label to identify which monitor is printing.
            eps (float): A small value added for geometric mean stability.
        """
        super().__init__()
        self.name = name
        self.eps = eps

    def forward(self, x):
        # Wrap everything in torch.no_grad() so we don't accidentally 
        # add these calculation nodes to your computation graph or eat up VRAM!
        with torch.no_grad():
            # Basic stats
            t_min = x.min().item()
            t_max = x.max().item()
            t_mean = x.mean().item()
            t_median = x.median().item()
            
            # Variance and Standard Deviation (requires more than 1 element)
            t_var = x.var().item() if x.numel() > 1 else 0.0
            t_std = x.std().item() if x.numel() > 1 else 0.0
            
            # Sparsity (percentage of exact zeros)
            sparsity = (x == 0).float().mean().item()
            
            # Geometric Mean: Exp(Mean(Log(abs(x) + eps)))
            # Note: Geometric mean is technically only defined for strictly positive 
            # numbers. To avoid crashing on negative values or zeros (common in NNs), 
            # we use absolute values and add a tiny epsilon.
            geo_mean = torch.exp(torch.log(x.abs() + self.eps).mean()).item()
            
            # Bonus debugging metrics
            l2_norm = torch.norm(x).item()
            num_nans = torch.isnan(x).sum().item()
            num_infs = torch.isinf(x).sum().item()
            
            # Print output formatted neatly
            print(f"\n--- Tensor Stats: {self.name} ---")
            print(f"Shape:     {list(x.shape)}")
            print(f"Min/Max:   {t_min:.4f} / {t_max:.4f}")
            print(f"Mean/Med:  {t_mean:.4f} / {t_median:.4f}")
            print(f"Std/Var:   {t_std:.4f} / {t_var:.4f}")
            print(f"Geo Mean:  {geo_mean:.4f} (calculated on absolute values)")
            print(f"Sparsity:  {sparsity:.2%}")
            print(f"L2 Norm:   {l2_norm:.4f}")
            
            # Alert strongly if math is breaking down
            if num_nans > 0 or num_infs > 0:
                print(f"⚠️ WARNING: {num_nans} NaNs and {num_infs} Infs detected!")
            print("-" * 40)
            
        # Return the tensor completely unmodified
        return x
    
class BatchTensorStatsMonitor(nn.Module):
    def __init__(self, name="BatchMonitorLayer", eps=1e-9):
        """
        Randomly selects one sample from a batch and uses TensorStatsMonitor 
        to print its statistics.
        """
        super().__init__()
        self.name = name
        
        # Instantiate the base monitor as a submodule
        self.base_monitor = TensorStatsMonitor(name=f"{name} (Sample)", eps=eps)

    def forward(self, x):
        if x.dim() == 0 or x.shape[0] == 0:
            return x

        with torch.no_grad():
            batch_size = x.shape[0]
            idx = torch.randint(0, batch_size, (1,)).item()
            sample = x[idx]
            
            # Print the batch context before the base monitor prints the stats
            print(f"\n[Batch Context] Layer: {self.name} | Selected Index: {idx}/{batch_size-1} | Full Batch Shape: {list(x.shape)}")
            
            # Pass the isolated sample through our base monitor
            # We don't need to capture the output since it's just an identity pass
            self.base_monitor(sample)
            
        # Return the original, full-batch tensor completely unmodified
        return x

def create_network(actv_funcs: list):
    return Network(
        nn.Linear(28 * 28, 200),
        actv_funcs[0],
        nn.Linear(200, 200),
        actv_funcs[1],
        nn.Linear(200, 200),
        actv_funcs[2],
        nn.Linear(200, 200),
        actv_funcs[3],
        BatchTensorStatsMonitor(name="Batch Stats Monitor"),
        nn.Linear(200, 100),
        actv_funcs[4],
        nn.Linear(100, 10),
    ).to(device="cuda") 

def train(model: Network, optimizer, loss_fn, data, answer) -> torch.Tensor:
    output = model(data)
    loss = loss_fn(output, answer)
    optimizer.zero_grad()
    loss.backward()

    # for name, param in model.named_parameters():
    #     print(f"{name}'s Gradient")
    #     param.grad[torch.abs(param.grad) < 0.0001] = 0.0
    #     print(param.grad)

    optimizer.step()
    return loss

if __name__ == "__main__":
    # Load and format the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 2. Split the training dataset
    # We want 1,000 for training, and we just ignore the remaining 59,000
    generator = torch.Generator().manual_seed(42) # Seed for reproducibility
    train_subset, ignored_data = random_split(train_dataset, [1_000, 59_000], generator=generator)

    # 3. Create the DataLoaders using the SUBSET
    # Now, one epoch is exactly 1,000 images (approx. 16 batches of size 64)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Architecture: Pairing BimodalNorm with your CustomActivation



    # model = create_network(
    #     [CustomActivation() for _ in range(5)]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), nn.ReLU()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), nn.ReLU())]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), ReLULeakyGrad()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), ReLULeakyGrad())]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), FGI_ReLU()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), FGI_ReLU())]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(5000), Custom_FGI_ReLU(base=10, threshold=0.5)) for _ in range(4)] + [Network(nn.BatchNorm1d(5000), Custom_FGI_ReLU(base=10, threshold=0.5))]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), RepurposedYeoJohnson()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), RepurposedYeoJohnson())]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), Swish(), SortByMagnitude()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), Swish(), SortByMagnitude())]
    # )  

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), Swish()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), Swish())]
    # )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), CustomAbsolute()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), CustomAbsolute())]
    # )

    model = create_network(
        [Network(nn.BatchNorm1d(200), CustomBinaryStep()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), CustomBinaryStep())]
    )

    # model = create_network(
    #     [Network(nn.BatchNorm1d(200), BinaryStep()) for _ in range(4)] + [Network(nn.BatchNorm1d(100), BinaryStep())]
    # )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 
    loss_fn = nn.CrossEntropyLoss().to(device="cuda")

    print("Starting training (BimodalNorm + Custom Activation)...")
    model.train() 
    
    total_samples = 0
    for epoch in range(10):
        for batch_idx, (img_data, labels) in enumerate(train_loader):
            img_data = img_data.to("cuda")
            labels = labels.to("cuda")
            
            # Binarize the images
            img_data = torch.where(img_data >= 0.5, 1.0, 0.0)

            loss = train(model=model, optimizer=optimizer, loss_fn=loss_fn, data=img_data, answer=labels)
            
            total_samples += img_data.size(0)
            
            if batch_idx % 1 == 0:
                print(f"Processed {total_samples} samples: Loss = {loss.item():.4f}")
                

    print("\nStarting testing...")
    model.eval() 
    correct = 0
    total_test = 0
    
    with torch.no_grad():
        for batch_idx, (img_data, labels) in enumerate(test_loader):
            img_data = img_data.to("cuda")
            labels = labels.to("cuda")
            
            output = model(img_data)
            predictions = output.argmax(dim=1, keepdim=True)
            correct += predictions.eq(labels.view_as(predictions)).sum().item()
            
            total_test += img_data.size(0)
            if total_test >= 10_000: 
                break

    print(f"\nFinal Accuracy: {correct / total_test:.2%}")

# Network: Linear(784, 100), ActvFunc(), Linear(100, 50), ActvFunc(), Linear(50, 20), ActvFunc(), Linear(20, 10)
# Binary Image Sharpener: True
# BatchNorm: False
# Learning Rate: 0.0001?
# Epoch: 1/12 (5,000 data)
# Custom: 94.20%
# Swish (with Beta): 87.50%
# Swish (without Beta): 89.20%
# ReLU: 90.00%

# Network: Linear(784, 200), ActvFunc(), Linear(200, 100), ActvFunc(), Linear(100, 100), ActvFunc(), Linear(100, 10)
# Binary Image Sharpener: True
# BatchNorm: False
# Learning Rate: 0.00005
# Epoch: 1/6 (10,000 data)
# Swish (without Beta): 91.20%
# Swish (With Beta): 88.60%
# Custom: 96.00%
# ReLU: 92.10%

# Network: Linear(784, 200), ActvFunc(), Linear(200, 100), ActvFunc(), Linear(100, 100), ActvFunc(), Linear(100, 10)
# Binary Image Sharpener: True
# Learning Rate: 0.001
# Epoch: 1/6 (10,000 data)
# BatchNorm: True
# Swish (without Beta): 92.97%, 92.19%
# Swish (With Beta): 92.38%, 92.19%
# Custom: 9.96%, 10.16%, 9.96%, 11.52%
# ReLU: 93.26%, 92.68%

# Network: Linear(784, 200), ActvFunc(), Linear(200, 100), ActvFunc(), Linear(100, 100), ActvFunc(), Linear(100, 10)
# Binary Image Sharpener: True
# Learning Rate: 0.001
# Epoch: 1/6 (10,000 data)
# BatchNorm: False
# Swish (without Beta): 89.06%, 87.99%, 87.40%
# Swish (With Beta): 87.70%, 87.70%, 89.55%
# Custom: 92.19%, 91.99%, 92.77%
# ReLU: 88.48%, 88.77%, 87.21%
