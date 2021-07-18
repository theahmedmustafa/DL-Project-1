import numpy as np
from json import dump, load
np.set_printoptions(suppress=True)

class Convolution:
    
    # Constructor
    def __init__(self, num_f, f_size, in_shape, stride = 1):
        # Extracts vars
        num_c, in_dim, _ = in_shape
        # Assign vars
        self.num_f = num_f; self.f_size = f_size; self.num_c = num_c; self.in_dim = in_dim; self.s = stride
        self.out_dim = int((in_dim - f_size) / stride) + 1
        self.out_shape = (num_f, self.out_dim, self.out_dim)
        # Initialize Weigths and Biases
        f_dims = (num_f, num_c, f_size, f_size)
        scale = 1 / np.sqrt(np.prod(f_dims))
        self.filters = np.random.normal(0, scale, f_dims)
        self.biases = np.random.randn(num_f, 1)
        # Initialize Accumulators
        self.sigma_acc = self.biases * 0
        self.delta_acc = self.filters * 0
        # Initialize Adam Paras
        self.Vdw = self.Sdw = self.delta_acc
        self.Vdb = self.Sdb = self.sigma_acc
        self.t = 1
        
    # ReLU Activation Function
    def relu(self, arr):
        return arr * (arr > 0)
    
    # Forward Propagation
    def step(self, img):
        # Assign Input
        self.in_ = img
        # Initialize Output
        out = np.zeros(self.out_shape)
        # Slide Window
        img_i = out_i = 0
        while img_i + self.f_size <= self.in_dim:
            img_j = out_j = 0
            while img_j + self.f_size <= self.in_dim:
                # Convolve
                window = img[:, img_i:img_i+self.f_size, img_j:img_j+self.f_size]
                out[:, out_i, out_j] = np.sum(window * self.filters, (1, 2, 3)) + self.biases[:,0]
                # Slide left
                img_j += self.s; out_j += 1
            # Slide down
            img_i += self.s; out_i += 1
        # Return
        self.out = self.relu(out)
        return self.out
    
    # Back Propagation
    def back(self, grad):
        # Reverse Activation
        grad = grad * (self.out > 0)
        # Initialize Outputs
        sigmas = np.sum(grad, (1, 2)).reshape(-1, 1)
        deltas = np.zeros(self.filters.shape)
        global_grad = np.zeros(self.in_.shape)
         # Slide Window
        img_i = out_i = 0
        while img_i + self.f_size <= self.in_dim:
            img_j = out_j = 0
            while img_j + self.f_size <= self.in_dim:
                # Calculate dF
                window = self.in_[:, img_i:img_i+self.f_size, img_j:img_j+self.f_size]
                tiled = np.repeat(window[None, :, :, :], self.num_f, 0)
                deltas += tiled * grad[:, out_i, out_j].reshape((self.num_f, 1, 1, 1))
                # Calculate dI
                gradients = grad[:, out_i, out_j].reshape((self.num_f, 1, 1, 1))
                global_grad[:, img_i:img_i+self.f_size, img_j:img_j+self.f_size] += np.sum(gradients * self.filters, 0)
                # Slide left
                img_j += self.s; out_j += 1
            # Slide down
            img_i += self.s; out_i += 1
        # Accumulate
        self.sigma_acc += sigmas
        self.delta_acc += deltas
        # Return
        return global_grad
    
    # Train
    def update(self, alpha, batch_size, beta_1 = 0.9, beta_2 = 0.99):
        # Mini-Batch Updates
        dw = self.delta_acc / batch_size; self.delta_acc *= 0
        db = self.sigma_acc / batch_size; self.sigma_acc *= 0
        # Momentum
        self.Vdw = (beta_1 * self.Vdw) + (1 - beta_1) * dw
        self.Vdb = (beta_1 * self.Vdb) + (1 - beta_1) * db
        # RMS Prop
        self.Sdw = (beta_2 * self.Sdw) + (1 - beta_2) * (dw ** 2)
        self.Sdb = (beta_2 * self.Sdb) + (1 - beta_2) * (db ** 2)
        # Corrected Momentum
        Vdw = self.Vdw / (1  - beta_1**self.t)
        Vdb = self.Vdb / (1  - beta_1**self.t)
        # Corrected RMS Prop
        Sdw = self.Sdw / (1  - beta_2**self.t)
        Sdb = self.Sdb / (1  - beta_2**self.t)
        # Update Parameters
        eps = 1e-9
        self.filters -= alpha * (Vdw / (np.sqrt(Sdw) + eps))
        self.biases -= alpha * (Vdb / (np.sqrt(Sdb) + eps))
        self.t += 0
        
    # Reset Adam Parameters Function
    def resetAdam(self):
        self.Vdw *= 0; self.Sdw *= 0
        self.Vdb *= 0; self.Sdb *= 0
        self.t = 1
        
class Pool:
    
    # Constructor
    def __init__(self, f_size, in_shape, stride = 1):
        # Extracts vars
        num_c, in_dim, _ = in_shape
        # Assign vars
        self.f_size = f_size; self.num_c = num_c; self.in_dim = in_dim; self.s = stride
        self.out_dim = int((in_dim - f_size) / stride) + 1
        self.out_shape = (self.num_c, self.out_dim, self.out_dim)
        self.size = np.prod(self.out_shape)
        
    # Forward Propagation
    def step(self, img):
        # Assign Input
        self.in_ = img
        # Initialize Output and Mask
        out = np.zeros(self.out_shape)
        self.masks = []
        # Slide Window
        img_i = out_i = 0
        while img_i + self.f_size <= self.in_dim:
            img_j = out_j = 0
            while img_j + self.f_size <= self.in_dim:
                # Pool
                window = img[:, img_i:img_i+self.f_size, img_j:img_j+self.f_size]
                pooled = np.max(window, (1, 2))
                out[:, out_i, out_j] = pooled
                # Update masks
                mask = pooled.reshape((self.num_c, 1, 1)) == window
                val = (img_i, img_j, mask)
                self.masks.append(val)
                # Slide left
                img_j += self.s; out_j += 1
            # Slide down
            img_i += self.s; out_i += 1
        # Return
        return out
    
    # Back Propagation
    def back(self, grad):
        # Initialize Output and Mask
        out = np.zeros((self.num_c, self.in_dim, self.in_dim))
        # Loop over grad
        for i, val in enumerate(self.masks):
            # Gradient Array Indices
            grad_i = int(i / self.out_dim)
            grad_j = i % self.out_dim
            # Unpack Mask Val
            out_i, out_j, mask = val
            # Back Pool
            gradients = grad[:, grad_i, grad_j].reshape((self.num_c, 1, 1))
            out[:, out_i:out_i+self.f_size, out_j:out_j+self.f_size] = mask * gradients
        # Return
        return out
        
class Flat:
    
    # Forward Propagation
    def step(self, img):
        self.in_dim = img.shape
        return np.reshape(img, (img.size, 1))
    
    # Back Propagation
    def back(self, vec):
        return vec.reshape(self.in_dim)
        
class Dense:
    
    # Constructor
    def __init__(self, size, in_size, activation = 'relu'):
        # Assign vars
        self.size = size; self.activation = activation
        # Initialize Weights and Biases
        weights_dims = (size, in_size)
        self.weights = np.random.standard_normal(weights_dims) * 0.1
        self.biases = np.zeros([size, 1])
        # Initialize Accumulators
        self.sigma_acc = self.biases * 0
        self.delta_acc = self.weights * 0
        # Initialize Adam Paras
        self.Vdb = self.Sdb = self.sigma_acc
        self.Vdw = self.Sdw = self.delta_acc
        self.t = 1
        
    # ReLU Activation Function
    def relu(self, arr):
        return arr * (arr > 0)
    
    # Softmax Activation Function
    def softmax(self, arr):
        arr -= arr.max()
        exp = np.exp(arr)
        return exp / np.sum(exp)
    
    # Activation Manager Function
    def activate(self, arr):
        if self.activation == 'relu': return self.relu(arr)
        if self.activation == 'softmax': return self.softmax(arr)
        
    # Forward Propagation
    def step(self, vec):
        # Assign Input
        self._in = vec
        # Dot
        z = np.dot(self.weights, vec) + self.biases
        a = self.activate(z)
        # Return
        self.out = a
        return self.out
    
    # Back Propagation
    def back(self, grad):
        # Calculate sigma
        sigma = grad if self.activation == 'softmax' else grad * (self.out > 0)
        # Calculate delta
        delta = np.dot(sigma, self._in.T)
        # Accumulate
        self.sigma_acc += sigma
        self.delta_acc += delta
        # Return global gradient
        global_grad = np.dot(self.weights.T, sigma)
        return global_grad
    
    # Train
    def update(self, alpha, batch_size, beta_1 = 0.9, beta_2 = 0.99):
        # Mini-Batch Updates
        dw = self.delta_acc / batch_size; self.delta_acc *= 0
        db = self.sigma_acc / batch_size; self.sigma_acc *= 0
        # Momentum
        self.Vdw = (beta_1 * self.Vdw) + (1 - beta_1) * dw
        self.Vdb = (beta_1 * self.Vdb) + (1 - beta_1) * db
        # RMS Prop
        self.Sdw = (beta_2 * self.Sdw) + (1 - beta_2) * (dw ** 2)
        self.Sdb = (beta_2 * self.Sdb) + (1 - beta_2) * (db ** 2)
        # Corrected Momentum
        Vdw = self.Vdw / (1  - beta_1**self.t)
        Vdb = self.Vdb / (1  - beta_1**self.t)
        # Corrected RMS Prop
        Sdw = self.Sdw / (1  - beta_2**self.t)
        Sdb = self.Sdb / (1  - beta_2**self.t)
        # Update Parameters
        eps = 1e-9
        self.weights -= alpha * (Vdw / (np.sqrt(Sdw) + eps))
        self.biases -= alpha * (Vdb / (np.sqrt(Sdb) + eps))
        self.t += 0
        
    # Reset Adam Parameters Function
    def resetAdam(self):
        self.Vdw *= 0; self.Sdw *= 0
        self.Vdb *= 0; self.Sdb *= 0
        self.t = 1
        
class CNN:
    
    # Constructor
    def __init__(self):
        # Initialize Lists
        self.layers = []; self.cost_history = []; self.valid_cost_history = []
        
    # Add Layer Function
    def add(self, layer):
        self.layers.append(layer)
        
    # Forward Propagation
    def forward(self, img):
        out = img
        for layer in self.layers: out = layer.step(out)
        self.out = out
        return self.out
        
    # Back Propagation
    def backward(self, grad):
        out = grad
        for layer in reversed(self.layers):
            out = layer.back(out)
        
     # Train Model Function
    def train(self, X, Y, epochs = 50, alpha = 0.01, batch_size = 1000, X_valid = [], Y_valid = []):
        # Set Parameters
        self.alpha, self.batch_size = alpha, batch_size
        # Epoch
        for i in range(epochs):
            # Verbose
            print(f'\nEPOCH {i+1}/{epochs}')
            # Train over Dataset
            self.train_dataset(X, Y, batch_size)
            # Reset Optimizer
            for layer in self.layers:
                if isinstance(layer, Dense) or isinstance(layer, Convolution): layer.resetAdam()
            # Validation Loss
            if len(X_valid) != 0 and len(Y_valid) != 0:
                valid_cost = self.cal_dataset_loss(X_valid, Y_valid)
                print(f'Vaidation Dataset Cost: {valid_cost:.3f}')
                self.valid_cost_history.append(valid_cost)
    
    # Train Over Dataset
    def train_dataset(self, X, Y, batch_size):
        # Total Iterations
        iters = int(len(X) / batch_size)
        # Iteration
        for i in range(iters):
            # Get batch X and Y
            start = i * batch_size
            stop = start + batch_size
            if start + batch_size <= len(X):
                batch_X = X[start:stop]; batch_Y = Y[start:stop]
            else: 
                batch_X = X[start:]; batch_Y = Y[start:]
            # Train Over Batch
            self.train_batch(batch_X, batch_Y)
            # Print Batch Cost
            print(f'Iteration {i + 1}/{iters} - Cost: {self.cost_history[-1]:.3f}')
        # Print Average Dataset Cost
        costs = self.cost_history[-iters:]
        print(f'Average Batch Cost: {np.mean(costs):.3f}')
        
    # Train Over Batch
    def train_batch(self, X, Y):
        # Initialize Batch Cost
        self.latest_batch_cost = 0
        # Train Batch
        for x,y in zip(X, Y):
            self.train_example(x, y)
        # Update Cost History
        self.cost_history.append(self.latest_batch_cost / self.batch_size)
        # Update Model
        self.update_model()
        
    # Cycle One Example
    def train_example(self, img, y):
        # Forward Prop
        pred = self.forward(img)
        # Cost
        cost = self.cross_entropy_loss(pred, y)
        self.latest_batch_cost += cost
        # Backward Prop
        error = pred - y
        self.backward(error)
        
    # Cross Entropy Cost Function
    def cross_entropy_loss(self, pred, y):
        pred += 1e-9
        return -np.sum(np.log(pred) * y) / pred.shape[0]
    
    # Dataset Cost Function
    def cal_dataset_loss(self, X, Y):
        cost = 0
        for x, y in zip(X, Y):
            pred = self.forward(x)
            cost += self.cross_entropy_loss(pred, y)
        return cost / len(X)
        
    # Update Model Function
    def update_model(self):
        for layer in self.layers:
            if isinstance(layer, Dense) or isinstance(layer, Convolution): layer.update(self.alpha, self.batch_size)
                
    # Save Weights Function
    def save_weights(self, path):
        # Intialize Data
        data = {}
        for i in range(len(self.layers)):
            # Pick Layer
            layer = self.layers[i]
            # Is Conv or Dense Layer
            if not isinstance(layer, Dense) and not isinstance(layer, Convolution): continue
            # Get Layer Data
            weights_flat = layer.weights.flatten().tolist() if isinstance(layer, Dense) else layer.filters.flatten().tolist()
            weights_shape = layer.weights.shape if isinstance(layer, Dense) else layer.filters.shape
            biases_flat = layer.biases.flatten().tolist()
            biases_shape = layer.biases.shape
            value = (weights_flat, weights_shape, biases_flat, biases_shape)
            # Store Data
            data[i] = value
        # Save Data
        with open(path, 'w') as file:
            dump(data, file, indent = 2)
        file.close()
        # Print
        print('Weights saved in file', path)
        
    # load Weights Function
    def load_weights(self, path):
        # Load Data
        with open(path) as f:
            data = load(f)
        f.close()
        # Loop through Layers
        for i in data.keys():
            # Choose Layer
            layer = self.layers[int(i)]
            # Get Layer Data
            weights_flat, weights_shape, biases_flat, biases_shape = data[i]
            weights = np.reshape(weights_flat, weights_shape)
            biases = np.reshape(biases_flat, biases_shape)
            # Assign Data to layer
            if isinstance(layer, Convolution): layer.filters = weights
            elif isinstance(layer, Dense): layer.weights = weights
            layer.biases = biases
        # Print
        print('Weights loaded from file', path)