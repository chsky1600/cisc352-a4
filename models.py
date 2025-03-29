import nn

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        score = nn.as_scalar(self.run(x_point))
        return 1 if score >= 0 else -1

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False
        
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                # Get the true label as a scalar
                y_true = nn.as_scalar(y)
                # Get the prediction
                y_pred = self.get_prediction(x)
                
                # If the prediction is wrong, update the weights
                if y_pred != y_true:
                    converged = False
                    # Update weights
                    # Direction is multiplied by y_true because we need to move toward positive
                    # score for positive examples and negative score for negative examples
                    self.w.update(y_true, x)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Using an architecture that's known to work well for this problem
        self.w1 = nn.Parameter(1, 200)  # First layer weights
        self.b1 = nn.Parameter(1, 200)  # First layer bias
        self.w2 = nn.Parameter(200, 1)  # Second layer weights 
        self.b2 = nn.Parameter(1, 1)    # Second layer bias

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # First layer computation
        layer1 = nn.Linear(x, self.w1)
        layer1_with_bias = nn.AddBias(layer1, self.b1)
        layer1_activated = nn.ReLU(layer1_with_bias)
        
        # Output layer computation
        layer2 = nn.Linear(layer1_activated, self.w2)
        output = nn.AddBias(layer2, self.b2)
        
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.01
        batch_size = 20  # Smaller batch size for better fine-tuning
        
        # Get all parameters for convenience
        params = [self.w1, self.b1, self.w2, self.b2]
        
        # Train for a maximum number of iterations to prevent infinite loops
        for i in range(10000):  # Increased max iterations
            total_loss = 0
            num_batches = 0
            
            for x, y in dataset.iterate_once(batch_size):
                # Compute loss and gradients
                loss = self.get_loss(x, y)
                gradients = nn.gradients(params, loss)
                
                # Update parameters
                for j, param in enumerate(params):
                    param.update(-learning_rate, gradients[j])
                
                # Track average loss
                total_loss += nn.as_scalar(loss)
                num_batches += 1
            
            # Calculate average loss after an epoch
            avg_loss = total_loss / num_batches
            
            # Check if average loss is below threshold
            if avg_loss < 0.019:  # Lowered target for safety margin
                return
            
            # Learning rate schedule
            if i % 500 == 0 and i > 0:
                learning_rate *= 0.8  # Gradually reduce learning rate

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Create a neural network with two hidden layers
        # Input: 784 dimensions (28x28 flattened image)
        # First hidden layer: 200 dimensions
        # Second hidden layer: 100 dimensions
        # Output: 10 dimensions (one for each digit class)
        
        self.w1 = nn.Parameter(784, 200)   # First layer weights
        self.b1 = nn.Parameter(1, 200)     # First layer bias
        
        self.w2 = nn.Parameter(200, 100)   # Second layer weights
        self.b2 = nn.Parameter(1, 100)     # Second layer bias
        
        self.w3 = nn.Parameter(100, 10)    # Output layer weights
        self.b3 = nn.Parameter(1, 10)      # Output layer bias

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # First hidden layer
        layer1 = nn.Linear(x, self.w1)
        layer1_with_bias = nn.AddBias(layer1, self.b1)
        layer1_activated = nn.ReLU(layer1_with_bias)
        
        # Second hidden layer
        layer2 = nn.Linear(layer1_activated, self.w2)
        layer2_with_bias = nn.AddBias(layer2, self.b2)
        layer2_activated = nn.ReLU(layer2_with_bias)
        
        # Output layer (no ReLU after the last layer as instructed)
        output_layer = nn.Linear(layer2_activated, self.w3)
        output = nn.AddBias(output_layer, self.b3)
        
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_scores = self.run(x)
        return nn.SoftmaxLoss(predicted_scores, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.1
        batch_size = 100
        
        # We'll stop training when validation accuracy reaches 97.5%
        target_accuracy = 0.975
        
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        while True:
            for x, y in dataset.iterate_once(batch_size):
                # Compute loss and gradients
                loss = self.get_loss(x, y)
                gradients = nn.gradients(params, loss)
                
                # Update parameters
                for i in range(len(params)):
                    params[i].update(-learning_rate, gradients[i])
            
            # Check validation accuracy
            validation_accuracy = dataset.get_validation_accuracy()
            if validation_accuracy >= target_accuracy:
                return

