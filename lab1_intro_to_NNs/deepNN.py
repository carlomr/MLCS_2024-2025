import torch
from torch.autograd import Variable
from tqdm import tqdm, trange


torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu" # uncomment this line for using the CPU
torch.set_default_device(device) # default tensor device
print("I'm using: ", device)

class DeepNet(torch.nn.Module):
    # def __init__(self, n_input, activation):
    #     """ construct empty NN """
    #     super(DeepNet, self).__init__()  # Constructor of the super class torch.nn.Module
    #     self.L = 0
    #     self.dim = n_input
    #     self.activation = activation
    #     self.hidden = torch.nn.ModuleList()
    #     self.output = None

    def __init__(self, activation, n_input, n_hidden=None, n_output=None):
        """
        construct a NN with
        activation: activation function
        n_input: input dimension
        n_hidden: list of hidden layer widths
        n_output: output dim

        for example,
        nn = DeepNet(torch.relu, 1, [4,4,4], 1)

        to produce an empty network with relu and input_dim 1,
        nn = DeepNet(torch.relu, 1)
        """
        super(DeepNet, self).__init__()  # Constructor of the super class torch.nn.Module
        torch.manual_seed(0) # set the seed for reproducibility
        self.dim = n_input
        self.activation = activation
        self.hidden = torch.nn.ModuleList()
        if n_hidden is not None:
            self.L = len(n_hidden)
            self.widths = n_hidden
            self.hidden.append(torch.nn.Linear(n_input, n_hidden[0]))
            torch.nn.init.xavier_normal_(self.hidden[0].weight)
            torch.nn.init.normal_(self.hidden[0].bias)
            for i in range(1, self.L):
                self.hidden.append(torch.nn.Linear(n_hidden[i-1], n_hidden[i]))
                torch.nn.init.xavier_normal_(self.hidden[i].weight)
                torch.nn.init.normal_(self.hidden[i].bias)
        else:
            self.L = 0

        if n_output is not None:
            self.dim_out = n_output
            self.output = torch.nn.Linear(n_hidden[-1], n_output, bias=False)
            torch.nn.init.xavier_normal_(self.output.weight)
        else:
            self.output = None

    def add_layer(self, width):
        """
        add a hidden layer of given width to a NN
        """
        if self.L == 0:
            prev_layer_out_dim = self.dim
        else:
            prev_layer_out_dim = self.hidden[self.L-1].weight.shape[0]

        self.hidden.append(torch.nn.Linear(prev_layer_out_dim, width))
        torch.nn.init.xavier_normal_(self.hidden[self.L].weight)
        torch.nn.init.normal_(self.hidden[self.L].bias)
        self.L += 1

    def add_output_layer(self, n_output, bias=False):
        assert self.L > 0
        assert self.output is None
        prev_layer_out_dim = self.hidden[-1].weight.shape[0]
        self.output = torch.nn.Linear(prev_layer_out_dim, n_output, bias=bias)
        torch.nn.init.xavier_normal_(self.output.weight)
        if bias:
            torch.nn.init.normal_(self.output.bias)

    def set_weight(self, layer, weight_mat, requires_grad=True):
        assert self.L > layer
        assert weight_mat.shape == self.hidden[layer].weight.shape
        self.hidden[layer].weight = torch.nn.Parameter(weight_mat, requires_grad=requires_grad)

    def set_bias(self, layer, bias_mat):
        assert self.L > layer
        assert bias_mat.flatten().shape == self.hidden[layer].bias.shape
        self.hidden[layer].bias = torch.nn.Parameter(bias_mat)

    def set_output_weight(self, weight_mat, requires_grad=True):
        assert self.output is not None
        assert weight_mat.shape == self.output.weight.shape
        self.output.weight = torch.nn.Parameter(weight_mat, requires_grad=requires_grad)

    def set_output_bias(self, bias_mat):
        assert self.output is not None
        assert bias_mat.flatten().shape == self.output.bias.shape
        self.output.bias = torch.nn.Parameter(bias_mat)

    def forward(self, x):
        """
        Given input vector x produces the output of the NN
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1) # add a dimension at the end
        for i in range(self.L):
            x = self.hidden[i](x)
            x = self.activation(x)
            
        if self.output is not None:
            x = self.output(x)
        return x

    def output_layer(self, x, layer):
        """
        get the output of the hidden layer number
        "layer" (indexed starting from 0) of the NN
        """
        if layer == self.L:
            return self.forward(x)

        # if layer < self.L then
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        for i in range(layer+1):
            x = self.hidden[i](x)
            x = self.activation(x)
            # x = x[:, 0][:, None]
        return x
    
    def gradient(self, x):
        """
        output the gradient of the NN at points x
        x: torch tensor of size n_points by input_dim
        output: torch.tensor of size n_points by output_dim by input_dim
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        assert x.shape[1] == self.dim

        #Variable is deprecated
        #xvar = Variable(x, requires_grad=True)
        x.requires_grad_(True)
        y = self.forward(x)
        y_sum = y.sum(dim=0)
        dy = []
        for yi in y_sum:
            dyi, = torch.autograd.grad(yi, x, create_graph=True)
            dy.append(dyi)

        # dy = torch.autograd.functional.jacobian(self.forward, x)
        return torch.stack(dy).permute(1, 0, 2)

    def laplacian(self, x):
        """
        output the (semipos def) laplacian of a NN at points x
        x: torch tensor of size n_points by space_dim
        output: torch.tensor of size n_points by output_dim
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        assert x.shape[1] == self.dim
        #Variable is deprecated
        #xvar = Variable(x, requires_grad=True)
        x.requires_grad_(True)
        y = self.forward(x)
        y_sum = y.sum(dim=0)
        deltay = []
        for yi in y_sum:
            dyi, = torch.autograd.grad(yi, x, retain_graph=True,
                                       create_graph=True)
            deltayi = torch.zeros(x.shape[0])
            for i in range(self.dim):
                deltayi -= torch.autograd.grad(dyi[:, i].sum(),
                                               x, create_graph=True)[0][:, i]
            deltay.append(deltayi)

        return torch.stack(deltay, dim=1)

    def box_init_one_layer(self, i):
        """
        initialize the weights of the i-th layer of the NN
        using a custom method (based on the box method)
        """
        assert i <= self.L
        if i < self.L:
            win, wout = self.hidden[i].in_features, self.hidden[i].out_features
        else:
            win, wout = self.output.in_features, self.output.out_features

        p = torch.rand((wout, win))
        n = torch.normal(mean=torch.zeros((wout, win)), std=1)
        n = torch.nn.functional.normalize(n)  # TODO: it should be the squared norm here?
        pmax = torch.max(torch.sign(n), torch.zeros_like(n))
        k = torch.div(1, torch.sum((pmax-p)*n, dim=1))

        W = k.unsqueeze(-1)*n
        b = k*torch.sum(n*p, dim=1)

        return W, b

    def box_init(self):
        """
        initialize all the weights of the NN using the box method
        """
        for i in range(self.L):
            W, b = self.box_init_one_layer(i)
            self.hidden[i].weight = torch.nn.Parameter(W)
            self.hidden[i].bias = torch.nn.Parameter(b)

        if self.output is not None:
            W, b = self.box_init_one_layer(self.L)
            self.output.weight = torch.nn.Parameter(W)
            if self.output.bias is True:
                self.output.bias = torch.nn.Parameter(b)

    def freeze_hidden_layers(self):
        for layer in self.hidden:
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False

    def unfreeze_hidden_layers(self):
        for layer in self.hidden:
            layer.weight.requires_grad = True
            if layer.bias is not None:
                layer.bias.requires_grad = True

    def freeze_output_layer(self):
        if self.output is not None:
            self.output.weight.requires_grad = False
            if self.output.bias is not None:
                self.output.bias.requires_grad = False

    def unfreeze_output_layer(self):
        if self.output is not None:
            self.output.weight.requires_grad = True
            if self.output.bias is not None:
                self.output.bias.requires_grad = True

    def lsgd(self, n_gd_epochs, n_ls_it, input_data, data, loss_fun):
        """
        train the NN using the least squares gradient descend method

        n_gd_epochs: number of epochs of gradient descent
        n_ls_it: number of least squares iterations
        input_data: input data
        data: output data
        loss_fun: loss function
        
        The lsgd method thus alternates between solving a 
        least squares problem to set the output layer weights 
        and performing gradient descent to optimize the rest 
        of the network's parameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)

        for it in range(n_ls_it):
            last_hidden_output = self.output_layer(input_data, self.L-1)
            size_out = last_hidden_output.shape[1]

            out_weight = torch.linalg.lstsq(last_hidden_output, data).solution
            out_weight = out_weight[:size_out, :]

            self.set_output_weight(out_weight.t())
            self.freeze_output_layer()

            pbar = trange(n_gd_epochs)
            for epoch in pbar:
                loss = loss_fun(input_data, data, self)

                pbar.set_description(f"Main it {it}, Loss {loss:.2g}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class ResNet(DeepNet):
    def forward(self, x):
        """
        Given input vector x produces the output of the NN
        """
        if x.dim() == 1:
            x = x[:, None]

        # the one dimensional case works better this way:
        # TODO: understand what is happening
        # if x.shape[1] == 1:
        #     for i in range(self.L):
        # x = self.activation(self.hidden[i](x)) + x
        # else:

        x = self.activation(self.hidden[0](x)) + torch.sum(x, dim=1)[:, None]

        for i in range(1, self.L):
            x = self.activation(self.hidden[i](x)) + x

        if self.output is not None:
            x = self.output(x)
        return x

    def box_init_one_layer(self, i):
        if i == 0:
            return DeepNet.box_init_one_layer(self, i)
        else:
            if i < self.L:
                win, wout = self.hidden[i].in_features, self.hidden[i].out_features
            else:
                win, wout = self.output.in_features, self.output.out_features
            m = (1.+1./(self.L-1))**(i+1)
            p = torch.rand((wout, win))
            n = torch.normal(mean=torch.zeros((wout, win)), std=1)
            n = torch.nn.functional.normalize(n)  # TODO: it should be the squared norm here?
            pmax = m*torch.max(torch.sign(n), torch.zeros_like(n))
            k = torch.div(1, torch.sum((pmax-p)*n*(self.L-1), dim=1))

            W = k[:, None]*n
            b = k*torch.sum(n*p, dim=1)

            return W, b

    def output_layer(self, x, layer):
        """
        get the output of the hidden layer number
        "layer" (indexed starting from 0) of the NN
        """
        if layer == self.L:
            return self.forward(x)

        if x.dim() == 1:
            x = x[:, None]

        # the one dimensional case works better this way:
        # TODO: understand what is happening
        # if x.shape[1] == 1:
        #     for i in range(layer+1):
        #         x = self.activation(self.hidden[i](x))+x
        # else:
        x = self.activation(self.hidden[0](x)) + torch.sum(x, dim=1)[:, None]
        for i in range(1, layer+1):
            x = self.activation(self.hidden[i](x))+x
        return x