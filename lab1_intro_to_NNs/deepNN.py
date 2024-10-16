import torch
from torch.autograd import Variable
from tqdm import tqdm, trange


torch.set_default_dtype(torch.float64)


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
        self.dim = n_input
        self.activation = activation
        self.hidden = torch.nn.ModuleList()
        if n_hidden is not None:
            self.L = len(n_hidden)
            self.hidden.append(torch.nn.Linear(n_input, n_hidden[0]))
            torch.nn.init.xavier_normal_(self.hidden[0].weight)
            for i in range(1, self.L):
                self.hidden.append(torch.nn.Linear(n_hidden[i-1], n_hidden[i]))
                torch.nn.init.xavier_normal_(self.hidden[i].weight)
        else:
            self.L = 0

        if n_output is not None:
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
        self.L += 1

    def add_output_layer(self, n_output, bias=True):
        assert self.L > 0
        assert self.output is None
        prev_layer_out_dim = self.hidden[-1].weight.shape[0]
        self.output = torch.nn.Linear(prev_layer_out_dim, n_output, bias=bias)

    def set_weight(self, layer, weight_mat, requires_grad=True):
        assert self.L > layer
        assert weight_mat.shape == self.hidden[layer].weight.shape
        self.hidden[layer].weight = torch.nn.Parameter(weight_mat, requires_grad=requires_grad)

    def set_output_weight(self, weight_mat, requires_grad=True):
        assert self.output is not None
        assert weight_mat.shape == self.output.weight.shape
        self.output.weight = torch.nn.Parameter(weight_mat, requires_grad=requires_grad)

    def set_output_bias(self, bias_mat):
        assert self.output is not None
        assert bias_mat.flatten().shape == self.output.bias.shape
        self.output.bias = torch.nn.Parameter(bias_mat)

    def set_bias(self, layer, bias_mat):
        assert self.L > layer
        assert bias_mat.flatten().shape == self.hidden[layer].bias.shape
        self.hidden[layer].bias = torch.nn.Parameter(bias_mat)

    def forward(self, x):
        """
        Given input vector x produces the output of the NN
        """
        if x.dim() == 1:
            x = x[:, None]
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
            x = x[:, None]
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
            x = x[:, None]
        assert x.shape[1] == self.dim
        xvar = Variable(x, requires_grad=True)
        y = self.forward(xvar)
        y_sum = y.sum(dim=0)
        dy = []
        for yi in y_sum:
            dyi, = torch.autograd.grad(yi, xvar, retain_graph=True)
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
            x = x[:, None]
        assert x.shape[1] == self.dim
        xvar = Variable(x, requires_grad=True)
        y = self.forward(xvar)
        y_sum = y.sum(dim=0)
        deltay = []
        for yi in y_sum:
            dyi, = torch.autograd.grad(yi, xvar, retain_graph=True,
                                       create_graph=True)
            deltayi = torch.zeros(x.shape[0])
            for i in range(self.dim):
                deltayi -= torch.autograd.grad(dyi[:, i].sum(),
                                               xvar, create_graph=True)[0][:, i]
            deltay.append(deltayi)

        return torch.stack(deltay, dim=1)

    def box_init_one_layer(self, i):
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

        W = k[:, None]*n
        b = k*torch.sum(n*p, dim=1)

        return W, b

    def box_init(self):
        for i in range(self.L):
            W, b = self.box_init_one_layer(i)
            self.hidden[i].weight = torch.nn.Parameter(W)
            self.hidden[i].bias = torch.nn.Parameter(b)

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
        self.output.weight.requires_grad = False
        if self.output.bias is not None:
            self.output.bias.requires_grad = False

    def unfreeze_output_layer(self):
        self.output.weight.requires_grad = True
        if self.output.bias is not None:
            self.output.bias.requires_grad = True

    def lsgd(self, n_gd_epochs, n_ls_it, input_data, data, loss_fun):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)

        for it in range(n_ls_it):
            last_hidden_output = self.output_layer(input_data, self.L-1)
            size_out = last_hidden_output.shape[1]

            out_weight, _ = torch.lstsq(data, last_hidden_output)
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

