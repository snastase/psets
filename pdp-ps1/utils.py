import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn as nn
import semantic_cognition

def plot_decision_regions(
    predict, X, y, resolution=0.02, size=100, ax=None,
    title=None,
    # This parameter forces the decision region plot to be PNG due to issues with SVG display.
    # This should be removed for notebooks that prefer only PNG display.
    force_matplotlib_output_png_hack=True):
    """Plot decision boundaries.

    Parameters
    ----------
    X : array
        The set of all Boolean pairs.
    y : array
        The corresponding XOR labels.
    resolution : float
        Resolution of decision contour plots.
    size : int
        Marker size.
    ax : Matplotlib object.
        Canvas to plot onto.
    """
    if force_matplotlib_output_png_hack:
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('png')

    ## HACK Adapting sizes from CGC's code to work with this piece of SZ's code
    X = X.T
    ## HACK Force arrays to numpy
    if torch.is_tensor(X):
        X = X.numpy()
    if torch.is_tensor(y):
        y = y.numpy()

    ## Define markers / colormap.
    markers = ('x', 'o', 's', '^', 'v')
    colors = ('gray', 'lightgreen', 'blue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:max(2, len(np.unique(y)))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution).astype(np.float32),
                           np.arange(x2_min, x2_max, resolution).astype(np.float32))
    #Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = predict(torch.tensor([xx1.ravel(), xx2.ravel()])).detach()
    Z = Z.reshape(xx1.shape)

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    if title is not None:
        ax.set(title=title)
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set(xlim=(xx1.min(), xx1.max()), ylim=(xx2.min(), xx2.max()))

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1], s=size, alpha=0.9,
                    marker=markers[idx], color=colors[idx], label=cl)

    if force_matplotlib_output_png_hack:
        import matplotlib.pyplot as plt
        plt.show()
        set_matplotlib_formats('svg')

    return ax


def fit(loss, parameters, *, lr=0.1, num_epochs=100, plot_loss=True, opt_fn=torch.optim.SGD):
    history, parameter_history = [], []
    parameters = list(parameters)
    opt = opt_fn(parameters, lr=lr)

    for t in range(num_epochs):
        J = loss()

        opt.zero_grad()
        J.backward()
        opt.step()

        history.append(J)
        parameter_history.append([p.detach().clone() for p in parameters])

    if plot_loss:
        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.xlabel('# epochs')
        plt.ylabel('training loss')

    return history, parameter_history


def fit_model(loss, model, **kw):
    history, parameter_history = fit(loss, model.parameters(), **kw)
    models = []
    for parameters in parameter_history:
        m = model.new_instance(parameters, with_noise=False)
        models.append(m)
    return models


class Rumelhart(nn.Module):
    def __init__(self, nouns, relations, ylabels, *,
                 with_noise=True,
                 previous_model=None,
                 representation_units=9,
                 relational_units=16):
        super(Rumelhart, self).__init__()

        self.nouns = nouns
        self.relations = relations
        self.ylabels = ylabels
        self.with_noise = with_noise

        # Defining layers.
        self.noun_representation = nn.Linear(len(nouns), representation_units)
        self.relational = nn.Linear(representation_units+len(relations), relational_units)
        self.output = nn.Linear(relational_units, len(ylabels))

    def forward(self, x):
        noun = x[:, :len(self.nouns)]
        relation = x[:, len(self.nouns):]
        x = self.noun_representation(noun).sigmoid()
        if self.with_noise:
            x = x + torch.randn(size=x.shape) * 0.05
        x = self.relational(torch.cat([x, relation], axis=1)).sigmoid()
        x = self.output(x).sigmoid()
        return x

    def copy_from_parameters(self, parameters):
        '''
        Copy parameters from a list. Should typically come from another model
        via `model.parameters()`.
        '''
        for source, dest in zip(parameters, self.parameters()):
            dest.data[:] = source.data[:]

    def clone(self):
        '''
        Clones the current network instance, making it possible to modify the clone
        with no effect to the original.
        '''
        return self.new_instance(parameters=self.parameters())

    def predict_query(self, noun, relation):
        xlabels = self.nouns + self.relations
        return self(onehot(xlabels, (noun, relation))[None, :]).squeeze()

    def new_instance(self, parameters=None, with_noise=None):
        '''
        This function creates a new model instance with the same architecture and dataset.
        It optionally copies the parameters from an argument.
        '''
        m = Rumelhart(
            self.nouns,
            self.relations,
            self.ylabels,
            with_noise=with_noise if with_noise is not None else self.with_noise,
            representation_units=self.noun_representation.weight.shape[0],
            relational_units=self.relational.weight.shape[0],
        )
        if parameters is not None:
            m.copy_from_parameters(parameters)
        return m

def onehot(ordered_values, query):
    '''
    Takes a list of all possible values and query to encode.
    Returns onehot encoded query. Examples:

    >>> onehot(['dog', 'cat', 'plant'], ['cat'])
    torch.tensor([0, 1, 0])
    >>> onehot(['cat', 'meowing', 'green'], ['cat', 'meowing'])
    torch.tensor([1, 1, 0])
    '''
    x = torch.zeros(len(ordered_values))
    for item in query:
        x[ordered_values.index(item)] = 1
    return x

def onehot_decode(ordered_values, coded):
    '''
    Takes a list of all possible values and a coded array to decode.
    Returns list of values. Examples:

    >>> onehot_decode(['dog', 'cat', 'plant'], torch.tensor([0, 1, 0]))
    ['cat']
    >>> onehot_decode(['cat', 'meowing', 'green'], torch.tensor([1, 1, 0]))
    ['cat', 'meowing']
    '''
    assert len(ordered_values) == len(coded), f'Length of all possible values was {len(ordered_values)} but coded array has length {len(coded)}'
    return [ordered_values[i] for i, c in enumerate(coded) if c]

def load_semantic_cognition_data(common_noun=None, equal_word_frequency=False, module=semantic_cognition):

    # Make sure common_noun is a list for the below...
    if common_noun is None:
        common_noun = []
    elif not isinstance(common_noun, list):
        common_noun = [common_noun]

    nouns = [
        'pine', 'oak', 'maple', 'birch',
        'rose', 'daisy', 'tulip', 'sunflower',
        'robin', 'canary', 'sparrow', 'penguin',
        'sunfish', 'salmon', 'flounder', 'cod']
    relations = module.relations

    xlabels = nouns + relations
    ylabels = module.ylabels

    ylabels_relations = module.ylabels_relations
    relation_filter = {
        rel: torch.tensor([rel == label for label in ylabels_relations]).float()
        for rel in relations
    }

    X = torch.zeros((len(nouns)*len(relations), len(xlabels)))
    Y = torch.zeros((len(nouns)*len(relations), len(ylabels)))

    frequency = torch.ones((Y.shape[0]))
    i = 0
    for idx, noun in enumerate(nouns):
        for relation in relations:
            # Onehot coding our inputs & outputs
            X[i] = onehot(xlabels, (noun, relation))
            # Filtering the onehot coded properties to only include those appropriate to the current relation
            Y[i] = onehot(ylabels, getattr(module, noun)) * relation_filter[relation]

            # Now we compute differences in frequency...
            # Increase frequency of all properties of common noun.
            if noun in common_noun:
                frequency[i] *= 8

            if equal_word_frequency:
                # Make sure words are of equal frequency. For our simplified dataset,
                # we simply make basic-level words happen 1/4 of the time
                # and general words 1/8 of the time.
                if relation == 'isa-general':
                    frequency[i] *= 1/8
                elif relation == 'isa-basic':
                    frequency[i] *= 1/4

            # In general, we decrease the frequency of words.
            if relation in ['isa-general', 'isa-basic', 'isa-specific']:
                frequency[i] *= 1/2

            i += 1

    # Normalize frequency
    frequency /= frequency.sum()

    return nouns, relations, xlabels, ylabels, X, Y, frequency
