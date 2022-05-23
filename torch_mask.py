
# Wrap model in prunerwrapper
pruner = get_pruner(model, pruner_name, sparsity, device, dataset, optimizer)
model = pruner.compress()
dim = 1
print(f'Dimension {dim}')
pruner.get_pruned_weights(dim=dim)


# Unwrap, remove PrunerModuleWrapper
pruner._unwrap_model()

weight_mask = wrapper.weight_mask
mask_size = weight_mask.size()
if len(mask_size) == 1:
    index = torch.nonzero(weight_mask.abs() != 0).tolist()
else:
    sum_idx = list(range(len(mask_size)))
    sum_idx.remove(dim)
    index = torch.nonzero(weight_mask.abs().sum(sum_idx) != 0).tolist()
    # remained filters: index
    # len(index)

masks = torch.load(temp_mask)
def print_mask(masks, d):
    print(f'Dimension {d}')
    for name, mask in masks.items():
        m_weight = mask['weight']
        sum_idx = list(range(len(m_weight.size())))
        sum_idx.remove(d)
        index = torch.nonzero(m_weight.abs().sum(sum_idx) != 0).tolist()
        print(f'{name} : Shape {list(m_weight.shape)} : reamined: {len(index)}')
        pass
        if d == 0 and 'bias' in mask and mask['bias'] is not None:
            bias_index = torch.nonzero(mask['bias'], as_tuple=True)[0]

### Dependency ###
"""
Build the graph for the model.
"""
from nni.common.graph_utils import TorchModuleGraph

# traced_model : torch._C.Graph, could be None
graph = TorchModuleGraph(model, dummy_input, traced_model)
dependency = dict()
CONV_TYPE = 'aten::_convolution'
ADD_TYPES = ['aten::add', 'aten::add_']
CAT_TYPE = 'aten::cat'

for node in graph.nodes_py.nodes_op:
    if node.op_type == 'Conv2d' or node.op_type == 'ConvTranspose2d':
        group = _get_conv_groups(node)

        if node.name in dependency:
            # the conv layer whose group is larger than 1 will require that
            # it's number of output channel to be divisible by the number of group.
            dependency[node.name] = max(
                dependency[node.name], group)
        else:
            dependency[node.name] = group
        if group > 1:
            # for the conv layer whose group is larger than 1, it will require the number
            # of output channels of their parent conv layer to be divisible by group.
            parent_convs = _get_parent_convs(node)
            for parent in parent_convs:
                if parent in dependency:
                    dependency[parent] = max(
                        dependency[parent], group)
                else:
                    dependency[parent] = group

def _get_parent_convs(graph, node):
    """
    Find the nearest father conv layers for the target node.

    Parameters
    ---------
    node : torch._C.Node
        target node.

    Returns
    -------
    parent_layers : list
        nearest father conv layers for the target node. Due to the group
        dependency only exists between the conv layers, so we only find
        the parent conv layers.
    """
    parent_layers = []
    # the input node is a Conv node
    predeessors = graph.find_predecessors(node.unique_name)
    predeessors = [graph.name_to_node[x] for x in predeessors]
    queue = predeessors
    while queue:
        curnode = queue.pop(0)
        if curnode.op_type == 'Conv2d' or curnode.op_type == 'ConvTranspose2d':
            # find the first met conv
            parent_layers.append(curnode.name)
            continue
        parents = graph.find_predecessors(curnode.unique_name)
        parents = [graph.name_to_node[name] for name in parents]
        for parent in parents:
            queue.append(parent)
    return parent_layers

def _get_conv_groups(node_group):
    """
    Get the number of groups for a convolutional layer.

    Parameters
    ----------
    node_group : NodePyGroup
        target node.

    Returns
    -------
    group : int
        the number of the groups of the target conv layer.
    """
    cpp_conv = list(filter(lambda x: x.kind() ==
                           CONV_TYPE, node_group.node_cpps))
    assert len(cpp_conv) == 1
    cpp_conv = cpp_conv[0]
    inputs = list(cpp_conv.inputs())
    # get the number of the group from the input parameters
    group = inputs[8].toIValue()
    return group


### Fix Mask ###
group_depen = GroupDependency(model, dummy_input)
depens = group_depen.dependency
import numpy as np
for layername in depens:
    group = depens[layername]
    if layername not in masks:
        # this layer not pruned
        continue
    w_mask = masks[layername]['weight']
    shape = w_mask.size()
    count = np.prod(shape[1:])
    all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
    all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
    if len(all_ones) + len(all_zeros) < w_mask.size(0):
        # In fine-grained pruning, skip this layer
        _logger.info('Layers %s using fine-grained pruning', layername)
        continue
    assert shape[0] % group == 0
    # Find the number of masked filter for each group (mini_masked).
    # Because we have to keep the pruned filter can still
    # be divided into the same number of groups, so we only can
    # prune mini_masked filters for each group.
    step = shape[0] / group
    group_masked = []
    for i in range(group):
        _start = step * i
        _end = step * (i + 1)
        _tmp_list = list(
            filter(lambda x: _start <= x and x < _end, all_zeros))
        group_masked.append(_tmp_list)
    mini_masked = min([len(x) for x in group_masked])
    print(f'name: {layername}, Group: {group}, Mask: {group_masked}, Mini: {mini_masked}')
