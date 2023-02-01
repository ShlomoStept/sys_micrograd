# helper fucntions to visualize everything happening behind the scenes

from graphviz import Digraph

# Step 1 - build the connections/tree of operations and operands
def trace(root):
    # This builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    
    # function to build everything
    def build(v):
        if v not in nodes:
            nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            build(child)
    
    # now use this to build the trace fro, the root
    build(root)
    return nodes, edges

# Step 2 - Draw the connections/tree of operations and operands
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR - for left to right
    
    nodes, edges = trace(root)
    for _n in nodes:
        uid = str(id(_n))
        
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label="{ %s |  data %.4f | grad %.4f }" % (_n.label, _n.data, _n.grad ), shape='record')
        
        if _n._op:
            # a - if this value is a result of some operation, create an op node for it
            dot.node(name = uid + _n._op , label = _n._op)
            # b - and connec this node to it
            dot.edge(uid + _n._op, uid)
            
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge( str(id(n1)), str(id(n2)) + n2._op)
        
    return dot