import matplotlib.pyplot as plt
import numpy as np

branch_node = dict(boxstyle = "round", ec = (1., 0.8, 0.8), fc = (1., 0.8, 0.8), )
leaf_node = dict(boxstyle = "sawtooth", ec = (1., 1., 1.), fc = "lightgreen", )
arrow_args = dict(arrowstyle = "<-")


def plotter(k, m, tree):
    figure = plt.figure(1, facecolor='white')
    figure.clf()

    plt.title('Decision Tree', color='b')
    axprops = dict(xticks=[], yticks=[])

    plotter.ax1 = plt.subplot(1, 1, 1, frameon = False, **axprops)
    plot_Tree.total_w = float(get_Leaf_Num(tree))
    plot_Tree.total_d = float(get_Tree_Depth(tree))
    plot_Tree.x_off = -0.5 / plot_Tree.total_w
    plot_Tree.y_off = 1.
    plot_Tree(tree, (0.5, 1.), '')
    
    filename = 'images/tree_k_' + str(k) + '_m_' + str(m) + '.png'
    plt.savefig(filename, bbox_inches='tight')

    # plt.show()
    

def plot_Mid_Text(center, parent, txt_string):
    # coordinate of the text
    # upper bound - lower bound + up & down
    x_mid = (parent[0] - center[0]) / 2. + center[0]
    y_mid = (parent[1] - center[1]) / 2. + center[1]

    plotter.ax1.text(x_mid, y_mid, txt_string)


def plot_Node(node_text, center, parent, node_type):
    # general shape of a node with arrow
    # create by using annotate
    if node_type == leaf_node:
        node_text = 'Y = ' + str(node_text)
    plotter.ax1.annotate(node_text, xy = parent, xycoords = 'axes fraction', 
                        xytext = center, textcoords = 'axes fraction', va = 'center', 
                        ha = 'center', bbox = node_type, arrowprops = arrow_args)

    
def plot_Tree(tree, parent, node_text):
    # get nums of leaves
    leaf_nums = get_Leaf_Num(tree)
        
    first_str = list(tree.keys())[0]

    # get coordinate
    center = (plot_Tree.x_off + (1.0 + float(leaf_nums)) / 2. / plot_Tree.total_w,
            plot_Tree.y_off)

    # text coordinate
    plot_Mid_Text(center, parent, node_text)

    # branch node
    plot_Node(first_str, center, parent, branch_node)
    child_dict = tree[first_str]
    plot_Tree.y_off -= 1. / plot_Tree.total_d

    # recursively draw the nodes
    for key in child_dict.keys():
        if type(child_dict[key]).__name__ == 'dict':
            plot_Tree(child_dict[key], center, str(key))
        else:
            plot_Tree.x_off += 1. / plot_Tree.total_w
            plot_Node(child_dict[key], (plot_Tree.x_off, plot_Tree.y_off),
                    center, leaf_node)
            plot_Mid_Text((plot_Tree.x_off, plot_Tree.y_off), center, str(key))
    plot_Tree.y_off += 1. / plot_Tree.total_d
    
    
def get_Leaf_Num(tree):
    # recursive find the nums of leaf nodes
    leaf_nums = 0
    first_str = list(tree.keys())[0]
    child_dict = tree[first_str]
    for key in child_dict.keys():
        # if node is a dict then not leaf, continue
        if type(child_dict[key]).__name__ == 'dict':
            leaf_nums += get_Leaf_Num(child_dict[key])
        else:
            # 每次发现一个节点就加一，最终的那个子叶也是加个1就跑了
            leaf_nums += 1
    
    return leaf_nums
    
def get_Tree_Depth(tree):
    # recursive find the depth of the decision tree
    max_depth = 0
    first_str = list(tree.keys())[0]
    child_dict = tree[first_str]
    for key in child_dict.keys():
        if type(child_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_Tree_Depth(child_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

 
if __name__ == '__main__':
    # test
    pass