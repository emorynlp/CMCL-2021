from graphviz import Digraph

colors = ['aliceblue','yellow','aquamarine',
'orange','orangered','orchid','palevioletred', 'seagreen',
'seashell','sienna','chartreuse','skyblue','steelblue','tan','yellowgreen','thistle',
'indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan'
]


def gen_graph_non_dir(pairs, name, color, emotions, cm):
    dot = Digraph(format='png')
    title = name.split('/')[-1]
    dot.attr(label=title, labelloc='t', fontsize='10')
    link_node(dot, pairs, color, emotions, cm, reverse=False, arrowhead='none')
    dot.render(name, view=False)


def gen_graph_non_dir_color(pairs, repeated_pairs, name, color, emotions, cm):
    dot = Digraph(format='png')
    title = name.split('/')[-1]
    dot.attr(label=title, labelloc='t', fontsize='10')

#    for pair in pairs:
#        value = cm[emotions.index(pair[0])][emotions.index(pair[1])]
#        if pair in repeated_pairs:
#            color = colors[repeated_pairs.index(pair)]
#            dot.attr('node', style='filled', fillcolor=color)
#            dot.node(pair[0])
#            dot.node(pair[1])
#            dot.attr('node', style='filled', fillcolor='white')
#        dot.edge(pair[0], pair[1], arrowhead='none', color='black', label='{0:.2f}'.format(value))

    for pair in pairs:
        if pair in repeated_pairs:
            color = colors[repeated_pairs.index(pair)]
            dot.attr('node', style='filled', fillcolor=color)
            dot.node(pair[0])
            dot.node(pair[1])

    dot.attr('node', style='filled', fillcolor='white')
    for pair in pairs:
        if pair not in repeated_pairs:
            dot.node(pair[0])
            dot.node(pair[1])

    for pair in pairs:
        value = cm[emotions.index(pair[0])][emotions.index(pair[1])]
        dot.edge(pair[0], pair[1], arrowhead='none', color='black', label='{0:.2f}'.format(value))

    dot.render(name, view=False)

def gen_graph_cluster(dot, pairs, name, color, emotions, cm):
    title = name.split('/')[-1]
    dot.attr(label=title, labelloc='t', fontsize='10')

    new_pairs = []
    updated = {}
    for pair in pairs:
        dot.node('{}_{}'.format(name, pair[0]), label=pair[0])
        dot.node('{}_{}'.format(name, pair[1]), label=pair[1])
        new_pairs.append(('{}_{}'.format(name, pair[0]), '{}_{}'.format(name, pair[1])))
    for i, pair in enumerate(new_pairs):
        value = cm[emotions.index(pairs[i][0])][emotions.index(pairs[i][1])]
        dot.edge(pair[0], pair[1], arrowhead='none', color=color, label='{0:.2f}'.format(value))


def gen_graph(pairs, name, color, emotions, cm, reverse):
    dot = Digraph()
    if reverse:
        dot.attr(rankdir='BT')
    title = name.split('/')[-1]
    dot.attr(label=title, labelloc='t', fontsize='10')
    link_node(dot, pairs, color, emotions, cm, reverse, arrowhead='vee')
    dot.render(name, view=False)


def link_node(dot, pairs, color, emotions, cm, reverse, arrowhead):
    for pair in pairs:
        value = cm[emotions.index(pair[0])][emotions.index(pair[1])]
        if reverse:
            dot.edge(pair[1], pair[0], arrowhead=arrowhead, color=color, label='{0:.2f}'.format(value))
        else:
            dot.edge(pair[0], pair[1], arrowhead=arrowhead, color=color, label='{0:.2f}'.format(value))
