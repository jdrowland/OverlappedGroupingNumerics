import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import numpy as np

# ============ Graph construction (shared) ============

def build_graph():
    G = nx.Graph()
    n_groups = 5
    group_size = 5

    core_nodes = list(range(25))
    G.add_nodes_from(core_nodes)

    for i in core_nodes:
        for j in core_nodes:
            if i < j:
                G.add_edge(i, j)

    groups = []
    for g in range(n_groups):
        group = [g * group_size + i for i in range(group_size)]
        groups.append(group)

    pendant_nodes = [25, 26, 27, 28]
    G.add_nodes_from(pendant_nodes)

    for p_idx, pendant in enumerate(pendant_nodes):
        for node in groups[p_idx]:
            G.add_edge(pendant, node)

    return G, core_nodes, groups, pendant_nodes

def build_positions(scale=1.0, offset=(0, 0)):
    pos = {}
    n_groups = 5
    group_size = 5

    group_radius = 2.0 * scale
    pendant_radius = 3.5 * scale
    cluster_radius = 0.5 * scale

    group_centers = []
    for g in range(n_groups):
        angle = np.pi/2 - 2 * np.pi * g / 5
        group_centers.append((group_radius * np.cos(angle), group_radius * np.sin(angle)))

    pendant_positions = []
    for g in range(4):
        angle = np.pi/2 - 2 * np.pi * g / 5
        pendant_positions.append((pendant_radius * np.cos(angle), pendant_radius * np.sin(angle)))

    groups = []
    for g in range(n_groups):
        group = [g * group_size + i for i in range(group_size)]
        groups.append(group)

    pendant_nodes = [25, 26, 27, 28]

    for g in range(n_groups):
        cx, cy = group_centers[g]
        for i, node in enumerate(groups[g]):
            angle = np.pi/2 - 2 * np.pi * i / 5
            pos[node] = (cx + cluster_radius * np.cos(angle) + offset[0],
                        cy + cluster_radius * np.sin(angle) + offset[1])

    for p_idx, pendant in enumerate(pendant_nodes):
        px, py = pendant_positions[p_idx]
        pos[pendant] = (px + offset[0], py + offset[1])

    return pos, groups, pendant_nodes

def rounded_hull_path(hull_points, corner_radius=0.15):
    n = len(hull_points)
    if n < 3:
        return hull_points

    path_verts = []
    path_codes = []

    for i in range(n):
        p_prev = hull_points[(i - 1) % n]
        p_curr = hull_points[i]
        p_next = hull_points[(i + 1) % n]

        v_prev = p_prev - p_curr
        v_next = p_next - p_curr

        len_prev = np.linalg.norm(v_prev)
        len_next = np.linalg.norm(v_next)

        r = min(corner_radius, len_prev * 0.4, len_next * 0.4)

        start = p_curr + r * v_prev / len_prev
        end = p_curr + r * v_next / len_next

        if i == 0:
            path_verts.append(start)
            path_codes.append(Path.MOVETO)
        else:
            path_verts.append(start)
            path_codes.append(Path.LINETO)

        path_verts.append(p_curr)  # control point
        path_codes.append(Path.CURVE3)
        path_verts.append(end)
        path_codes.append(Path.CURVE3)

    path_verts.append(path_verts[0])
    path_codes.append(Path.CLOSEPOLY)

    return Path(path_verts, path_codes)

def draw_rounded_hull(ax, nodes, pos, color, alpha=0.25, padding=0.3, corner_radius=0.15, extra_padding_nodes=None, extra_padding=0.0):
    if len(nodes) < 3:
        return

    if extra_padding_nodes is None:
        extra_padding_nodes = []

    points = np.array([pos[n] for n in nodes])
    centroid = points.mean(axis=0)

    expanded_points = []
    for i, p in enumerate(points):
        direction = p - centroid
        norm = np.linalg.norm(direction)
        node_padding = padding
        if nodes[i] in extra_padding_nodes:
            node_padding = padding + extra_padding
        if norm > 0:
            expanded = p + node_padding * direction / norm
        else:
            expanded = p
        expanded_points.append(expanded)

    expanded_points = np.array(expanded_points)

    hull = ConvexHull(expanded_points)
    hull_points = expanded_points[hull.vertices]

    path = rounded_hull_path(hull_points, corner_radius=corner_radius)

    patch = PathPatch(path, facecolor=color, edgecolor=color,
                      alpha=alpha, linewidth=1.5)
    ax.add_patch(patch)

    patch_border = PathPatch(path, facecolor='none', edgecolor=color,
                             alpha=0.8, linewidth=2)
    ax.add_patch(patch_border)

def draw_singleton_clique(ax, node, pos, color, radius=0.35):
    x, y = pos[node]
    circle = Circle((x, y), radius, facecolor=color, edgecolor=color,
                    alpha=0.25, linewidth=1.5)
    ax.add_patch(circle)
    circle_border = Circle((x, y), radius, facecolor='none', edgecolor=color,
                           alpha=0.8, linewidth=2)
    ax.add_patch(circle_border)

def draw_graph_base(ax, G, pos, core_nodes, groups, pendant_nodes):
    core_edges = [(i, j) for i in core_nodes for j in core_nodes if i < j]
    nx.draw_networkx_edges(G, pos, edgelist=core_edges, ax=ax,
                           width=0.3, alpha=0.2, edge_color='gray')

    pendant_edges = []
    for p_idx, pendant in enumerate(pendant_nodes):
        for node in groups[p_idx]:
            pendant_edges.append((pendant, node))
    nx.draw_networkx_edges(G, pos, edgelist=pendant_edges, ax=ax,
                           width=1.2, alpha=0.6, edge_color='darkgray')

    nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, ax=ax,
                           node_color='lightblue', node_size=25,
                           edgecolors='black', linewidths=0.4)

    nx.draw_networkx_nodes(G, pos, nodelist=pendant_nodes, ax=ax,
                           node_color='lightblue', node_size=25,
                           edgecolors='black', linewidths=0.4)

# ============ Main figure ============

fig_width_cm = 8.5
fig_height_cm = 8.0
fig, ax = plt.subplots(1, 1, figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54))

plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

G, core_nodes, groups, pendant_nodes = build_graph()

scale = 0.38

offset_left = (-2.0, -2.0)
pos_left, groups_left, pendant_left = build_positions(scale=scale, offset=offset_left)

offset_right = (2.0, -2.0)
pos_right, groups_right, pendant_right = build_positions(scale=scale, offset=offset_right)

offset_top = (0, 2.0)
pos_top, groups_top, pendant_top = build_positions(scale=scale, offset=offset_top)

clique_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
k25_color = '#e41a1c'
k6_colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
k6_colors_light = ['#a8cee4', '#b2df8a', '#cab2d6', '#fdbf6f']

# ============ Draw bottom left: natural covering ============
for p_idx in range(4):
    k6_nodes = groups[p_idx] + [pendant_nodes[p_idx]]
    draw_rounded_hull(ax, k6_nodes, pos_left, k6_colors[p_idx], alpha=0.2, padding=0.20, corner_radius=0.2,
                      extra_padding_nodes=[pendant_nodes[p_idx]], extra_padding=0.20)
draw_rounded_hull(ax, groups[4], pos_left, k25_color, alpha=0.2, padding=0.20, corner_radius=0.2)
draw_graph_base(ax, G, pos_left, core_nodes, groups, pendant_nodes)

# ============ Draw bottom right: K25 + singletons ============
draw_rounded_hull(ax, core_nodes, pos_right, k25_color, alpha=0.12, padding=0.20, corner_radius=0.2)
singleton_colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for p_idx, pendant in enumerate(pendant_nodes):
    draw_singleton_clique(ax, pendant, pos_right, singleton_colors[p_idx], radius=0.18)
draw_graph_base(ax, G, pos_right, core_nodes, groups, pendant_nodes)

# ============ Draw top center: overlapping ============
draw_rounded_hull(ax, core_nodes, pos_top, k25_color, alpha=0.10, padding=0.20, corner_radius=0.2)
for p_idx in range(4):
    k6_nodes = groups[p_idx] + [pendant_nodes[p_idx]]
    draw_rounded_hull(ax, k6_nodes, pos_top, k6_colors[p_idx], alpha=0.18, padding=0.20, corner_radius=0.2,
                      extra_padding_nodes=[pendant_nodes[p_idx]], extra_padding=0.20)
draw_graph_base(ax, G, pos_top, core_nodes, groups, pendant_nodes)

# ============ Draw arrows (dainty Hasse-style) ============
arrow_style = "->"
arrow_color = 'black'

left_center = np.array(offset_left)
top_center = np.array(offset_top)
left_dir = top_center - left_center
left_dir_norm = left_dir / np.linalg.norm(left_dir)
arrow_left_start = left_center + 1.5 * left_dir_norm
arrow_left_end = top_center - 1.5 * left_dir_norm

arrow_left = FancyArrowPatch(
    posA=tuple(arrow_left_start),
    posB=tuple(arrow_left_end),
    connectionstyle="arc3,rad=0.0",
    arrowstyle=arrow_style,
    color=arrow_color,
    alpha=0.8,
    linewidth=1.0,
    mutation_scale=10
)
ax.add_patch(arrow_left)

right_center = np.array(offset_right)
right_dir = top_center - right_center
right_dir_norm = right_dir / np.linalg.norm(right_dir)
arrow_right_start = right_center + 1.5 * right_dir_norm
arrow_right_end = top_center - 1.5 * right_dir_norm

arrow_right = FancyArrowPatch(
    posA=tuple(arrow_right_start),
    posB=tuple(arrow_right_end),
    connectionstyle="arc3,rad=0.0",
    arrowstyle=arrow_style,
    color=arrow_color,
    alpha=0.8,
    linewidth=1.0,
    mutation_scale=10
)
ax.add_patch(arrow_right)

# ============ Add labels (matching pendant label style) ============
label_fontsize = 12

ax.text(offset_left[0] - 1.0, offset_left[1] + 1.3, r"$\mathcal{G}$",
        fontsize=label_fontsize, ha='center', va='center')

ax.text(offset_right[0] + 1.5, offset_right[1] + 1.3, r"$\mathcal{G}'$",
        fontsize=label_fontsize, ha='center', va='center')

ax.text(offset_top[0] - 1.0, offset_top[1] + 1.3, r"$\mathcal{R}$",
        fontsize=label_fontsize, ha='center', va='center')

ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()

plt.savefig('../output/partial_order.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('../output/partial_order.png',
            format='png', bbox_inches='tight', dpi=300)

print("Partial order diagram saved to partial_order.pdf and partial_order.png")
