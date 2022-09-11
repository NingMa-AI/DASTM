from ms_g3d.msg3d import MS_G3D_Model
from AGCN.agcn import AGCN_Model

backbone = "ms_g3d"
dataset = "ntu_rgb_d"

if backbone == 'ms_g3d':
    graph = 'ms_g3d_graph.ntu_rgb_d.AdjMatrixGraph'
    model = MS_G3D_Model(
        num_class=0,
        num_point=node,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph=graph,
    )
    in_channel = 192
elif backbone == '2s_AGCN':
    graph = 'AGCN_gcn_graph.ntu_rgb_d.Graph'
    model = AGCN_Model(
        num_class=60,
        num_point=node,
        num_person=2,
        graph=grpah,
        graph_args={'labeling_mode': 'spatial'}
    )
    in_channel = 256
