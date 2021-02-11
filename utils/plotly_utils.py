# https://github.com/YurongYou/point_cloud_utilities

import plotly.graph_objects as go
import plotly.express as px
from .kitti_util import compute_box_3d
from PIL import Image
import numpy as np

ptc_layout_config={
    'title': {
        'text': 'LiDAR',
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'xanchor': 'left',
        'yanchor': 'top'},
    'paper_bgcolor': 'rgb(0,0,0)',
    'width' : 900,
    'height' : 600,
    'margin' : {
        'l': 20,
        'r': 20,
        'b': 20,
        't': 20
    },
    'legend': {
        'font':{
            'size':20,
            'color': 'rgb(150,150,150)',
        },
        'itemsizing': 'constant'
    },
    "hoverlabel": {
        "namelength": -1,
    },
    'showlegend': False,
    'scene': {
          'aspectmode': 'manual',
          'aspectratio': {'x': 0.75, 'y': 0.25, 'z': 0.05},
          'camera': {'eye': {'x': 0, 'y': 0, 'z': 0.5}},
          'xaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-150, 150],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'x:'},
          'yaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-50, 50],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'y:'},
          'zaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-10, 10],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'z:'}},
}

def showimg(img, labels=None, predictions=None, classes=['Car', 'Truck', 'Van'], scale_factor=0.7):
    # Create figure
    fig = go.Figure()

    # Constants
    img_width = img.shape[1]
    img_height = img.shape[0]

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=Image.fromarray(img[:, :, [2,1,0]]))
    )

    if labels is not None:
        for label in labels:
            if label.cls_type in classes:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=label.xmin * scale_factor,
                    y0=(img_height-label.ymin) * scale_factor,
                    x1=label.xmax * scale_factor,
                    y1=(img_height-label.ymax) * scale_factor,
                    line=dict(
                        color="LightGreen", width=2
                    ),
                )

    if predictions is not None:
        for label in predictions:
            if label.cls_type in classes:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=label.xmin * scale_factor,
                    y0=(img_height-label.ymin) * scale_factor,
                    x1=label.xmax * scale_factor,
                    y1=(img_height-label.ymax) * scale_factor,
                    line=dict(
                        color="Red", width=2
                    ),
                    name=label.cls_type
                )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plot.ly/python/configuration-options/
    fig.show(config={'doubleClick': 'reset'})
    return fig

def get_linemarks(obj, calib):
    _, corners = compute_box_3d(obj, calib.P)
    corners = calib.project_rect_to_velo(corners)
    mid_front = (corners[0] + corners[1]) / 2
    mid_left = (corners[0] + corners[3]) / 2
    mid_right = (corners[1] + corners[2]) / 2
    corners = np.vstack(
        (corners, np.vstack([mid_front, mid_left, mid_right])))
    idx = [0,8,9,10,8,1,2,3,0,4,5,1,5,6,2,6,7,3,7,4]
    return corners[idx, :]

def get_bbox(obj, calib, name='bbox' , color='yellow'):
    markers = get_linemarks(obj, calib)
    return go.Scatter3d(
        mode='lines',
        x=markers[:, 0],
        y=markers[:, 1],
        z=markers[:, 2],
        line=dict(color=color, width=3),
        name=name)

def get_lidar(ptc, name='LiDAR', size=0.8):
    return [go.Scatter3d(
        x=ptc[:,0],
        y=ptc[:,1],
        z=ptc[:,2],
        mode='markers',
        marker_size=size,
        name=name)]

def showvelo(lidar, calib, labels=None, predictions=None, classes=('Car', 'Truck', 'Van'), size=0.8):
    gt_bboxes = [] if labels is None else [get_bbox(obj, calib, name='gt_bbox', color='lightgreen') for obj in labels if obj.cls_type in classes]
    pred_bboxes = [] if predictions is None else [get_bbox(obj, calib, name='pred_bbox', color='red') for obj in predictions if obj.cls_type in classes]
    fig = go.Figure(data=get_lidar(lidar, size=size) +
                    gt_bboxes + pred_bboxes, layout=ptc_layout_config)
    fig.show()
    return fig

def showvelo2(lidar_common, lidar_before, lidar_after, calib, labels_before, labels_after, classes=('Car', 'Van'), size=0.8):
    bboxes_before = [get_bbox(obj, calib, name='bbox_before', color='red') for obj in labels_before if obj.cls_type in classes]
    bboxes_after = [get_bbox(obj, calib, name='bbox_after', color='lightgreen') for obj in labels_after if obj.cls_type in classes]
    fig = go.Figure(data=get_lidar(lidar_common, name="Environment", size=size) +
                         get_lidar(lidar_before, name="Object before stat_norm", size=size) +
                         get_lidar(lidar_after, name="Object after stat_norm", size=size) +
                         bboxes_before + bboxes_after, layout=ptc_layout_config)
    fig.show()
    return fig
