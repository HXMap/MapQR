import torch
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import \
    MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner.base_module import BaseModule
import warnings
from mmcv.cnn import xavier_init, constant_init
import torch.nn as nn
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
import copy
import math

def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False,
                 query_pos_embedding='none',
                 num_pts_per_vec=20,
                 pt_dim=2,
                 **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.query_pos_embedding = query_pos_embedding
        self.num_pts_per_vec = num_pts_per_vec
        self.pt_dim = pt_dim
        if query_pos_embedding == 'instance':
            def _get_clones(module, N):
                return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            pt_pos_query_proj = nn.Linear(self.embed_dims, self.embed_dims)
            self.pt_pos_query_projs = _get_clones(pt_pos_query_proj, self.num_layers)

    def init_weights(self):
        if self.query_pos_embedding == 'instance':
            for m in self.pt_pos_query_projs:
                xavier_init(m, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        bs = reference_points.shape[0]
        for lid, layer in enumerate(self.layers):
            if self.query_pos_embedding == 'instance':
                reference_points_reshape = reference_points.view(bs, -1, self.num_pts_per_vec, self.pt_dim)
                reference_points_reshape = reference_points_reshape.view(bs, -1, self.pt_dim)
                query_sine_embed = gen_sineembed_for_position(reference_points_reshape[..., :2])
                query_sine_embed = query_sine_embed.view(bs, -1, self.num_pts_per_vec, self.embed_dims)
                point_query_pos = self.pt_pos_query_projs[lid](query_sine_embed)
                query_pos_lid = None
                reference_points_input = reference_points_reshape[..., :2].unsqueeze(2)
            else:
                point_query_pos = None
                reference_points_input = reference_points[..., :2].unsqueeze(2)
                query_pos_lid = query_pos
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos_lid,
                pt_query_pos=point_query_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class InstancePointAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 num_pts_per_vec=20,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dims*num_pts_per_vec, embed_dims*num_pts_per_vec//2),
            nn.ReLU(),
            nn.Linear(embed_dims*num_pts_per_vec//2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                pt_query_pos=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        if pt_query_pos is not None:
            bs, num_iq, num_pt, _ = pt_query_pos.shape
            query = query.unsqueeze(2) + pt_query_pos
            query = query.reshape(bs, num_iq * num_pt, -1)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets \
                / (offset_normalizer[None, None, None, :, None, :] * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.view(bs, num_iq, num_pt, -1)
        output = output.flatten(2)
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@TRANSFORMER_LAYER.register_module()
class CustomTransformerLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(CustomTransformerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=kwargs['self_attn_mask'],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER.register_module()
class DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 num_vec=50,
                 num_pts_per_vec=20,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DecoupledDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 8
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        # 
        num_vec = kwargs['num_vec']
        num_pts_per_vec = kwargs['num_pts_per_vec']
        for layer in self.operation_order:
            if layer == 'self_attn':
                # import ipdb;ipdb.set_trace()
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec,n_batch,n_dim).flatten(1,2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec,n_batch,n_dim).flatten(1,2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=kwargs['self_attn_mask'],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    # import ipdb;ipdb.set_trace()
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0,1)
                    attn_index += 1
                    identity = query
                else:
                    # import ipdb;ipdb.set_trace()
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec,n_batch,n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec,n_batch,n_dim).permute(1,0,2,3).contiguous().flatten(1,2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs)
                    # import ipdb;ipdb.set_trace()
                    query = query.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    query_pos = query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
                    attn_index += 1
                    identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

