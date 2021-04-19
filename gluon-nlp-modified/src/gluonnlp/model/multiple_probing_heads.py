
__all__ = ['MultiheadProbing1D', 'MultiheadProbing2D', 'MultiheadProbing1DGated']

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

def _get_multi_probing_heads(units, num_heads, mph_type, probing_head, dropout, dim_hidden_mlps, prefix):
    if mph_type == 'mph1d':
        return MultiheadProbing1D(units, num_heads, probing_head, dropout, dim_hidden_mlps, prefix)
    elif mph_type == 'mph2d':
        num_features = 12
        return MultiheadProbing2D(units, num_heads, probing_head, dropout, dim_hidden_mlps,
                                    num_features, prefix)
    elif mph_type == 'mph1d_gated':
        return MultiheadProbing1DGated(units, num_heads, probing_head, dropout, dim_hidden_mlps, prefix)
    elif mph_type == 'mph1d_gated2':
        return MultiheadProbing1DGated2(units, num_heads, probing_head, dropout, dim_hidden_mlps, prefix)
    else:
        raise NotImplementedError


def _get_head_cell(units, probing_head, dropout, dim_hidden_mlps, prefix=None):
    if probing_head == 'linear_head':
        return LinearHeadCell(units=units, dim_hidden_mlps=dim_hidden_mlps, dropout=dropout)

    elif probing_head == 'sd_attn_head': # scaled dot attention head (units, units, units)
        return SDAttnHeadCell(query_units=units, key_units=units, value_units=units,
                                        scaled=True, dropout=dropout)
    elif probing_head == 'sd_attn_head1': # scaled dot attention head (768, 768, units)
        return SDAttnHeadCell(query_units=768, key_units=768, value_units=units,
                                        scaled=True, dropout=dropout)
    elif probing_head == 'ml_sd_attn_head1': # scaled dot attention head (768, 768, units)
        return MLSDAttnHeadCell(query_units=768, key_units=768, value_units=units,
                                        scaled=True, dropout=dropout, dim_hidden_mlps=dim_hidden_mlps)
    elif probing_head == 'sd_attn_head2': # scaled dot attention head (768, 768, 768)->(1, units)
        return SDAttnHeadCellLinear(query_units=768, key_units=768,
                                                value_units=768, units=units, scaled=True, dropout=dropout)
    elif probing_head == 'sd_attn_head3': # scaled dot attention head (hidden_size, hidden_size, units)
        return SDAttnHeadCell(query_units=4096, key_units=4096, value_units=units,
                                        scaled=True, dropout=dropout)
    elif probing_head == 'sd_attn_head_raw':
        return SDAttnHeadCellRaw(query_units=units, key_units=units,
                                                value_units=units, scaled=True, dropout=dropout)
    elif probing_head == 'sd_attn_head_res':
        return SDAttnHeadCellResidual(query_units=768, key_units=768,
                                                value_units=units, scaled=True, dropout=dropout)
    else:
        raise NotImplementedError

class MultiheadProbing2D(HybridBlock):
    def __init__(self, units, num_heads, probing_head, dropout, dim_hidden_mlps, num_features, prefix=None):

        super(MultiheadProbing2D, self).__init__(prefix=prefix)

        assert num_features * num_heads * units == 768

        self._num_heads = num_heads
        self._num_features = num_features

        with self.name_scope():
            self.multi_probing_heads = nn.HybridSequential(prefix=prefix)
            for i in range(num_heads):
                head_cell = _get_head_cell(units, probing_head, dropout, dim_hidden_mlps, prefix)
                self.multi_probing_heads.add(head_cell)

            self.layer_norm = nn.LayerNorm(prefix=prefix)
            self.proj_flatten = nn.Dense(units=768, flatten=True)

            self.proj_linear_layers = nn.HybridSequential(prefix=prefix) # probing hidden layer
            for dim_mlp in dim_hidden_mlps:
                self.proj_linear_layers.add(nn.Dense(units=dim_mlp))

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(MultiheadProbing2D, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        inputs = inputs.reshape(shape=(0, self._num_features, -1))
        outputs = None
        for i, probing_head in enumerate(self.multi_probing_heads):
            output = probing_head(inputs)
            if i == 0:
                outputs = output
            else:
                outputs = mx.symbol.Concat(outputs, output, dim=1)

        unnormed_outputs = outputs
        outputs = self.proj_flatten(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.proj_linear_layers(outputs)
        return outputs, unnormed_outputs


class MultiheadProbing1D(HybridBlock):
    def __init__(self, units, num_heads, probing_head, dropout, dim_hidden_mlps, prefix=None):

        super(MultiheadProbing1D, self).__init__(prefix=prefix)

        with self.name_scope():
            self.multi_probing_heads = nn.HybridSequential(prefix=prefix)
            for i in range(num_heads):
                head_cell = _get_head_cell(units, probing_head, dropout, dim_hidden_mlps, prefix)
                self.multi_probing_heads.add(head_cell)

            self.layer_norm = nn.LayerNorm(prefix=prefix)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(MultiheadProbing1D, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        outputs = None
        for i, probing_head in enumerate(self.multi_probing_heads):
            output = probing_head(inputs)
            output = output.reshape(shape=(0, -1))
            if i == 0:
                outputs = output
            else:
                outputs = mx.symbol.Concat(outputs, output, dim=1)

        unnormed_outputs = outputs
        outputs = self.layer_norm(outputs)
        return outputs, unnormed_outputs


class LinearHeadCell(HybridBlock):
    def __init__(self, units, dim_hidden_mlps, prefix=None, dropout=0.0):
        super(LinearHeadCell, self).__init__(prefix=prefix)

        with self.name_scope():
            self.hidden_layers = nn.HybridSequential(prefix=prefix) # probing hidden layer

            for dim_mlp in dim_hidden_mlps:
                self.hidden_layers.add(nn.Dense(units=dim_mlp))

            self.hidden_layers.add(nn.Dense(units=units))


    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(LinearHeadCell, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        return self.hidden_layers(inputs)


class SDAttnHeadCell(HybridBlock):
    def __init__(self, query_units, key_units, value_units, scaled=True, dropout=0.0, prefix=None):
        super(SDAttnHeadCell, self).__init__(prefix=prefix)
        self._scaled = scaled

        assert query_units == key_units

        with self.name_scope():
            self.proj_query = nn.Dense(units=query_units)
            self.proj_key = nn.Dense(units=key_units)
            self.proj_value = nn.Dense(units=value_units)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(SDAttnHeadCell, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        query = self.proj_query(inputs).reshape(shape=(0, 1, -1))
        key = self.proj_key(inputs).reshape(shape=(0, 1, -1))
        value = self.proj_value(inputs).reshape(shape=(0, 1, -1))

        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True)
        att_weights = F.softmax(att_score, axis=-1)
        outputs = F.batch_dot(att_weights, value)
        #att_weights = self._dropout_layer(att_weights)

        return outputs


class MLSDAttnHeadCell(HybridBlock):
    def __init__(self, query_units, key_units, value_units, scaled=True, dropout=0.0,
                        dim_hidden_mlps=[], prefix=None):
        super(MLSDAttnHeadCell, self).__init__(prefix=prefix)
        self._scaled = scaled

        assert query_units == key_units

        with self.name_scope():
            self.proj_query_layers = nn.HybridSequential(prefix=prefix)
            self.proj_key_layers = nn.HybridSequential(prefix=prefix)
            self.proj_value_layers = nn.HybridSequential(prefix=prefix)
            for dim_mlp in dim_hidden_mlps:
                proj_query = nn.Dense(units=query_units)
                proj_key = nn.Dense(units=key_units)
                proj_value = nn.Dense(units=dim_mlp)
                query_units = dim_mlp
                key_units = dim_mlp
                self.proj_query_layers.add(proj_query)
                self.proj_key_layers.add(proj_key)
                self.proj_value_layers.add(proj_value)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(MLSDAttnHeadCell, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        for proj_query, proj_key, proj_value in zip(self.proj_query_layers,
                                                    self.proj_key_layers, self.proj_value_layers):
            query = proj_query(inputs).reshape(shape=(0, 1, -1))
            key = proj_key(inputs).reshape(shape=(0, 1, -1))
            value = proj_value(inputs).reshape(shape=(0, 1, -1))

            if self._scaled:
                query = F.contrib.div_sqrt_dim(query)

            att_score = F.batch_dot(query, key, transpose_b=True)
            att_weights = F.softmax(att_score, axis=-1)
            outputs = F.batch_dot(att_weights, value)
            inputs = outputs

        return outputs



class SDAttnHeadCellLinear(HybridBlock):
    def __init__(self, query_units, key_units, value_units, units, scaled=True, dropout=0.0, prefix=None):
        super(SDAttnHeadCellLinear, self).__init__(prefix=prefix)
        self._scaled = scaled

        assert query_units == key_units

        with self.name_scope():
            self.proj_query = nn.Dense(units=query_units)
            self.proj_key = nn.Dense(units=key_units)
            self.proj_value = nn.Dense(units=value_units)
            self.proj_linear = nn.Dense(units=units)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(SDAttnHeadCellLinear, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        query = self.proj_query(inputs).reshape(shape=(0, 1, -1))
        key = self.proj_key(inputs).reshape(shape=(0, 1, -1))
        value = self.proj_value(inputs).reshape(shape=(0, 1, -1))

        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True)
        att_weights = F.softmax(att_score, axis=-1)
        outputs = F.batch_dot(att_weights, value)
        outputs = self.proj_linear(outputs)
        return outputs


class SDAttnHeadCellRaw(HybridBlock):
    def __init__(self, query_units, key_units, value_units, scaled=True, dropout=0.0, prefix=None):
        super(SDAttnHeadCellRaw, self).__init__(prefix=prefix)
        self._scaled = scaled

        assert query_units == key_units

        with self.name_scope():
            self.proj_query = nn.Dense(units=query_units)
            self.proj_key = nn.Dense(units=key_units)
            self.proj_value = nn.Dense(units=value_units)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(SDAttnHeadCellRaw, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        query = self.proj_query(inputs)
        key = self.proj_key(inputs)
        value = self.proj_value(inputs)

        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True)
        att_weights = F.softmax(att_score, axis=-1)
        outputs = F.batch_dot(att_weights, value)
        #att_weights = self._dropout_layer(att_weights)

        return outputs


class SDAttnHeadCellResidual(HybridBlock):
    def __init__(self, query_units, key_units, value_units, scaled=True, dropout=0.0, prefix=None):
        super(SDAttnHeadCellResidual, self).__init__(prefix=prefix)
        self._scaled = scaled

        assert query_units == key_units

        with self.name_scope():
            self.proj_query = nn.Dense(units=query_units)
            self.proj_key = nn.Dense(units=key_units)
            self.proj_value = nn.Dense(units=value_units)
            self.proj_res = nn.Dense(units=value_units)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(SDAttnHeadCellResidual, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        query = self.proj_query(inputs).reshape(shape=(0, 1, -1))
        key = self.proj_key(inputs).reshape(shape=(0, 1, -1))
        value = self.proj_value(inputs).reshape(shape=(0, 1, -1))

        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True)
        att_weights = F.softmax(att_score, axis=-1)
        outputs = F.batch_dot(att_weights, value)
        #att_weights = self._dropout_layer(att_weights)
        outputs = outputs + self.proj_res(inputs).reshape(shape=(0, 1, -1))

        return outputs


class MultiheadProbing1DGated(HybridBlock):
    def __init__(self, units, num_heads, probing_head, dropout, dim_hidden_mlps, prefix=None):

        super(MultiheadProbing1DGated, self).__init__(prefix=prefix)

        with self.name_scope():
            self.multi_probing_heads = nn.HybridSequential(prefix=prefix)
            self.f_gates = nn.HybridSequential(prefix=prefix)
            self.i_gates = nn.HybridSequential(prefix=prefix)
            for i in range(num_heads):
                head_cell = _get_head_cell(units, probing_head, dropout, dim_hidden_mlps, prefix)
                self.multi_probing_heads.add(head_cell)

                self.f_gates.add(nn.Dense(units=units, flatten=False, activation='sigmoid',
                              prefix=prefix))
                self.i_gates.add(nn.Dense(units=units, flatten=False, activation='tanh',
                              prefix=prefix))

            self.layer_norm = nn.LayerNorm(prefix=prefix)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(MultiheadProbing1DGated, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        outputs = None
        for i, probing_head in enumerate(self.multi_probing_heads):
            output = probing_head(inputs)
            output = output.reshape(shape=(0, -1))

            output = F.broadcast_mul(self.f_gates[i](output), self.i_gates[i](output))
            if i == 0:
                outputs = output
            else:
                #outputs = F.broadcast_add(outputs, output)
                outputs = mx.symbol.Concat(outputs, output, dim=1)

        #unnormed_outputs = outputs
        outputs = self.layer_norm(outputs)
        return outputs, outputs


class MultiheadProbing1DGated2(HybridBlock):
    def __init__(self, units, num_heads, probing_head, dropout, dim_hidden_mlps, prefix=None):

        super(MultiheadProbing1DGated2, self).__init__(prefix=prefix)

        self._num_heads = num_heads
        with self.name_scope():
            self.f_gates = nn.HybridSequential(prefix=prefix)
            self.i_gates = nn.HybridSequential(prefix=prefix)
            for i in range(num_heads):
                self.f_gates.add(nn.Dense(units=units, flatten=False, activation='sigmoid',
                              prefix=prefix))
                self.i_gates.add(nn.Dense(units=units, flatten=False, activation='tanh',
                              prefix=prefix))

            self.layer_norm = nn.LayerNorm(prefix=prefix)

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(MultiheadProbing1DGated2, self).__call__(inputs)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        outputs = None
        for i in range(self._num_heads):
            output = F.broadcast_mul(self.f_gates[i](inputs), self.i_gates[i](inputs))
            if i == 0:
                outputs = output
            else:
                #outputs = F.broadcast_add(outputs, output)
                outputs = mx.symbol.Concat(outputs, output, dim=1)

        #unnormed_outputs = outputs
        outputs = self.layer_norm(outputs)
        return outputs, outputs

