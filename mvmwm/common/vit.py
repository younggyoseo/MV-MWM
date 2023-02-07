import tensorflow as tf
import tfimm.layers as tfml


class ViTBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: str,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        norm_layer = tfml.norm_layer_factory(norm_layer)

        self.norm1 = norm_layer(name="norm1")
        self.attn = ViTMultiHeadAttention(
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            name="attn",
        )
        self.drop_path = tfml.DropPath(drop_prob=drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp = tfml.MLP(
            hidden_dim=int(embed_dim * mlp_ratio),
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            act_layer=act_layer,
            name="mlp",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = self.attn(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


class ViTMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        head_dim = embed_dim // nb_heads
        self.scale = head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(
            units=3 * embed_dim, use_bias=qkv_bias, name="qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        # B (batch size), N (sequence length), D (embedding dimension),
        # H (number of heads)
        batch_size, seq_length = tf.unstack(tf.shape(x)[:2])
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = tf.reshape(qkv, (batch_size, seq_length, 3, self.nb_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.scale * tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, N, N)
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, N, N)
        attn = self.attn_drop(attn, training=training)

        x = tf.linalg.matmul(attn, v)  # (B, H, N, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, H, D/H)
        x = tf.reshape(x, (batch_size, seq_length, -1))  # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x
