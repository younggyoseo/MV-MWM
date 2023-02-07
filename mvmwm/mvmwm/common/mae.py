import numpy as np
import pickle
import tfimm

from tfimm.layers import PatchEmbeddings
from tfimm.layers.factory import norm_layer_factory

import common

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers as tfkl
from tensorflow.keras import mixed_precision as prec


class MaskedViTEncoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        ncams,
        patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        early_conv=False,
        view_masking=0,
        additional_cams_list=[],
        viewpoint_pos_emb=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.ncams = ncams
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.early_conv = early_conv
        self.view_masking = view_masking
        self.additional_cams_list = additional_cams_list
        self.viewpoint_pos_emb = viewpoint_pos_emb
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        each_view_w = self.img_w_size // self.ncams
        each_view_h = self.img_h_size

        self.cls_pos_embed = tf.constant(
            np.zeros([1, self.embed_dim]), name="cls_pos_embed", dtype=tf.float32
        )
        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="front_pos_embed",
            dtype=tf.float32,
        )

    def random_view_masking(self, x, mask_ratio, T, ncams, view_masking=0):
        N, L, D = x.shape

        if view_masking == 0 or ncams == 1:
            len_keep = int(L * (1 - mask_ratio))
            noise = tf.random.uniform([N, L], 0.0, 1.0)
            if mask_ratio == 0.0:
                noise = tf.sort(noise)
        else:
            # We should adjust the effective masking ratio with view-masking
            # For instance, when we use view-masking of 1 with 2 viewpoints
            # 50% will be already masked with view-masking,
            # 40% remaining masks will be sampled from 50%,
            # so 90% masking ratio becomes effectively 80%
            # so we have to adjust the effective masking ratio
            if mask_ratio == 0.0:
                noise = tf.random.uniform([N, L], 0.0, 1.0)
                noise = tf.sort(noise)
                len_keep = L
            else:
                _mask_ratio = mask_ratio + (1.0 - mask_ratio) * (view_masking / ncams)
                len_keep = int(L * (1 - _mask_ratio))
                img_x = x
                img_xs = tf.split(img_x, T, axis=1)
                noises = []
                for t in range(T):
                    _img = img_xs[t]
                    _img_size = _img.shape[1] // ncams

                    t_noises = []
                    conds = []
                    for i in range(ncams):
                        uniform_noise = tf.random.uniform([N, _img_size], 0.0, 1.0)
                        view_noise = (
                            tf.ones([N, _img_size], dtype=uniform_noise.dtype) + 1e-2
                        )

                        cond_noise = tf.random.uniform([N, 1])
                        cond = cond_noise > 0.5

                        if i != 0:
                            num_masked_views = tf.reduce_sum(
                                tf.concat(conds, 1), 1, keepdims=True
                            )
                            all_views_masked = num_masked_views == self.view_masking
                            # if specified number of views are already masked,
                            # let's set view_mask = False
                            # e.g., view_masked = True if N views are already masked
                            # Then cond (=view masking) will be set to False
                            cond = ~all_views_masked & cond

                        not_cond = ~cond

                        noise = (
                            tf.cast(cond, uniform_noise.dtype) * view_noise
                            + tf.cast(not_cond, uniform_noise.dtype) * uniform_noise
                        )

                        t_noises.append(noise)
                        conds.append(tf.cast(cond, uniform_noise.dtype))
                    noise = tf.concat(t_noises, 1)
                    noises.append(noise)
                noise = tf.concat(noises, axis=1)

        # sort noise for each sample
        # keep small, remove large
        ids_shuffle = tf.argsort(noise, axis=1)
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # trick for tensorflow-gather
        row_ids = tf.ones_like(ids_shuffle) * tf.expand_dims(tf.range(N), 1)
        _ids_shuffle = tf.stack([row_ids, ids_shuffle], -1)  # [N, L, 2]
        _ids_restore = tf.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        # keep the first subset
        ids_keep = _ids_shuffle[:, :len_keep]
        x_masked = tf.gather_nd(x, ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = tf.concat([tf.zeros([N, len_keep]), tf.ones([N, L - len_keep])], axis=1)
        # unshuffle to get ther binary mask
        mask = tf.gather_nd(mask, _ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x,
        mask_ratio,
        viewpoint,
        T,
    ):
        # embed patches
        x = self._cast(x)
        batch_size = tf.shape(x)[0]

        w = x.shape[-2]
        _ncams = 1 if w == (self.img_w_size // self.ncams) else self.ncams

        if self.early_conv:
            x = self.forward_early_conv(x, _ncams)
        else:
            x = self.forward_patch_embedding(x, _ncams)

        # Reshape to sequential shape
        x = tf.reshape(x, [batch_size // T, T * x.shape[1], x.shape[2]])

        pos_embed = self.get_pos_embed(
            x,
            T,
            viewpoint,
        )
        if self.viewpoint_pos_emb:
            unused_viewpoint = [
                vp for vp in self.additional_cams_list if vp not in viewpoint
            ]
            for vp in unused_viewpoint:
                unused_pos_embed = self.get_pos_embed(x, T, [vp] * len(viewpoint))
                pos_embed += unused_pos_embed * 0.0

        ncams = len(viewpoint)
        # add pos embed w/o cls token
        x = x + self._cast(pos_embed)

        # masking: length -> length * mask_ratio
        masking_fn = self.random_view_masking
        x, mask, ids_restore = masking_fn(x, mask_ratio, T, ncams, self.view_masking)

        # append class token
        cls_token = self.get(
            "cls_token", common.mae_utils.Token, "cls", self.embed_dim
        )(x)
        cls_token = cls_token + self.cls_pos_embed
        cls_tokens = tf.repeat(cls_token, repeats=x.shape[0], axis=0)
        x = tf.concat([self._cast(cls_tokens), x], axis=1)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_encoder_block_{j}",
                common.ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_encoder_norm", norm_layer_factory(self.norm_layer))(x)

        return x, mask, ids_restore

    def forward_early_conv(self, x, ncams):
        # x : [B, H, W, C]
        B, H, W, C = x.shape

        if ncams != 1:
            # split across W dimension and concat to B
            x = tf.split(x, ncams, axis=2)
            x = tf.concat(x, axis=0)

        nconvs = int(np.log2(self.patch_size))
        for i in range(nconvs):
            depth = self.embed_dim // (2 ** (nconvs - i))
            x = self.get(
                f"early_conv_{i}",
                tfkl.Conv2D,
                depth,
                4,
                2,
                padding="SAME",
            )(x)
            x = tf.nn.relu(x)
        x = self.get("early_conv_proj", tfkl.Conv2D, self.embed_dim, 1, 1)(x)
        x = tf.reshape(x, [x.shape[0], -1, self.embed_dim])

        if ncams != 1:
            # split across B dimension and concat to W
            x = tf.split(x, ncams, axis=0)
            x = tf.concat(x, axis=1)
        return x

    def forward_patch_embedding(self, x, ncams):
        if ncams != 1:
            # split across W dimension and concat to B
            x = tf.split(x, ncams, axis=2)
            x = tf.concat(x, axis=0)

        x = self.get(
            "encoder_patch_embed",
            PatchEmbeddings,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer="",
        )(x)

        if ncams != 1:
            # split across B dimension and concat to W
            x = tf.split(x, ncams, axis=0)
            x = tf.concat(x, axis=1)
        return x

    def get_pos_embed(self, x, T, viewpoint):
        cams = [vp for vp in viewpoint]
        if self.viewpoint_pos_emb:
            cams = [f"cam_decoder_{c}" for c in cams]
        else:
            cams = [f"cam_decoder_0" for c in cams]

        _pos_embed = []
        for t in range(T):
            for cam, vp in zip(cams, viewpoint):
                cam_pos_embed = self.pos_embed
                cam_token = self.get(
                    f"{cam}_token", common.mae_utils.Token, cam, self.embed_dim
                )(x)

                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                cam_time_token = tf.tile(
                    cam_token + time_token, [1, cam_pos_embed.shape[1], 1]
                )
                _pos_embed.append(cam_pos_embed + cam_time_token)

        img_pos_embed = tf.concat(_pos_embed, axis=1)
        pos_embed = img_pos_embed

        return pos_embed


class MaskedViTDecoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        ncams,
        patch_size,
        in_chans=3,
        embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        reward_pred=True,
        additional_cams_list=[],
        viewpoint_pos_emb=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.ncams = ncams
        self.in_chans = in_chans
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.reward_pred = reward_pred
        self.additional_cams_list = additional_cams_list
        self.viewpoint_pos_emb = viewpoint_pos_emb
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        each_view_w = self.img_w_size // self.ncams
        each_view_h = self.img_h_size

        self.cls_pos_embed = tf.constant(
            np.zeros([1, self.embed_dim]),
            name="cls_pos_embed",
            dtype=tf.float32,
        )
        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="pos_embed",
            dtype=tf.float32,
        )

    def patchify(self, imgs):
        """
        imgs: [N, H, W, 3]
        x: [N, L, patch_size**2 * 3]
        """
        p = self.patch_size
        c = imgs.shape[-1]
        assert imgs.shape[1] % p == 0 and imgs.shape[2] % p == 0

        x = tf.image.extract_patches(
            imgs, [1, p, p, 1], [1, p, p, 1], [1, 1, 1, 1], "VALID"
        )
        x = tf.reshape(x, [imgs.shape[0], -1, p ** 2 * c])

        return x

    def unpatchify(self, x, _ncams=None):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, H, W, 3)
        """
        p = self.patch_size
        c = x.shape[-1] // (p ** 2)
        h = self.img_h_size // p
        w = self.img_w_size // p

        if _ncams is not None:
            w = h * _ncams
        else:
            _ncams = self.ncams

        assert h * w == x.shape[1]

        x_split = tf.split(x, _ncams, axis=1)
        imgs = []
        for _x in x_split:
            _h, _w = h, int(w // _ncams)
            _x = tf.reshape(_x, [_x.shape[0], _h, _w, p, p, c])
            _x = tf.einsum("nhwpqc->nhpwqc", _x)
            _img = tf.reshape(_x, [_x.shape[0], _h * p, _w * p, c])
            imgs.append(_img)
        imgs = tf.concat(imgs, axis=-2)
        return imgs

    def forward_decoder(
        self,
        x,
        ids_restore,
        viewpoint,
        T,
    ):
        # embed tokens
        x = self._cast(x)
        x = self.get(
            "decoder_embed",
            tfkl.Dense,
            self.embed_dim,
        )(x)

        # trick for tensorflow-gather
        N = ids_restore.shape[0]
        row_ids = tf.ones_like(ids_restore) * tf.expand_dims(tf.range(N), 1)
        ids_restore = tf.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        mask_token = self.get(
            "mask_token", common.mae_utils.Token, "mask", self.embed_dim
        )(x)
        mask_tokens = self._cast(
            tf.tile(
                mask_token,
                [x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1],
            )
        )
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        x_ = tf.gather_nd(x_, ids_restore)  # unshuffle

        camera_size = x_.shape[1]

        # append mask token for reward prediction
        # we use same mask token for rew prediction. Maybe try different token?
        if self.reward_pred:
            rew_mask_token = self._cast(
                tf.tile(
                    mask_token,
                    [x.shape[0], T, 1],
                )
            )
            x_ = tf.concat([x_, rew_mask_token], axis=1)

        x = tf.concat([x[:, :1, :], x_], axis=1)  # append cls token

        pos_embed = self.get_pos_embed(x, T, viewpoint)
        if self.viewpoint_pos_emb:
            unused_viewpoint = [
                vp for vp in self.additional_cams_list if vp not in viewpoint
            ]
            for vp in unused_viewpoint:
                unused_pos_embed = self.get_pos_embed(x, T, [vp] * len(viewpoint))
                pos_embed += unused_pos_embed * 0.0

        dec_pos_embed = self._cast(pos_embed)
        x = x + tf.repeat(dec_pos_embed, repeats=x.shape[0], axis=0)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_decoder_block_{j}",
                common.ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_decoder_norm", norm_layer_factory(self.norm_layer))(x)

        dec = self.get(
            "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
        )(x[:, 1 : 1 + camera_size])
        # Revert to batch-wise shape
        dec = tf.reshape(dec, [dec.shape[0] * T, dec.shape[1] // T, dec.shape[2]])

        if self.reward_pred:
            rew = self.get("vit_reward_pred", tfkl.Dense, 1)(x[:, -T:, :])
            rew = tf.reshape(rew, [rew.shape[0] * T, 1, rew.shape[2]])
        else:
            rew = None, None
        return dec, rew

    def get_pos_embed(self, x, T, viewpoint):
        cls_token = self.get(
            "cls_token", common.mae_utils.Token, "cls", self.embed_dim
        )(x)
        cls_pos_embed = cls_token

        cams = [vp for vp in viewpoint]
        if self.viewpoint_pos_emb:
            cams = [f"cam_decoder_{c}" for c in cams]
        else:
            cams = [f"cam_decoder_0" for c in cams]

        _pos_embed = []
        for t in range(T):
            for cam in cams:
                cam_pos_embed = self.pos_embed
                cam_token = self.get(
                    f"{cam}_token", common.mae_utils.Token, cam, self.embed_dim
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                cam_time_token = tf.tile(
                    cam_token + time_token, [1, cam_pos_embed.shape[1], 1]
                )
                _pos_embed.append(cam_pos_embed + cam_time_token)

        img_pos_embed = tf.concat(_pos_embed, axis=1)
        pos_embed = tf.concat([cls_pos_embed, img_pos_embed], axis=1)

        if self.reward_pred:
            _reward_pos_embed = []
            for t in range(T):
                reward_token = self.get(
                    "reward_token",
                    common.mae_utils.Token,
                    "reward",
                    self.embed_dim,
                )(x)
                time_token = self.get(
                    f"time_{t}_token",
                    common.mae_utils.Token,
                    f"time_{t}",
                    self.embed_dim,
                )(x)
                reward_time_token = reward_token + time_token
                _reward_pos_embed.append(reward_time_token)
            reward_pos_embed = tf.concat(_reward_pos_embed, axis=1)
            pos_embed = tf.concat([pos_embed, reward_pos_embed], axis=1)

        return pos_embed

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, H, W, 3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        imgs = tf.cast(imgs, tf.float32)
        pred = tf.cast(pred, tf.float32)
        mask = tf.cast(mask, tf.float32)

        imgs_split = tf.split(imgs, self.ncams, axis=-2)
        target_split = [self.patchify(split) for split in imgs_split]
        target = tf.concat(target_split, axis=1)

        loss = (pred - target) ** 2
        loss = tf.reduce_mean(loss, -1)  # [N, L], mean loss per patch
        loss = loss.mean()
        return loss

    def forward_reward_loss(self, rews, preds):
        rews = tf.cast(rews, tf.float32)
        preds = tf.cast(preds, tf.float32)
        dist = common.SymlogDist(preds, 1, "mean")
        loss = -dist.log_prob(rews)
        return loss.mean()


class ViTEncoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        ncams = int(self.img_w_size // self.img_h_size)
        each_view_w = self.img_w_size // ncams
        each_view_h = self.img_h_size
        self.ncams = ncams

        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="pos_embed",
            dtype=tf.float32,
        )

    def forward_encoder(self, x):
        # embed patches
        x = self._cast(x)
        x = self.get("encoder_embed", tfkl.Dense, self.embed_dim)(x)
        pos_embed = self.get_pos_embed(x)
        x = x + self._cast(pos_embed)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_encoder_block_{j}",
                common.ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_encoder_norm", norm_layer_factory(self.norm_layer))(x)

        return x

    def get_pos_embed(self, x):
        img_pos_embed = tf.concat([self.pos_embed] * self.ncams, axis=1)

        img_tokens = []
        for i in range(self.ncams):
            token = self.get(
                f"token_{i}",
                common.mae_utils.Token,
                f"{i}",
                self.embed_dim,
            )(x)
            token = tf.tile(
                token,
                [1, self.pos_embed.shape[1], 1],
            )
            img_tokens.append(token)
        img_token = tf.concat(img_tokens, axis=1)
        img_pos_embed = img_pos_embed + img_token

        pos_embed = img_pos_embed

        mae_cls_token = self.get(
            "mae_cls_token", common.mae_utils.Token, "mae_cls", self.embed_dim
        )(x)
        mae_cls_pos_embed = mae_cls_token
        pos_embed = tf.concat([pos_embed, mae_cls_pos_embed], axis=1)

        return pos_embed


class ViTDecoder(common.Module):
    def __init__(
        self,
        img_h_size,
        img_w_size,
        patch_size,
        in_chans=3,
        embed_dim=512,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_h_size, self.img_w_size = img_h_size, img_w_size
        self.in_chans = in_chans
        self.num_patches = (img_h_size // patch_size) * (img_w_size // patch_size) + 1
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

        self.construct_pos_embed()

    def construct_pos_embed(self):
        ncams = int(self.img_w_size // self.img_h_size)
        each_view_w = self.img_w_size // ncams
        each_view_h = self.img_h_size
        self.ncams = ncams

        self.pos_embed = tf.constant(
            common.mae_utils.get_2d_sincos_pos_embed(
                self.embed_dim,
                int(each_view_h // self.patch_size),
                int(each_view_w // self.patch_size),
            )[None],
            name="pos_embed",
            dtype=tf.float32,
        )

    def forward_decoder(self, x):
        # embed tokens
        x = self._cast(x)
        x = self.get(
            "decoder_embed",
            tfkl.Dense,
            self.embed_dim,
        )(x)

        mask_token = self.get(
            "mask_token", common.mae_utils.Token, "mask", self.embed_dim
        )(x)
        mask_tokens = self._cast(
            tf.tile(
                mask_token,
                [x.shape[0], self.num_patches, 1],
            )
        )
        x = tf.concat([x[:, :1, :], mask_tokens], axis=1)  # append cls token

        # add pos embed
        decoder_pos_embed = self.get_pos_embed(x)
        x = x + tf.repeat(self._cast(decoder_pos_embed), repeats=x.shape[0], axis=0)

        # apply Transformer blocks
        for j in range(self.depth):
            x = self.get(
                f"vit_decoder_block_{j}",
                common.ViTBlock,
                embed_dim=self.embed_dim,
                nb_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=self.norm_layer,
                act_layer="gelu",
            )(x)
        x = self.get("vit_decoder_norm", norm_layer_factory(self.norm_layer))(x)

        # predictor projection
        x = self.get(
            "vit_decoder_pred", tfkl.Dense, self.patch_size ** 2 * self.in_chans
        )(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def get_pos_embed(self, x):
        cls_token = self.get(
            "cls_token", common.mae_utils.Token, "cls", self.embed_dim
        )(x)
        cls_pos_embed = cls_token

        img_pos_embed = tf.concat([self.pos_embed] * self.ncams, axis=1)

        img_tokens = []
        for i in range(self.ncams):
            token = self.get(
                f"token_{i}",
                common.mae_utils.Token,
                f"{i}",
                self.embed_dim,
            )(x)
            token = tf.tile(
                token,
                [1, self.pos_embed.shape[1], 1],
            )
            img_tokens.append(token)
        img_token = tf.concat(img_tokens, axis=1)
        img_pos_embed = img_pos_embed + img_token

        pos_embed = tf.concat([cls_pos_embed, img_pos_embed], axis=1)

        mae_cls_token = self.get(
            "mae_cls_token", common.mae_utils.Token, "mae_cls", self.embed_dim
        )(x)
        mae_cls_pos_embed = mae_cls_token
        pos_embed = tf.concat([pos_embed, mae_cls_pos_embed], axis=1)

        return pos_embed


def mae_factory(
    img_h_size,
    img_w_size,
    ncams,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    decoder_embed_dim,
    decoder_depth,
    decoder_num_heads,
    reward_pred=True,
    in_chans=3,
    early_conv=False,
    view_masking=False,
    additional_cams_list=[],
    viewpoint_pos_emb=False,
):
    encoder = MaskedViTEncoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        ncams=ncams,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        early_conv=early_conv,
        view_masking=view_masking,
        additional_cams_list=additional_cams_list,
        viewpoint_pos_emb=viewpoint_pos_emb,
    )

    decoder = MaskedViTDecoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        ncams=ncams,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=decoder_embed_dim,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
        reward_pred=reward_pred,
        additional_cams_list=additional_cams_list,
        viewpoint_pos_emb=viewpoint_pos_emb,
    )
    return encoder, decoder


def flat_vit_factory(
    img_h_size,
    img_w_size,
    patch_size,
    embed_dim,
    depth,
    num_heads,
    decoder_embed_dim,
    decoder_depth,
    decoder_num_heads,
    in_chans=3,
):
    encoder = ViTEncoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    )
    decoder = ViTDecoder(
        img_h_size=img_h_size,
        img_w_size=img_w_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=decoder_embed_dim,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        mlp_ratio=4.0,
        norm_layer="layer_norm_eps_1e-6",
    )
    return encoder, decoder
