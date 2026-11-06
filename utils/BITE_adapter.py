from BITE.src.smal_pytorch.smal_model.smal_torch_new import SMAL as _BiteSMAL


class SMAL(_BiteSMAL):
    """ZeroCOI adapter for the original BITE SMAL implementation.

    Adds the global `scale` argument used by ZeroCOI and preserves
    ZeroCOI's expected return signature.
    """

    def __call__(
        self,
        beta,
        betas_limbs,
        theta=None,
        pose=None,
        trans=None,
        scale=None,
        del_v=None,
        get_skin=True,
        keyp_conf="red",
        get_all_info=False,
        shapedirs_sel=None,
        vert_off_compact=None,
    ):
        output = super().__call__(
            beta=beta,
            betas_limbs=betas_limbs,
            theta=theta,
            pose=pose,
            trans=None,
            del_v=del_v,
            get_skin=True,
            keyp_conf=keyp_conf,
            get_all_info=get_all_info,
            shapedirs_sel=shapedirs_sel,
            vert_off_compact=vert_off_compact,
        )

        if get_all_info:
            verts, _, rotations, bone_lengths = output
        else:
            verts, _, _ = output

        if scale is not None:
            if scale.ndim == 1:
                scale = scale.unsqueeze(-1)
            verts = verts * scale[:, None, :]

        if trans is not None:
            verts = verts + trans[:, None, :]

        joints = self.get_joints_from_verts(verts, keyp_conf=keyp_conf)

        if get_all_info:
            return verts, joints, rotations, bone_lengths
        if get_skin:
            return verts, joints
        return joints
