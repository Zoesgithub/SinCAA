from openfold.utils.rigid_utils import Rigid, Rotation, torch, rot_vec_mul, rot_matmul


def handle_zero(cosv, sinv):
    pad_pos = ((cosv == 0) & (sinv == 0))
    pad_pos = pad_pos.float()
    return cosv*(1-pad_pos)+pad_pos*cosv.new_ones(cosv.shape)


class RobustRigid(Rigid):
    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8
    ) -> Rigid:
        """
            Implements algorithm 21. Constructs transformations from sets of 3 
            points using the Gram-Schmidt algorithm.

            Args:
                p_neg_x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e0_is_zero = torch.stack([abs(_) for _ in e0], -1).max(-1).values < eps
        assert e0_is_zero.shape == e0[0].shape
        e0_robust = e0[0].new_ones(e0[0].shape)
        e0[0] = e0[0]*(1-e0_is_zero.float())+e0_robust*e0_is_zero.float()

        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]
        e1_is_zero = torch.stack([abs(_) for _ in e1], -1).max(-1).values < eps
        assert e1_is_zero.shape == e1[1].shape
        e1_robust = e1[1].new_ones(e1[1].shape)
        e1[1] = e1[1]*(1-e1_is_zero.float())+e1_robust*e1_is_zero.float()

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        return RobustRigid.from_3_points(n_xyz, ca_xyz, c_xyz, eps=eps)
