import jax.numpy as jnp


VALID_AP_FORMS = {"none", "mult", "add"}


def make_indoor_outdoor_mean(X_train, y_train):
    indoor_mask = X_train[:, 2].astype(bool)
    indoor_mean = y_train[indoor_mask].mean()
    outdoor_mean = y_train[~indoor_mask].mean()

    def m(obs):
        return obs[2] * indoor_mean + (1 - obs[2]) * outdoor_mean

    return m


def make_wifi_kernel(
    *,
    ls_xy,
    ls_z,
    os_xyz,
    ls_t,
    os_t,
    ls_ap,
    os_ap=0.0,
    ap_form="mult",
):
    """
    Build the Wi-Fi GP kernel used by the Gibbs sampler.

    ap_form="mult" preserves the previous model:
        os_xyz * exp(-(d_xyz + d_ap)) + os_t * exp(-d_t)

    ap_form="add" gives access point identity its own output scale:
        os_xyz * exp(-d_xyz) + os_ap * exp(-d_ap) + os_t * exp(-d_t)

    ap_form="none" ignores access point identity:
        os_xyz * exp(-d_xyz) + os_t * exp(-d_t)
    """
    if ap_form not in VALID_AP_FORMS:
        raise ValueError(f"ap_form must be one of {sorted(VALID_AP_FORMS)}.")

    ls_xy = float(ls_xy)
    ls_z = float(ls_z)
    os_xyz = float(os_xyz)
    ls_t = float(ls_t)
    os_t = float(os_t)
    ls_ap = float(ls_ap)
    os_ap = float(os_ap)
    ls_xyz = jnp.array([ls_xy, ls_xy, ls_z])

    def K(obs1, obs2):
        d_xyz = ((obs1[:3] - obs2[:3]) ** 2 / (2 * ls_xyz ** 2)).sum()
        d_t = (obs1[3] - obs2[3]) ** 2 / (2 * ls_t ** 2)

        spatial = os_xyz * jnp.exp(-d_xyz)
        temporal = os_t * jnp.exp(-d_t)

        if ap_form == "none":
            return spatial + temporal

        d_ap = jnp.where(obs1[4] == obs2[4], 0.0, 1.0) / (2 * ls_ap ** 2)
        if ap_form == "mult":
            return spatial * jnp.exp(-d_ap) + temporal

        access_point = os_ap * jnp.exp(-d_ap)
        return spatial + access_point + temporal

    return K
