import gc
import pickle
import numpy as np
import healpy as hp
from matplotlib import cm
from datetime import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sphere.distribution import fb8_mle, FB8Distribution


all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',
                   'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
                   'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                   'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']

joint_names = ['head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
               'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
               'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'pelvis',
               'spine', 'head']

joint_tree = [1, 15, 1, 2, 3,
              1, 5, 6, 14, 8,
              9, 14, 11, 12, 14,
              14, 1]

child_joints = [all_joint_names.index(joint_name) for joint_name in joint_names]
parent_joints = [all_joint_names.index(joint_names[i]) for i in joint_tree]
pair_joints = list(zip(child_joints, parent_joints))


def asSpherical(xyz):
    # takes list xyz (single coord)
    from math import sqrt, acos, atan2, pi
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = sqrt(x * x + y * y + z * z)
    theta = acos(z / r) * 180 / pi  # to degrees
    phi = atan2(y, x) * 180 / pi
    return [r, theta, phi]


def calc_rarenesses(models, data):
    Rs = []
    for pose in data:
        rareness = 0
        for i in range(len(models)):
            if pose[i][0] == 0:
                continue
            xs = FB8Distribution.spherical_coordinates_to_nu(pose[i, 1], pose[i, 2])
            p = models[i][1].pdf(xs)
            rareness += p
        Rs.append(rareness)
    return Rs


def make_title(fb8, kbdec=0):
    def FBname(n):
        return r'\rm{{FB}}_{}'.format(n)

    def FBtitle(n, ps):
        return r'${}({})$'.format(FBname(n), ps)

    kapbet = r'\kappa = {:.' + str(kbdec) + r'f}, \beta = {:.' + str(kbdec) + r'f}'
    kapbet = kapbet.format(fb8.kappa, fb8.beta)
    if fb8.nu[0] == 1.:
        if fb8.eta == 1.:
            return FBtitle(5, kapbet)
        if fb8.eta == -1.:
            return FBtitle(4, kapbet)
        return FBtitle(6, kapbet + r', \eta={:.1g}'.format(fb8.eta))
    return FBtitle(8, kapbet + r', \eta={:.1g}, \vec{{\nu}}=({:.3g},{:.3g},{:.3g})'.format(
        fb8.eta, np.round(fb8.nu[0], 3), np.round(fb8.nu[1], 3), np.round(fb8.nu[2], 3)))


def hp_plot_fb8(fb8, nside):
    import healpy as hp
    npix = hp.nside2npix(nside)
    fb8_map = fb8.pdf(fb8.spherical_coordinates_to_nu(
        *hp.pix2ang(nside, np.arange(npix))))

    plt.figure(figsize=(9, 6))
    vmap = cm.plasma
    vmap.set_under('w')
    vmap.set_bad('w')
    hp.mollview(fb8_map,
                title=make_title(fb8, 1),
                min=0,
                max=np.round(np.nanmax(fb8_map), 2),
                cmap=vmap, hold=True,
                cbar=True,
                xsize=1600)
    hp.graticule()


def hp_fits(num_v, ths, phs, nside=64, toy=False):
    xs = FB8Distribution.spherical_coordinates_to_nu(ths, phs)
    z, x, y = xs.T
    s_index = np.random.choice(xs.shape[0], 10000, replace=False)
    s_ths = [ths[i] for i in s_index]
    s_phs = [phs[i] for i in s_index]

    fit8 = fb8_mle(xs, True)
    hp_plot_fb8(fit8, nside)
    hp.projscatter(s_ths, s_phs, marker='.', linewidths=0, s=5, c='k')
    ax = plt.gca()
    ax.annotate(r"$\bf{-180^\circ}$", xy=(1.7, 0.625), size="medium")
    ax.annotate(r"$\bf{180^\circ}$", xy=(-1.95, 0.625), size="medium")
    ax.annotate("Galactic", xy=(0.8, -0.05),
                size="medium", xycoords="axes fraction")
    if toy:
        plt.savefig('figs/{}_toy_fb8.pdf'.format(num_v))
    else:
        plt.savefig('figs/{}_fb8.pdf'.format(num_v))
    plt.clf()
    plt.close('all')
    gc.collect()
    return fit8


def calc_fitted_fb8(data, toy):
    print("Start FB8 fitting.")
    fitted_fb8 = []
    for i, j in enumerate(range(data.shape[1])):
        try:
            xs = np.array([asSpherical(xyz) for xyz in data[:, j]])
            fitted_fb8.append([i, hp_fits(j, xs[:, 1], xs[:, 2], toy=toy)])
        except Exception:
            print("error at {}: {}".format(j, i))
    return fitted_fb8


def calc_rareness(data, toy):
    print("Start rareness calculation.")
    direct_dist = []
    for ann in data[:, 0]:
        direct_dist.append(np.array([ann[p] - ann[c] for c, p in pair_joints]))
    direct_dist = np.array(direct_dist)
    fb8_models = calc_fitted_fb8(direct_dist, toy)

    update_date = datetime.now().strftime("%y%m%d%H%M%S")
    output_path1 = "./res/fb8_res_{}.pck".format(update_date) if not toy \
                   else "./res/fb8_res_toy_{}.pck".format(update_date)
    with open(output_path1, "wb") as f:
        pickle.dump(fb8_models, f)

    Rs = []
    for pose in direct_dist:
        rareness = 0
        for i in range(len(fb8_models)):
            if pose[i][0] == 0:
                continue
            xs = FB8Distribution.spherical_coordinates_to_nu(pose[i, 1], pose[i, 2])
            p = fb8_models[i][1].pdf(xs)
            rareness += p
        Rs.append(rareness)

    res = [[d[0], d[1], r] for d, r in zip(data, Rs)]
    output_path2 = "./res/rareness_{}.pck".format(update_date) if not toy \
                   else "./res/rareness_toy_{}.pck".format(update_date)
    with open(output_path2, "wb") as f:
        pickle.dump(res, f)

    return res


def calc_rareness_with_model(data, model, toy):
    print("Start rareness calculation with model.")
    direct_dist = []
    for ann in data[:, 0]:
        direct_dist.append(np.array([ann[p] - ann[c] for c, p in pair_joints]))
    direct_dist = np.array(direct_dist)

    print("get direct distribute complete.")

    Rs = []
    for pose in direct_dist:
        rareness = 0
        for i in range(len(model)):
            if pose[i][0] == 0:
                continue
            xs = FB8Distribution.spherical_coordinates_to_nu(pose[i, 1], pose[i, 2])
            p = model[i][1].pdf(xs)
            rareness *= p
        Rs.append(rareness)

    update_date = datetime.now().strftime("%y%m%d%H%M%S")
    res = [[d[0], d[1], r] for d, r in zip(data, Rs)]
    output_path = "./res/rareness_{}.pck".format(update_date) if not toy \
                  else "./res/rareness_toy_{}.pck".format(update_date)
    with open(output_path, "wb") as f:
        pickle.dump(res, f)

    return res
