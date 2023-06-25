import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import polars as pl
from PIL import Image

saturn_fpath = os.path.join('images', 'InterstellarWormhole_Fig6a-1750x875.jpeg')
galaxies_fpath = os.path.join('images', 'InterstellarWormhole_Fig10.jpeg')

saturn_im = Image.open(saturn_fpath)
galaxies_im = Image.open(galaxies_fpath)

print('saturn', saturn_im.size, saturn_im.mode)
print('galaxies', galaxies_im.size, galaxies_im.mode)

# Choose which one is where
lower_im = saturn_im
upper_im = galaxies_im
out_scale = 1000
out_size = tuple(np.array([2, 1]) * out_scale) # Ratio + pixels

out_im = Image.new("RGB", out_size, color='#ff00e5')

def roll_image(img, roll_type='horizontal', delta = 1):
    x , y = img.size
    if roll_type=='horizontal':
        part1 = img.crop((0,0, delta, y ))
        part2 = img.crop((delta, 0, x, y))
    else:
        part1 = img.crop((0, 0, x, delta ))
        part2 = img.crop((0, delta, x, y))

    part1.load()
    part2.load()

    img.paste(part2)
    img.paste(part1, box = (x-delta, 0) if roll_type=='horizontal' else (0, y-delta))
    return img

# Algunas constantes
W_M = -np.log( 1/np.cos(np.pi/(2*np.sqrt(2))) ) + (np.pi/(2*np.sqrt(2)))*np.tan(np.pi/(2*np.sqrt(2)))
print("W_M = ", W_M) # Close enough. paper says 1.42053... but we get like 1.42953...

# Parametros del wormhole
throat_radius = 1 # km
lensing = 0.05 * throat_radius # km
throat_length = 0.01 * throat_radius # km

# Same parameters but in their equation's notation
rho = throat_radius # km
W = lensing
a = throat_length / 2 # km

# Constantes del wormhole
M = W / W_M
dr_dl = lambda l: (2 / np.pi) * np.arctan( 2*l / (np.pi * M))


# Choose a camera location. 'c' means 'camera'
l_c = -50 * a
theta_c = np.pi / 2
phi_c = np.pi/2

n_phi_samples = out_size[0]
n_theta_samples = out_size[1]
n_total_samples = n_phi_samples * n_theta_samples

theta_sep = np.pi / n_theta_samples
phi_sep = np.pi * 2 / n_phi_samples

# print('phi_cs_samples', np.linspace(0, np.pi*2, n_phi_samples, endpoint=False))
# print('theta_cs_samples', np.linspace(0, np.pi, n_theta_samples, endpoint=False))

# STAHP

i = 0
results = []
for (yidx_cs, theta_cs) in enumerate(np.linspace(0, np.pi, n_theta_samples, endpoint=False)):
    for (xidx_cs, phi_cs) in enumerate(np.linspace(0, np.pi*2, n_phi_samples, endpoint=False)):
        print('i / total:', i, '/', n_total_samples)
        # Choose 'celestial sphere' angle (must ITERATE this)
        # theta_cs = np.pi * 1/4
        # phi_cs = np.pi * 0

        res = dict(i=i, theta_cs = theta_cs, phi_cs = phi_cs, xidx_cs = xidx_cs, yidx_cs = yidx_cs)

        # Unit vector pointing in one direction to sky
        # 'cs' stands for 'celesital sphere' NOT camera
        N_x = np.sin(theta_cs) * np.cos(phi_cs)
        N_y = np.sin(theta_cs) * np.sin(phi_cs)
        N_z = np.cos(theta_cs)

        # Incoming ray's components in global spherical polar basis
        n_l = -N_x
        n_phi = -N_y
        n_theta = +N_z

        # Initial conditions for ray
        l_0 = l_c
        theta_0 = theta_c
        phi_0 = phi_c

        def get_r(l):
            x = 2 * (np.abs(l) - a) / ( np.pi * M )
            return np.where(
                np.abs(l) > a,
                rho + M*(x*np.arctan(x) - 1/2*np.log(1 + x**2)),
                rho
            )

        r_0 = get_r(l_0)

        # Incoming ray's canonical momenta
        p_l0 = n_l
        p_theta0 = r_0 * n_theta
        p_phi0 = r_0 * np.sin(theta_0) * n_phi

        # Ray's constants of motion
        b = p_phi0
        B = np.sqrt( p_theta0**2 + (p_phi0 / np.sin(theta_0))**2 )



        # Missing: b, B, r, dr_dl
        def ray_geodesic(t, state):
            l, theta, phi, p_l, p_theta = state
            r = get_r(l)

            dl = p_l
            dtheta = p_theta / r**2
            dphi = b / (r * np.sin(theta))**2
            dp_l = B**2 * (dr_dl(l)) / r**3 # FILL IN dr_dl args
            dp_theta = (b / r)**2 * np.cos(theta) / np.sin(theta)**3

            return [dl, dtheta, dphi, dp_l, dp_theta]

        y0 = [l_0, theta_0, phi_0, p_l0, p_theta0]
        t_span = [0, -10]
        result_solve_ivp = solve_ivp(ray_geodesic, t_span, y0, 
                                    # method='LSODA',
                                    # max_step=0.01
                                    )

        # print(result_solve_ivp.message)
        # print(result_solve_ivp)

        ts = result_solve_ivp.t
        ys = result_solve_ivp.y

        tlast = ts[-1]
        # ylast = ys[:, -1]

        res['tlast'] = tlast
        res['l_last'] = ys[0, -1]
        res['theta_last'] = ys[1, -1]
        res['phi_last'] = ys[2, -1]

        # ls = ys[0]
        # lnorms = ls / a

        # thetas = ys[1]
        # phis = ys[2]

        # rs = get_r(ls)
        # rxs = rs * np.cos(phis)
        # rys = rs * np.sin(phis)

        results.append(res)
        i += 1

pl.Config.set_tbl_cols(20)
results = (
    pl.DataFrame(results)
      .with_columns([
          (pl.col('theta_last').apply(lambda x: x % (np.pi*2))).alias('theta_last_mod'),
            (pl.col('phi_last').apply(lambda x: x % (np.pi*2))).alias('phi_last_mod'),
      ])
      .with_columns([
          
          pl.when(pl.col('l_last') > 0)
            .then( pl.lit('upper') )
            .otherwise( pl.lit('lower') )
            .alias('hemisphere_last'),

          pl.when(pl.col('theta_last_mod') > np.pi)
            .then( 2*np.pi - pl.col('theta_last_mod') )
            .otherwise( pl.col('theta_last_mod') )
            .alias('theta_last_fixed'),

          pl.when(pl.col('theta_last_mod') > np.pi)
            .then( (pl.col('phi_last_mod') + np.pi).apply(lambda x: x % (np.pi*2)) )
            .otherwise( pl.col('phi_last_mod') )
            .alias('phi_last_fixed'),
        
      ])
      .with_columns([
          
          (pl.col('theta_last_fixed') / np.pi).alias('theta_pct'),
          (pl.col('phi_last_fixed') / (np.pi*2)).alias('phi_pct'),
            

      ])
      .with_columns([
        pl.when(pl.col('hemisphere_last') == 'lower')
                .then( pl.col('phi_pct') * lower_im.size[0] )
                .otherwise( pl.col('phi_pct') * upper_im.size[0] )
                .floor()
                .cast(int)
                .alias('phi_im_x'),

        pl.when(pl.col('hemisphere_last') == 'lower')
                .then( pl.col('theta_pct') * lower_im.size[1] )
                .otherwise( pl.col('theta_pct') * upper_im.size[1] )
                .floor()
                .cast(int)
                .alias('theta_im_y'),



        # ((pl.col('phi_cs') / (2*np.pi)) * out_im.size[0])
        # .cast(int)
        # .alias('phi_cs_im_x'),

        
        # ((pl.col('theta_cs') / np.pi) * out_im.size[1])
        # .cast(int)
        # .alias('theta_cs_im_y'),



            
      ])
)
print(
    results
        .select('theta_last', 'phi_last', 'theta_last_mod', 'phi_last_mod', 'theta_last_fixed', 'phi_last_fixed',
                'theta_pct', 'phi_pct', 'hemisphere_last',
                'phi_cs', 'theta_cs', 
                # 'phi_cs_im_x', 'theta_cs_im_y',
                'phi_im_x', 'theta_im_y'
                )
)

print('saturn', saturn_im.size, saturn_im.mode)
print('galaxies', galaxies_im.size, galaxies_im.mode)
print('out', out_im.size, out_im.mode)

print(
    results
        .select([
            # pl.col('phi_cs_im_x').min().alias('phi_cs_im_x_min'),
            # pl.col('phi_cs_im_x').max().alias('phi_cs_im_x_max'),
            # pl.col('theta_cs_im_y').min().alias('theta_cs_im_y_min'),
            # pl.col('theta_cs_im_y').max().alias('theta_cs_im_y_max'),
            # pl.lit(None).alias('spacer'),
            pl.col('phi_im_x').min().alias('phi_im_x_min'),
            pl.col('phi_im_x').max().alias('phi_im_x_max'),
            pl.col('theta_im_y').min().alias('theta_im_y_min'),
            pl.col('theta_im_y').max().alias('theta_im_y_max'),
        ])
)
# where_cerca_wormhole = (np.abs(lnorms) < 10000)

# for i in range(out_size[0]):
#     for j in range(out_size[1]):
#         if i > 30:
#             out_im.putpixel((i, j), (0, 0, 0))

# out_im.show()
for row in results.iter_rows(named=True):
    # i = row['phi_cs_im_x']
    # j = row['theta_cs_im_y']
    i = row['xidx_cs']
    j = row['yidx_cs']
    if row['hemisphere_last'] == 'lower':
        im = lower_im
    else:
        im = upper_im
    i_im = min(row['phi_im_x'], im.size[0]-1)
    j_im = min(row['theta_im_y'], im.size[1]-1)
    im_pixel = im.getpixel((i_im, j_im))
    # print('placing im_pixel', im_pixel, 'at', (i,j))
    out_im.putpixel((i, j), im_pixel)

rolled_out = roll_image(out_im, roll_type='horizontal', delta=out_im.size[0]//2)
rolled_out.show()
# print(ts)
# print(ls)

# fig, axs = plt.subplots(2, 2)
# print('axs:', axs)

# ax = axs[0, 0]
# ax.plot(ts[where_cerca_wormhole], lnorms[where_cerca_wormhole], '-o')
# ax.fill_between(ts[where_cerca_wormhole],y1=-1, y2=1, color='red', alpha=0.5)
# # ax.set_title('title name')
# ax.set_xlabel('ts')
# ax.set_ylabel('lnorms')

# ax = axs[0, 1]
# ax.plot(ts[where_cerca_wormhole], thetas[where_cerca_wormhole], '-o')
# # ax.set_title('title name')
# ax.set_xlabel('ts')
# ax.set_ylabel('thetas')

# ax = axs[1, 0]
# ax.plot(ts[where_cerca_wormhole], phis[where_cerca_wormhole], '-o')
# # ax.set_title('title name')
# ax.set_xlabel('ts')
# ax.set_ylabel('phis')

# ax = axs[1, 1]
# ax.plot([0], [0], 'x', color='black')
# ax.plot(rxs[where_cerca_wormhole], rys[where_cerca_wormhole], '-o')
# # ax.set_title('title name')
# ax.set_xlabel('rxs')
# ax.set_ylabel('rys')

# plt.show()