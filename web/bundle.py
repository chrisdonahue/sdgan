import os
import shutil

bundle_dir = 'bundle'

paths = [
  'ckpts/sdbegan_548235_small',
  'css',
  'img',
  'js/deeplearn-0.3.2.min.js',
  'js/sdgan_cfg.js',
  'js/sdgan_net.js',
  'js/sdgan_reqs.js',
  'js/sdgan_ui.js',
  'index.html'
]

if os.path.exists(bundle_dir):
  shutil.rmtree(bundle_dir)

for path in paths:
  out_path = os.path.join(bundle_dir, path)
  print '{}->{}'.format(path, out_path)

  if os.path.isdir(path):
    shutil.copytree(path, out_path)
  else:
    out_dir = os.path.split(out_path)[0]
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    shutil.copy(path, out_path)

sdgan_cfg_fp = os.path.join(bundle_dir, 'js', 'sdgan_cfg.js')
with open(sdgan_cfg_fp, 'r') as f:
  sdgan_cfg = f.read()

sdgan_cfg = sdgan_cfg.replace('var debug = true;', 'var debug = false;')

with open(sdgan_cfg_fp, 'w') as f:
  f.write(sdgan_cfg)
