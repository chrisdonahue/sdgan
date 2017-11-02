window.sdgan = window.sdgan || {};

(function (deeplearn, sdgan) {
    // Config
    var cfg = sdgan.cfg;
    if (cfg.reqs.userCanceled) {
        return;
    }

    // Network state
    var net = {
        vars: null,
        ready: false
    };

    // Hardware state
    var hw = {
        math: null,
        ready: false
    };

    // Initialize hardware (uses WebGL if possible)
    var initHw = function (graph) {
        try {
            hw.math = new deeplearn.NDArrayMathGPU();
            cfg.debugMsg('WebGL supported');
        }
        catch(err) {
            hw.math = new deeplearn.NDArrayMathCPU();
            cfg.debugMsg('WebGL not supported');
        }

        hw.ready = true;
        cfg.debugMsg('Hardware ready');
    };

    // Initialize network and hardware
    var initVars = function () {
        var varLoader = new deeplearn.CheckpointLoader(cfg.net.ckpt_dir);
        varLoader.getAllVariables().then(function (vars) {
            net.vars = vars;
            /*
            sdgan.net.fig1_grid_zis = vars['fig1/grid_zis'];
            sdgan.net.fig1_grid_zos = vars['fig1/grid_zos'];
            sdgan.net.fig1_grid_Gzs = vars['fig1/grid_Gzs'];
            sdgan.net.fig1_lerp_Gzs = vars['fig1/lerp_Gzs'];
            */
            net.ready = true;

            cfg.debugMsg('Variables loaded');
        });
    };

    // Exports
    sdgan.net = {};

    sdgan.net.fig1_grid_zis = null;
    sdgan.net.fig1_grid_zos = null;
    sdgan.net.fig1_grid_Gzs = null;
    sdgan.net.fig1_lerp_Gzs = null;

    sdgan.net.isReady = function () {
        return net.ready && hw.ready;
    };

    sdgan.net.eval = function (_zi, _zo) {
        if (!sdgan.net.isReady()) {
            throw 'Hardware not ready';
        }
        if (!(_zi.length == cfg.net.d_i && _zo.length == cfg.net.d_o)) {
            throw 'Input shape incorrect'
        }

        var m = hw.math;

        _zi = deeplearn.Array1D.new(_zi);
        _zo = deeplearn.Array1D.new(_zo);

        var _output = m.scope(function () {
            // Hacky implementation of ELU... Super slow :(
            function elu(x) {
                x_pos = m.multiply(x, m.step(x));

                x_neg = m.sub(x, x_pos);
                x_neg = m.exp(x_neg);
                x_neg = m.sub(x_neg, deeplearn.Scalar.new(1.));

                return m.add(x_neg, x_pos);
            };

            // Hacky implementation of nearest neighbors integer upscale...
            function nn_upscale(x, s) {
                /*
                    NumPy implementation for nearest-neighbor upscaling 2x2 color image

                    _x = np.arange(12).reshape([2, 2, 3])

                    _x = np.stack([_x.copy(), _x.copy()], axis=1)
                    _x = np.reshape(_x, [4, 2, 3])

                    _x = np.stack([_x.copy(), _x.copy()], axis=2)
                    _x = np.reshape(_x, [4, 4, 3])
                */

                var h = x.shape[0];
                var w = x.shape[1];
                var nch = x.shape[2];

                x = x.as4D(h, 1, w, nch);
                x = m.concat4D(x, x, 1);
                x = x.reshape([h * s, w, nch]);

                x = x.as4D(h * s, w, 1, nch);
                x = m.concat4D(x, x, 2);
                x = x.reshape([h * s, w * s, nch]);

                return x;
            };

            function batchnorm(x, name) {
                var mean = net.vars[name + '/batch_normalization/moving_mean'];
                var variance = net.vars[name + '/batch_normalization/moving_variance'];
                var scale = net.vars[name + '/batch_normalization/gamma'];
                var offset = net.vars[name + '/batch_normalization/beta'];

                return m.batchNormalization3D(x, mean, variance, 0.001, scale, offset);
            };

            // Concat identity and observation vector [100]
            var x = m.concat1D(_zi, _zo);

            // Project to [4, 4, 1024]
            x = m.vectorTimesMatrix(x, net.vars['G/z_project/dense/kernel']);
            x = m.add(x, net.vars['G/z_project/dense/bias']);
            x = x.reshape([4, 4, 1024]);
            x = batchnorm(x, 'G/z_project');
            x = m.relu(x);

            // Conv 0 to [8, 8, 512]
            x = m.conv2dTranspose(x, net.vars['G/upconv_2d_0/conv2d_transpose/kernel'], [8, 8, 512], [2, 2], 'same');
            x = batchnorm(x, 'G/upconv_2d_0');
            x = m.relu(x);

            // Conv 1 to [16, 16, 256]
            x = m.conv2dTranspose(x, net.vars['G/upconv_2d_1/conv2d_transpose/kernel'], [16, 16, 256], [2, 2], 'same');
            x = batchnorm(x, 'G/upconv_2d_1');
            x = m.relu(x);

            // Conv 2 to [32, 32, 128]
            x = m.conv2dTranspose(x, net.vars['G/upconv_2d_2/conv2d_transpose/kernel'], [32, 32, 128], [2, 2], 'same');
            x = batchnorm(x, 'G/upconv_2d_2');
            x = m.relu(x);

            // Conv 3 to [64, 64, 3]
            x = m.conv2dTranspose(x, net.vars['G/upconv_2d_3/conv2d_transpose/kernel'], [64, 64, 3], [2, 2], 'same');
            x = m.tanh(x);

            // Denorm image and clip
            x = m.add(x, deeplearn.Scalar.new(1.));
            x = m.multiply(x, deeplearn.Scalar.new(127.5));
            x = m.clip(x, 0., 255.);

            return x;
        });

        return _output;
    };

    // Run immediately
    initVars();
    initHw();

})(window.deeplearn, window.sdgan);
