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
            sdgan.net.fig1_grid_zis = vars['fig1/grid_zis'];
            sdgan.net.fig1_grid_zos = vars['fig1/grid_zos'];
            sdgan.net.fig1_grid_Gzs = vars['fig1/grid_Gzs'];
            sdgan.net.fig1_lerp_Gzs = vars['fig1/lerp_Gzs'];
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

            // Concat identity and observation vector [100]
            var x = m.concat1D(_zi, _zo);

            // Project to [8, 8, 128]
            x = m.vectorTimesMatrix(x, net.vars['G/fully_connected/weights']);
            x = m.add(x, net.vars['G/fully_connected/biases']);
            x = x.reshape([128, 8, 8]);
            x = m.switchDim(x, [1, 2, 0]);

            // Conv 0
            x = m.conv2d(x, net.vars['G/Conv/weights'], net.vars['G/Conv/biases'], [1, 1], 'same');
            x = elu(x);

            // Conv 1
            x = m.conv2d(x, net.vars['G/Conv_1/weights'], net.vars['G/Conv_1/biases'], [1, 1], 'same');
            x = elu(x);

            // Upscale to [16, 16, 128]
            x = nn_upscale(x, 2);

            // Conv 2
            x = m.conv2d(x, net.vars['G/Conv_2/weights'], net.vars['G/Conv_2/biases'], [1, 1], 'same');
            x = elu(x);

            // Conv 3
            x = m.conv2d(x, net.vars['G/Conv_3/weights'], net.vars['G/Conv_3/biases'], [1, 1], 'same');
            x = elu(x);

            // Upscale to [32, 32, 128]
            x = nn_upscale(x, 2);

            // Conv 4
            x = m.conv2d(x, net.vars['G/Conv_4/weights'], net.vars['G/Conv_4/biases'], [1, 1], 'same');
            x = elu(x);

            // Conv 5
            x = m.conv2d(x, net.vars['G/Conv_5/weights'], net.vars['G/Conv_5/biases'], [1, 1], 'same');
            x = elu(x);

            // Upscale to [64, 64, 128]
            x = nn_upscale(x, 2);

            // Conv 6
            x = m.conv2d(x, net.vars['G/Conv_6/weights'], net.vars['G/Conv_6/biases'], [1, 1], 'same');
            x = elu(x);

            // Conv 7
            x = m.conv2d(x, net.vars['G/Conv_7/weights'], net.vars['G/Conv_7/biases'], [1, 1], 'same');
            x = elu(x);

            // Conv 8 to [64, 64, 3]
            x = m.conv2d(x, net.vars['G/Conv_8/weights'], net.vars['G/Conv_8/biases'], [1, 1], 'same');

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
