window.sdgan = window.sdgan || {};

(function (sdgan) {
    var debug = false;

    // Config
    sdgan.cfg = {
        reqs: {
            userCanceled: false,
            noWebGlWarning: 'Warning: We did not find WebGL in your browser. This demo uses WebGL to accelerate neural network computation. Performance will be slow and may hang your browser. Continue?',
            mobileWarning: 'Warning: This demo runs a neural network in your browser. It appears you are on a mobile device. Consider running the demo on your laptop instead. Continue?'
        },
        net: {
            ckpt_dir: 'ckpts/sdbegan_548235_small',
            d_i: 50,
            d_o: 50,
            img_h: 64,
            img_w: 64
        },
        ui: {
            canvasFlushDelayMs: 25,
            background_color: '#fefefd',
            nids: 4,
            nobs: 4,
            lerp_nids: 4,
            lerp_nobs: 4,
            fig1_z: true,
            fig1_zo: [0, 1, 2, 3]
        },
    };

    sdgan.cfg.debugMsg = function (msg) {
        if (debug) {
            console.log(msg);
        }
    };

})(window.sdgan);
