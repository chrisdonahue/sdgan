window.sdgan = window.sdgan || {};

(function (deeplearn, sdgan) {
    // Config
    var cfg = sdgan.cfg;
    if (cfg.reqs.userCanceled) {
        document.getElementById('demo').setAttribute('hidden', '');
        document.getElementById('canceled').removeAttribute('hidden');
        return;
    }

    // Make a new random vector
    var random_vector = function (d) {
        var z = new Float32Array(d);
        for (var i = 0; i < d; ++i) {
            z[i] = (Math.random() * 2.) - 1.;
        }
        return z;
    };

    // Make identity latent codes
    var zis = new Array(cfg.ui.nids);
    for (var i = 0; i < cfg.ui.nids; ++i) {
        zis[i] = random_vector(cfg.net.d_i);
    }

    // Make observation latent codes
    var zos = new Array(cfg.ui.nobs);
    for (var i = 0; i < cfg.ui.nobs; ++i) {
        zos[i] = random_vector(cfg.net.d_o);
    }

    // Make object to store latent codes for table entries
    var io_key_to_z = {};
    var io_key_to_Gz = {};
    var select0 = null;
    var readyForInput = false;

    // Combine disentangled latent codes into one
    var zi_zo_to_z = function (zi, zo) {
        z_length = zi.length + zo.length;
        var z = new Float32Array(z_length);

        var i = 0;
        for (; i < zi.length; ++i) {
            z[i] = zi[i];
        }
        for (; i < z_length; ++i) {
            z[i] = zo[i - zi.length];
        }

        return z;
    };

    // Linear interpolation between two vectors
    var z_lerp = function (z0, z1, a) {
        if (z0.length !== z1.length) {
            throw 'Vector length differs';
        }

        var interp = new Float32Array(z0.length);
        for (var i = 0; i < z0.length; ++i) {
            interp[i] = (1. - a) * z0[i] + a * z1[i];
        }

        return interp;
    };

    // Hacky hash function to keep track of vectors
    var hash_io_key = function (i, o) {
        return 'i' + String(i) + 'o' + String(o);
    };

    var io_key_to_io = function (io_key) {
        return [Number(io_key[1]), Number(io_key[3])];
    };

    // Hacky latent code equality function
    var latent_code_equal = function (z0, z1) {
        return z0[0] == z1[0] && z0[50] == z1[50]
    };

    // Pause input from users
    var pauseUserInput = function () {
        document.getElementById('b_all').setAttribute('disabled', '');
        for (var i = 0; i < cfg.ui.nids; ++i) {
            document.getElementById('b_i' + String(i)).setAttribute('disabled', '');
        }
        for (var o = 0; o < cfg.ui.nobs; ++o) {
            document.getElementById('b_o' + String(o)).setAttribute('disabled', '');
        }

        readyForInput = false;
    };

    // Allow input from users
    var resumeUserInput = function () {
        document.getElementById('b_all').removeAttribute('disabled');
        for (var i = 0; i < cfg.ui.nids; ++i) {
            document.getElementById('b_i' + String(i)).removeAttribute('disabled');
        }
        for (var o = 0; o < cfg.ui.nobs; ++o) {
            document.getElementById('b_o' + String(o)).removeAttribute('disabled');
        }

        readyForInput = true;
    };

    // Renders either text or Float32Arr image on a context
    var drawCell = function (ctx, contents, selected) {
        if (typeof(contents) === 'string') {
            var msg = contents;

            ctx.clearRect(0, 0, cfg.net.img_h, cfg.net.img_w);

            if (msg === '') {
                return;
            }

            ctx.fillStyle = cfg.ui.background_color;
            ctx.fillRect(1, 24, 62, 18);
            ctx.fillStyle = '#000000';

            ctx.font = '11px Arial';

            var y = cfg.net.img_h;

            var msgs = msg.split('\n');
            if (msgs.length == 1) {
                y = 36;
            }
            else if (msgs.length == 2) {
                y = 31;
            }
            else {
                throw 'Too many lines';
            }
            for (var i = 0; i < msgs.length; ++i) {
                var msg = msgs[i];
                var msgWidth = ctx.measureText(msg);
                ctx.fillText(msg, (cfg.net.img_w - msgWidth.width) / 2., y);
                y += 11;
            }
        }
        else {
            var Gz = contents;

            var imgData = ctx.createImageData(cfg.net.img_h, cfg.net.img_w);
            var pixel = 0;
            for (var i = 0; i < cfg.net.img_h; ++i) {
                for (var j = 0; j < cfg.net.img_w; ++j) {
                    imgData.data[pixel++] = Gz.get(i, j, 0);
                    imgData.data[pixel++] = Gz.get(i, j, 1);
                    imgData.data[pixel++] = Gz.get(i, j, 2);
                    imgData.data[pixel++] = 255;
                }
            }
            ctx.putImageData(imgData, 0, 0);

            if (Boolean(selected)) {
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 4;
                ctx.rect(0, 0, cfg.net.img_h, cfg.net.img_h);
                ctx.stroke();
            }
        }
    };

    // Get canvas context for grid cell
    var getGridCtx = function (i, o) {
        var canvas = document.getElementById('c_i' + String(i) + '_o' + String(o));
        return canvas.getContext('2d');
    };

    // Get canvas context for lerp cell
    var getLerpCtx = function (i, o) {
        var canvas = document.getElementById('c_lerp_i' + String(i) + '_o' + String(o));
        return canvas.getContext('2d');
    };

    // Generate one cell
    var generate = function (i, o) {
        if (hash_io_key(i, o) === select0) {
            selectImageCallbackFactory(i, o)('override');
        }

        var zi = zis[i];
        var zo = zos[o];
        var Gz = sdgan.net.eval(zi, zo);

        var io_key = hash_io_key(i, o);
        io_key_to_z[io_key] = zi_zo_to_z(zi, zo);
        io_key_to_Gz[io_key] = Gz;

        drawCell(getGridCtx(i, o), Gz);
    };

    // Generate all cells
    var generateAll = function () {
        for (var i = 0; i < cfg.ui.nids; ++i) {
            for (var o = 0; o < cfg.ui.nobs; ++o) {
                generate(i, o);
            }
        }
    };

    // Clear lerp grid
    var lerpClear = function (msg, msg_tl, msg_tr, msg_br, msg_bl) {
        for (var i = 0; i < cfg.ui.lerp_nids; ++i) {
            for (var o = 0; o < cfg.ui.lerp_nobs; ++o) {
                var realMsg = msg;
                if (i === 0 && o === 0) {
                    realMsg = msg_tl;
                }
                if (i === 0 && o === (cfg.ui.lerp_nobs - 1)) {
                    realMsg = msg_tr;
                }
                if (i === (cfg.ui.lerp_nids - 1) && o === (cfg.ui.lerp_nobs - 1)) {
                    realMsg = msg_br;
                }
                if (i === (cfg.ui.lerp_nids - 1) && o === 0) {
                    realMsg = msg_bl;
                }
                drawCell(getLerpCtx(i, o), realMsg);
            }
        }
    };

    // Generate and render lerp grid
    var lerpGenerateAndRender = function (z0, z1) {
        var zi0 = z0.slice(0, cfg.net.d_i);
        var zi1 = z1.slice(0, cfg.net.d_i);

        var zo0 = z0.slice(cfg.net.d_i, cfg.net.d_i + cfg.net.d_o);
        var zo1 = z1.slice(cfg.net.d_i, cfg.net.d_i + cfg.net.d_o);

        for (var i = 0; i < cfg.ui.lerp_nids; ++i) {
            var zi_lerp = z_lerp(zi0, zi1, i / (cfg.ui.lerp_nids - 1));
            for (var o = 0; o < cfg.ui.lerp_nobs; ++o) {
                if (i === 0 && o === 0) {
                    continue;
                }
                if (i === 0 && o === (cfg.ui.lerp_nobs - 1)) {
                    continue;
                }
                if (i === (cfg.ui.lerp_nids - 1) && o === 0) {
                    continue;
                }
                if (i === (cfg.ui.lerp_nids - 1) && o === (cfg.ui.lerp_nobs - 1)) {
                    continue;
                }
                var zo_lerp = z_lerp(zo0, zo1, o / (cfg.ui.lerp_nobs - 1));

                var Gz = sdgan.net.eval(zi_lerp, zo_lerp);
                drawCell(getLerpCtx(i, o), Gz);
            }
        }
    };

    // Callbacks for randomizing z
    var changeAllCallback = function () {
        if (!(readyForInput)) {
            return;
        }

        pauseUserInput();
        for (var i = 0; i < cfg.ui.nids; ++i) {
            for (var o = 0; o < cfg.ui.nobs; ++o) {
                drawCell(getGridCtx(i, o), 'Thinking...');
            }
        }

        setTimeout(function () {
            for (var i = 0; i < cfg.ui.nids; ++i) {
                zis[i] = random_vector(cfg.net.d_i);
            }
            for (var o = 0; o < cfg.ui.nobs; ++o) {
                zos[o] = random_vector(cfg.net.d_o);
            }
            generateAll();
            resumeUserInput();
        }, cfg.ui.canvasFlushDelayMs);
    };

    var changeIdCallbackFactory = function (i) {
        return function () {
            if (!(readyForInput)) {
                return;
            }

            pauseUserInput();
            for (var o = 0; o < cfg.ui.nobs; ++o) {
                drawCell(getGridCtx(i, o), 'Thinking...');
            }

            setTimeout(function () {
                zis[i] = random_vector(cfg.net.d_i);
                for (var o = 0; o < cfg.ui.nobs; ++o) {
                    generate(i, o);
                }
                resumeUserInput();
            }, cfg.ui.canvasFlushDelayMs);
        };
    };

    var changeObsCallbackFactory = function (o) {
        return function () {
            if (!(readyForInput)) {
                return;
            }

            pauseUserInput();
            for (var i = 0; i < cfg.ui.nids; ++i) {
                drawCell(getGridCtx(i, o), 'Thinking...');
            }

            setTimeout(function () {
                zos[o] = random_vector(cfg.net.d_o);
                for (var i = 0; i < cfg.ui.nids; ++i) {
                    generate(i, o);
                }
                resumeUserInput();
            }, cfg.ui.canvasFlushDelayMs);
        };
    };

    // Callback to select image for lerp
    var selectImageCallbackFactory = function (i, o) {
        return function (override) {
            if (override !== 'override' && !(readyForInput)) {
                return;
            }

            var io_key = hash_io_key(i, o);
            if (!(io_key in io_key_to_Gz)) {
                return;
            }

            var Gz = io_key_to_Gz[io_key];

            if (select0 === null) {
                select0 = io_key;
                lerpClear('', Gz, '', 'Select B', '');
                drawCell(getGridCtx(i, o), Gz, true)
            }
            else {
                if (io_key === select0) {
                    // Clear if unselecting
                    lerpClear('', 'Select A', '', 'Select B', '');
                }
                else {
                    // Compute and render lerp
                    pauseUserInput();

                    var a_io = io_key_to_io(select0);
                    var b_io = io_key_to_io(io_key);
                    var tr_key = hash_io_key(a_io[0], b_io[1]);
                    var bl_key = hash_io_key(b_io[0], a_io[1]);

                    lerpClear('Thinking...', 
                        io_key_to_Gz[select0], 
                        io_key_to_Gz[tr_key], 
                        io_key_to_Gz[io_key], 
                        io_key_to_Gz[bl_key]);

                    var z0 = io_key_to_z[select0];
                    var z1 = io_key_to_z[io_key];
                    setTimeout(function () {
                        lerpGenerateAndRender(z0, z1);
                        resumeUserInput();
                    }, cfg.ui.canvasFlushDelayMs);
                }

                // Clear border on old select0
                var io0 = io_key_to_io(select0);
                var Gz_old = io_key_to_Gz[select0];
                select0 = null;
                drawCell(getGridCtx(io0[0], io0[1]), Gz_old);
            }
        };
    };

    var onResize = function (event) {
        var demo = document.getElementById('demo');
        var demoHeight = demo.offsetTop + demo.offsetHeight;
        var viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);

        var content = document.getElementById('content');
        if (demoHeight > viewportHeight) {
            content.style.height = String(demoHeight) + 'px';
        }
        else {
            content.style.height = '';
        }
    };

    // Run once DOM loads
    var domReady = function () {
        cfg.debugMsg('DOM ready');

        window.addEventListener('resize', onResize, true);
        onResize();

        pauseUserInput();

        if (cfg.ui.fig1_z) {
            // Render "Loading..."
            for (var i = 0; i < cfg.ui.nids; ++i) {
                for (var o = 0; o < cfg.ui.nobs; ++o) {
                    drawCell(getGridCtx(i, o), 'Loading...');
                    drawCell(getLerpCtx(i, o), 'Loading...');
                }
            }
        }
        else {
            // Render "Thinking..."
            for (var i = 0; i < cfg.ui.nids; ++i) {
                for (var o = 0; o < cfg.ui.nobs; ++o) {
                    drawCell(getGridCtx(i, o), 'Thinking...');
                }
            }
            lerpClear('', '', '', '', '');
        }

        // (Gross) wait for net to be ready
        var wait = function () {
            if (sdgan.net.isReady()) {
                if (cfg.ui.fig1_z) {
                    var fig1_zis = sdgan.net.fig1_grid_zis;
                    for (var i = 0; i < cfg.ui.nids; ++i) {
                        for (var j = 0; j < cfg.net.d_i; ++j) {
                            zis[i][j] = fig1_zis.get(i, j);
                        }
                    }

                    var fig1_zos = sdgan.net.fig1_grid_zos;
                    for (var o = 0; o < cfg.ui.nobs; ++o) {
                        for (var j = 0; j < cfg.net.d_i; ++j) {
                            zos[o][j] = fig1_zos.get(cfg.ui.fig1_zo[o], j);
                        }
                    }

                    var fig1_grid_Gzs = sdgan.net.fig1_grid_Gzs;
                    var fig1_lerp_Gzs = sdgan.net.fig1_lerp_Gzs;
                    for (var i = 0; i < cfg.ui.nids; ++i) {
                        var zi = zis[i];
                        for (var o = 0; o < cfg.ui.nobs; ++o) {
                            var zo = zos[o];

                            var Gz = deeplearn.Array3D.new([cfg.net.img_h, cfg.net.img_w, 3], new Float32Array(cfg.net.img_h * cfg.net.img_w * 3));
                            for (var j = 0; j < cfg.net.img_h; ++j) {
                                for (var k = 0; k < cfg.net.img_w; ++k) {
                                    Gz.set(fig1_grid_Gzs.get(i, o, j, k, 0), j, k, 0);
                                    Gz.set(fig1_grid_Gzs.get(i, o, j, k, 1), j, k, 1);
                                    Gz.set(fig1_grid_Gzs.get(i, o, j, k, 2), j, k, 2);
                                }
                            }

                            drawCell(getGridCtx(i, o), Gz);

                            var io_key = hash_io_key(i, o);
                            io_key_to_z[io_key] = zi_zo_to_z(zi, zo);
                            io_key_to_Gz[io_key] = Gz;

                            var Gz = deeplearn.Array3D.new([cfg.net.img_h, cfg.net.img_w, 3], new Float32Array(cfg.net.img_h * cfg.net.img_w * 3));
                            for (var j = 0; j < cfg.net.img_h; ++j) {
                                for (var k = 0; k < cfg.net.img_w; ++k) {
                                    Gz.set(fig1_lerp_Gzs.get(i, o, j, k, 0), j, k, 0);
                                    Gz.set(fig1_lerp_Gzs.get(i, o, j, k, 1), j, k, 1);
                                    Gz.set(fig1_lerp_Gzs.get(i, o, j, k, 2), j, k, 2);
                                }
                            }

                            drawCell(getLerpCtx(i, o), Gz);
                        }
                    }
                }
                else {
                    generateAll();
                    lerpClear('', 'Select A', '', 'Select B', '');
                }

                resumeUserInput();
            }
            else {
                setTimeout(wait, 5);
            }
        };
        setTimeout(wait, 5);


        // Bind all button events
        var button = document.getElementById('b_all');
        button.onclick = changeAllCallback;
        for (var i = 0; i < cfg.ui.nids; ++i) {
            var button = document.getElementById('b_i' + String(i));
            button.onclick = changeIdCallbackFactory(i);
        }
        for (var o = 0; o < cfg.ui.nobs; ++o) {
            var button = document.getElementById('b_o' + String(o));
            button.onclick = changeObsCallbackFactory(o);
        }

        // Bind click events
        for (var i = 0; i < cfg.ui.nids; ++i) {
            for (var o = 0; o < cfg.ui.nobs; ++o) {
                var canvas = document.getElementById('c_i' + String(i) + '_o' + String(o));
                canvas.onmousedown = selectImageCallbackFactory(i, o);
            }
        }
    };

    // DOM load callbacks
    if (document.addEventListener) document.addEventListener("DOMContentLoaded", domReady, false);
    else if (document.attachEvent) document.attachEvent("onreadystatechange", domReady);
    else window.onload = domReady;

    // Exports
    sdgan.ui = {};

})(window.deeplearn, window.sdgan);
