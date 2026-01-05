let wasm;

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

let WASM_VECTOR_LEN = 0;

const __wbindgen_enum_BinaryType = ["blob", "arraybuffer"];

const BciAppFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bciapp_free(ptr >>> 0, 1));

const BciVizPipelineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bcivizpipeline_free(ptr >>> 0, 1));

const ChannelStatsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_channelstats_free(ptr >>> 0, 1));

const DemoRunnerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_demorunner_free(ptr >>> 0, 1));

const DisplaySettingsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_displaysettings_free(ptr >>> 0, 1));

const EdaRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_edarenderer_free(ptr >>> 0, 1));

const EmgRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_emgrenderer_free(ptr >>> 0, 1));

const FnirsMapRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fnirsmaprenderer_free(ptr >>> 0, 1));

const MultiDeviceDashboardFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_multidevicedashboard_free(ptr >>> 0, 1));

const PlaybackControllerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_playbackcontroller_free(ptr >>> 0, 1));

const SimulationGeneratorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_simulationgenerator_free(ptr >>> 0, 1));

const SnsVizAppFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_snsvizapp_free(ptr >>> 0, 1));

const SnsWebAppFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_snswebapp_free(ptr >>> 0, 1));

const StimControlPanelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_stimcontrolpanel_free(ptr >>> 0, 1));

const StimPresetFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_stimpreset_free(ptr >>> 0, 1));

const StreamConfigBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_streamconfigbuilder_free(ptr >>> 0, 1));

const TimeseriesRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_timeseriesrenderer_free(ptr >>> 0, 1));

const TopomapRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_topomaprenderer_free(ptr >>> 0, 1));

const VrPreviewRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vrpreviewrenderer_free(ptr >>> 0, 1));

const WebDeployConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_webdeployconfig_free(ptr >>> 0, 1));

/**
 * BCI visualization application state
 */
export class BciApp {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BciAppFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bciapp_free(ptr, 0);
    }
    /**
     * Disconnect from data stream
     */
    disconnect() {
        wasm.bciapp_disconnect(this.__wbg_ptr);
    }
    /**
     * Set VR preview EDA data
     * @param {Float32Array} scl
     * @param {number} arousal
     */
    set_vr_eda(scl, arousal) {
        const ptr0 = passArrayF32ToWasm0(scl, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.bciapp_set_vr_eda(this.__wbg_ptr, ptr0, len0, arousal);
    }
    /**
     * Set VR preview EMG data
     * @param {Float32Array} rms
     * @param {number} valence
     * @param {number} arousal
     */
    set_vr_emg(rms, valence, arousal) {
        const ptr0 = passArrayF32ToWasm0(rms, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.bciapp_set_vr_emg(this.__wbg_ptr, ptr0, len0, valence, arousal);
    }
    /**
     * Check if connected
     * @returns {boolean}
     */
    is_connected() {
        const ret = wasm.bciapp_is_connected(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Push raw EDA values directly (4 sites in µS)
     * @param {Float32Array} sites
     */
    push_eda_raw(sites) {
        const ptr0 = passArrayF32ToWasm0(sites, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_push_eda_raw(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Push raw EEG values directly (8 channels in µV)
     * @param {Float32Array} channels
     */
    push_eeg_raw(channels) {
        const ptr0 = passArrayF32ToWasm0(channels, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_push_eeg_raw(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Push EMG RMS values (8 channels in µV)
     * @param {Float32Array} channels
     */
    push_emg_rms(channels) {
        const ptr0 = passArrayF32ToWasm0(channels, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_push_emg_rms(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set VR preview fNIRS data
     * @param {Float32Array} hbo
     * @param {Float32Array} hbr
     */
    set_vr_fnirs(hbo, hbr) {
        const ptr0 = passArrayF32ToWasm0(hbo, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(hbr, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.bciapp_set_vr_fnirs(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Get the EDA canvas element
     * @returns {HTMLCanvasElement}
     */
    get_eda_canvas() {
        const ret = wasm.bciapp_get_eda_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the EMG canvas element
     * @returns {HTMLCanvasElement}
     */
    get_emg_canvas() {
        const ret = wasm.bciapp_get_emg_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Push raw fNIRS values directly
     * @param {Float32Array} hbo2
     * @param {Float32Array} hbr
     */
    push_fnirs_raw(hbo2, hbr) {
        const ptr0 = passArrayF32ToWasm0(hbo2, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(hbr, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_push_fnirs_raw(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get stimulation parameters from control panel as JSON
     * @returns {any}
     */
    get_stim_params() {
        const ret = wasm.bciapp_get_stim_params(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Set stimulation parameters from JSON
     * @param {any} params_js
     */
    set_stim_params(params_js) {
        const ret = wasm.bciapp_set_stim_params(this.__wbg_ptr, params_js);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Tick VR preview animation
     * @param {number} dt
     */
    tick_vr_preview(dt) {
        wasm.bciapp_tick_vr_preview(this.__wbg_ptr, dt);
    }
    /**
     * Get the fNIRS map canvas element
     * @returns {HTMLCanvasElement}
     */
    get_fnirs_canvas() {
        const ret = wasm.bciapp_get_fnirs_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Process incoming EEG data (raw protocol bytes)
     *
     * Parses a packet buffer and extracts EEG sample data.
     * @param {Uint8Array} data
     */
    process_eeg_data(data) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_process_eeg_data(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set VR preview EEG band power
     * @param {number} delta
     * @param {number} theta
     * @param {number} alpha
     * @param {number} beta
     * @param {number} gamma
     */
    set_vr_eeg_bands(delta, theta, alpha, beta, gamma) {
        wasm.bciapp_set_vr_eeg_bands(this.__wbg_ptr, delta, theta, alpha, beta, gamma);
    }
    /**
     * Get the topomap canvas element
     * @returns {HTMLCanvasElement}
     */
    get_topomap_canvas() {
        const ret = wasm.bciapp_get_topomap_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Process incoming fNIRS data (raw protocol bytes)
     *
     * Parses a packet buffer and extracts fNIRS sample data.
     * @param {Uint8Array} data
     */
    process_fnirs_data(data) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_process_fnirs_data(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set VR preview fingerprint state
     * @param {number} similarity
     * @param {string} target
     * @param {string} modality
     */
    set_vr_fingerprint(similarity, target, modality) {
        const ptr0 = passStringToWasm0(target, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(modality, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.bciapp_set_vr_fingerprint(this.__wbg_ptr, similarity, ptr0, len0, ptr1, len1);
    }
    /**
     * Push EDA decomposed values (SCL/SCR)
     *
     * `is_scr_peak` is a bitmask indicating which sites have an SCR peak.
     * @param {Float32Array} scl
     * @param {Float32Array} scr
     * @param {number} is_scr_peak
     */
    push_eda_decomposed(scl, scr, is_scr_peak) {
        const ptr0 = passArrayF32ToWasm0(scl, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(scr, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_push_eda_decomposed(this.__wbg_ptr, ptr0, len0, ptr1, len1, is_scr_peak);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get the timeseries canvas element
     * @returns {HTMLCanvasElement}
     */
    get_timeseries_canvas() {
        const ret = wasm.bciapp_get_timeseries_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the VR preview canvas element
     * @returns {HTMLCanvasElement}
     */
    get_vr_preview_canvas() {
        const ret = wasm.bciapp_get_vr_preview_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set VR preview EEG topography
     * @param {Float32Array} values
     */
    set_vr_eeg_topography(values) {
        const ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.bciapp_set_vr_eeg_topography(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Create a new BCI application instance
     */
    constructor() {
        const ret = wasm.bciapp_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        BciAppFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Render all visualizations
     */
    render() {
        const ret = wasm.bciapp_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Connect to WebSocket data stream
     * @param {string} url
     */
    connect(url) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bciapp_connect(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) BciApp.prototype[Symbol.dispose] = BciApp.prototype.free;

/**
 * Real-time BCI to visualization pipeline
 */
export class BciVizPipeline {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(BciVizPipeline.prototype);
        obj.__wbg_ptr = ptr;
        BciVizPipelineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BciVizPipelineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bcivizpipeline_free(ptr, 0);
    }
    /**
     * Disconnect from BCI data stream
     */
    disconnect() {
        wasm.bcivizpipeline_disconnect(this.__wbg_ptr);
    }
    /**
     * Process incoming message (called from JS callback)
     * @param {Uint8Array} data
     */
    on_message(data) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.bcivizpipeline_on_message(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Clear error state
     */
    clear_error() {
        wasm.bcivizpipeline_clear_error(this.__wbg_ptr);
    }
    /**
     * Check if connected
     * @returns {boolean}
     */
    is_connected() {
        const ret = wasm.bcivizpipeline_is_connected(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Set the colormap for activation display
     * @param {string} colormap
     */
    set_colormap(colormap) {
        const ptr0 = passStringToWasm0(colormap, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.bcivizpipeline_set_colormap(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set smoothing factor (0.0-1.0)
     * @param {number} factor
     */
    set_smoothing(factor) {
        wasm.bcivizpipeline_set_smoothing(this.__wbg_ptr, factor);
    }
    /**
     * Get activation value for a specific receptor
     * @param {number} mesh_id
     * @param {number} receptor_index
     * @returns {number}
     */
    get_activation(mesh_id, receptor_index) {
        const ret = wasm.bcivizpipeline_get_activation(this.__wbg_ptr, mesh_id, receptor_index);
        return ret;
    }
    /**
     * Render activation heatmap to pixel buffer
     * @param {number} mesh_id
     * @param {number} width
     * @param {number} height
     * @returns {Uint8Array}
     */
    render_heatmap(mesh_id, width, height) {
        const ret = wasm.bcivizpipeline_render_heatmap(this.__wbg_ptr, mesh_id, width, height);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Get number of buffered samples
     * @returns {number}
     */
    buffered_samples() {
        const ret = wasm.bcivizpipeline_buffered_samples(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get all activations as flat array [mesh_id, receptor_idx, activation, ...]
     * @returns {Float32Array}
     */
    get_all_activations() {
        const ret = wasm.bcivizpipeline_get_all_activations(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Initialize receptor activations for a scene
     * @param {number} num_receptors
     * @param {number} mesh_id
     */
    initialize_for_scene(num_receptors, mesh_id) {
        wasm.bcivizpipeline_initialize_for_scene(this.__wbg_ptr, num_receptors, mesh_id);
    }
    /**
     * Set activation value range
     * @param {number} min_val
     * @param {number} max_val
     */
    set_activation_range(min_val, max_val) {
        wasm.bcivizpipeline_set_activation_range(this.__wbg_ptr, min_val, max_val);
    }
    /**
     * Create new pipeline with default configuration
     */
    constructor() {
        const ret = wasm.bcivizpipeline_new();
        this.__wbg_ptr = ret >>> 0;
        BciVizPipelineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Process buffered samples and update activations
     * @param {bigint} timestamp_us
     * @returns {boolean}
     */
    update(timestamp_us) {
        const ret = wasm.bcivizpipeline_update(this.__wbg_ptr, timestamp_us);
        return ret !== 0;
    }
    /**
     * Connect to BCI data stream
     */
    connect() {
        const ret = wasm.bcivizpipeline_connect(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Create pipeline with custom WebSocket URL
     * @param {string} url
     * @returns {BciVizPipeline}
     */
    static with_url(url) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bcivizpipeline_with_url(ptr0, len0);
        return BciVizPipeline.__wrap(ret);
    }
    /**
     * Get last error message
     * @returns {string | undefined}
     */
    get_error() {
        const ret = wasm.bcivizpipeline_get_error(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
}
if (Symbol.dispose) BciVizPipeline.prototype[Symbol.dispose] = BciVizPipeline.prototype.free;

/**
 * Statistics for a single channel
 */
export class ChannelStats {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(ChannelStats.prototype);
        obj.__wbg_ptr = ptr;
        ChannelStatsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ChannelStatsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_channelstats_free(ptr, 0);
    }
    /**
     * Mean value (µV)
     * @returns {number}
     */
    get mean() {
        const ret = wasm.__wbg_get_channelstats_mean(this.__wbg_ptr);
        return ret;
    }
    /**
     * Mean value (µV)
     * @param {number} arg0
     */
    set mean(arg0) {
        wasm.__wbg_set_channelstats_mean(this.__wbg_ptr, arg0);
    }
    /**
     * Standard deviation (µV)
     * @returns {number}
     */
    get std_dev() {
        const ret = wasm.__wbg_get_channelstats_std_dev(this.__wbg_ptr);
        return ret;
    }
    /**
     * Standard deviation (µV)
     * @param {number} arg0
     */
    set std_dev(arg0) {
        wasm.__wbg_set_channelstats_std_dev(this.__wbg_ptr, arg0);
    }
    /**
     * Minimum value (µV)
     * @returns {number}
     */
    get min() {
        const ret = wasm.__wbg_get_channelstats_min(this.__wbg_ptr);
        return ret;
    }
    /**
     * Minimum value (µV)
     * @param {number} arg0
     */
    set min(arg0) {
        wasm.__wbg_set_channelstats_min(this.__wbg_ptr, arg0);
    }
    /**
     * Maximum value (µV)
     * @returns {number}
     */
    get max() {
        const ret = wasm.__wbg_get_channelstats_max(this.__wbg_ptr);
        return ret;
    }
    /**
     * Maximum value (µV)
     * @param {number} arg0
     */
    set max(arg0) {
        wasm.__wbg_set_channelstats_max(this.__wbg_ptr, arg0);
    }
    /**
     * Number of samples
     * @returns {number}
     */
    get sample_count() {
        const ret = wasm.__wbg_get_channelstats_sample_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of samples
     * @param {number} arg0
     */
    set sample_count(arg0) {
        wasm.__wbg_set_channelstats_sample_count(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) ChannelStats.prototype[Symbol.dispose] = ChannelStats.prototype.free;

/**
 * Color scheme for visualizations
 * @enum {0 | 1 | 2 | 3 | 4 | 5}
 */
export const ColorScheme = Object.freeze({
    /**
     * Blue to red gradient (default for EEG)
     */
    BlueRed: 0, "0": "BlueRed",
    /**
     * Red to blue gradient (inverse)
     */
    RedBlue: 1, "1": "RedBlue",
    /**
     * Green to red gradient (for fNIRS HbO2)
     */
    GreenRed: 2, "2": "GreenRed",
    /**
     * Blue to yellow gradient (for fNIRS HbR)
     */
    BlueYellow: 3, "3": "BlueYellow",
    /**
     * Viridis colormap
     */
    Viridis: 4, "4": "Viridis",
    /**
     * Plasma colormap
     */
    Plasma: 5, "5": "Plasma",
});

/**
 * Demo data runner that generates continuous synthetic BCI data
 */
export class DemoRunner {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DemoRunnerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_demorunner_free(ptr, 0);
    }
    /**
     * Check if demo is running
     * @returns {boolean}
     */
    is_running() {
        const ret = wasm.demorunner_is_running(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Generate EDA values (SCL and SCR for 4 sites)
     * Returns 8 values: [SCL_0..3, SCR_0..3]
     * @returns {Float32Array}
     */
    generate_eda() {
        const ret = wasm.demorunner_generate_eda(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Generate EEG sample for current time
     * Returns 8 channel values in µV
     * @returns {Float32Array}
     */
    generate_eeg() {
        const ret = wasm.demorunner_generate_eeg(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Generate EMG RMS values (8 facial muscles)
     * @returns {Float32Array}
     */
    generate_emg() {
        const ret = wasm.demorunner_generate_emg(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Set the demo scenario
     * @param {DemoScenario} scenario
     */
    set_scenario(scenario) {
        wasm.demorunner_set_scenario(this.__wbg_ptr, scenario);
    }
    /**
     * Get current scenario name
     * @returns {string}
     */
    scenario_name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.demorunner_scenario_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Generate fNIRS sample (HbO2, HbR for 4 optodes)
     * Returns 8 values: [HbO2_0, HbO2_1, HbO2_2, HbO2_3, HbR_0, HbR_1, HbR_2, HbR_3]
     * @returns {Float32Array}
     */
    generate_fnirs() {
        const ret = wasm.demorunner_generate_fnirs(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Get current simulation time in ms
     * @returns {number}
     */
    current_time_ms() {
        const ret = wasm.demorunner_current_time_ms(this.__wbg_ptr);
        return ret;
    }
    /**
     * Generate band power values for VR preview
     * @returns {Float32Array}
     */
    generate_band_power() {
        const ret = wasm.demorunner_generate_band_power(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Set EEG sample rate
     * @param {number} rate
     */
    set_eeg_sample_rate(rate) {
        wasm.demorunner_set_eeg_sample_rate(this.__wbg_ptr, rate);
    }
    /**
     * Check if fNIRS sample should be generated this tick
     * @returns {boolean}
     */
    should_sample_fnirs() {
        const ret = wasm.demorunner_should_sample_fnirs(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Set fNIRS sample rate
     * @param {number} rate
     */
    set_fnirs_sample_rate(rate) {
        wasm.demorunner_set_fnirs_sample_rate(this.__wbg_ptr, rate);
    }
    /**
     * Create a new demo runner
     */
    constructor() {
        const ret = wasm.demorunner_new();
        this.__wbg_ptr = ret >>> 0;
        DemoRunnerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Stop the demo
     */
    stop() {
        wasm.demorunner_stop(this.__wbg_ptr);
    }
    /**
     * Advance simulation and return EEG samples generated
     * Returns number of samples generated during this tick
     * @param {number} delta_ms
     * @returns {number}
     */
    tick(delta_ms) {
        const ret = wasm.demorunner_tick(this.__wbg_ptr, delta_ms);
        return ret >>> 0;
    }
    /**
     * Reset the demo to initial state
     */
    reset() {
        wasm.demorunner_reset(this.__wbg_ptr);
    }
    /**
     * Start the demo
     */
    start() {
        wasm.demorunner_start(this.__wbg_ptr);
    }
}
if (Symbol.dispose) DemoRunner.prototype[Symbol.dispose] = DemoRunner.prototype.free;

/**
 * Demo scenarios representing different brain states
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6}
 */
export const DemoScenario = Object.freeze({
    /**
     * Relaxed eyes-closed state (high alpha)
     */
    Relaxed: 0, "0": "Relaxed",
    /**
     * Focused attention task (high beta)
     */
    Focused: 1, "1": "Focused",
    /**
     * Drowsy/sleepy state (high theta)
     */
    Drowsy: 2, "2": "Drowsy",
    /**
     * Active thinking/problem solving (high gamma)
     */
    Thinking: 3, "3": "Thinking",
    /**
     * Meditation state (high alpha + theta)
     */
    Meditation: 4, "4": "Meditation",
    /**
     * Motor imagery (mu rhythm suppression)
     */
    MotorImagery: 5, "5": "MotorImagery",
    /**
     * Emotional response (asymmetric frontal)
     */
    EmotionalResponse: 6, "6": "EmotionalResponse",
});

/**
 * Connection status for a device.
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const DeviceStatus = Object.freeze({
    /**
     * Device discovered but not connected
     */
    Discovered: 0, "0": "Discovered",
    /**
     * Connecting to device
     */
    Connecting: 1, "1": "Connecting",
    /**
     * Connected and receiving data
     */
    Connected: 2, "2": "Connected",
    /**
     * Device disconnected
     */
    Disconnected: 3, "3": "Disconnected",
    /**
     * Error state
     */
    Error: 4, "4": "Error",
});

/**
 * Visualization display settings
 */
export class DisplaySettings {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DisplaySettingsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_displaysettings_free(ptr, 0);
    }
    /**
     * Get background color
     * @returns {string}
     */
    get_background() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.displaysettings_get_background(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Set background color
     * @param {string} color
     */
    set_background(color) {
        const ptr0 = passStringToWasm0(color, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.displaysettings_set_background(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get color scheme
     * @returns {ColorScheme}
     */
    get_color_scheme() {
        const ret = wasm.displaysettings_get_color_scheme(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set color scheme
     * @param {ColorScheme} scheme
     */
    set_color_scheme(scheme) {
        wasm.displaysettings_set_color_scheme(this.__wbg_ptr, scheme);
    }
    /**
     * Create default display settings
     */
    constructor() {
        const ret = wasm.displaysettings_new();
        this.__wbg_ptr = ret >>> 0;
        DisplaySettingsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Width in pixels
     * @returns {number}
     */
    get width() {
        const ret = wasm.__wbg_get_displaysettings_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Width in pixels
     * @param {number} arg0
     */
    set width(arg0) {
        wasm.__wbg_set_displaysettings_width(this.__wbg_ptr, arg0);
    }
    /**
     * Height in pixels
     * @returns {number}
     */
    get height() {
        const ret = wasm.__wbg_get_channelstats_sample_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Height in pixels
     * @param {number} arg0
     */
    set height(arg0) {
        wasm.__wbg_set_channelstats_sample_count(this.__wbg_ptr, arg0);
    }
    /**
     * Time window in seconds (for timeseries)
     * @returns {number}
     */
    get time_window_s() {
        const ret = wasm.__wbg_get_displaysettings_time_window_s(this.__wbg_ptr);
        return ret;
    }
    /**
     * Time window in seconds (for timeseries)
     * @param {number} arg0
     */
    set time_window_s(arg0) {
        wasm.__wbg_set_displaysettings_time_window_s(this.__wbg_ptr, arg0);
    }
    /**
     * Amplitude scale (µV per division)
     * @returns {number}
     */
    get amplitude_scale() {
        const ret = wasm.__wbg_get_displaysettings_amplitude_scale(this.__wbg_ptr);
        return ret;
    }
    /**
     * Amplitude scale (µV per division)
     * @param {number} arg0
     */
    set amplitude_scale(arg0) {
        wasm.__wbg_set_displaysettings_amplitude_scale(this.__wbg_ptr, arg0);
    }
    /**
     * Show grid lines
     * @returns {boolean}
     */
    get show_grid() {
        const ret = wasm.__wbg_get_displaysettings_show_grid(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Show grid lines
     * @param {boolean} arg0
     */
    set show_grid(arg0) {
        wasm.__wbg_set_displaysettings_show_grid(this.__wbg_ptr, arg0);
    }
    /**
     * Show channel labels
     * @returns {boolean}
     */
    get show_labels() {
        const ret = wasm.__wbg_get_displaysettings_show_labels(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Show channel labels
     * @param {boolean} arg0
     */
    set show_labels(arg0) {
        wasm.__wbg_set_displaysettings_show_labels(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) DisplaySettings.prototype[Symbol.dispose] = DisplaySettings.prototype.free;

/**
 * EDA visualization renderer
 */
export class EdaRenderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EdaRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_edarenderer_free(ptr, 0);
    }
    /**
     * Set arousal level
     * @param {number} arousal
     */
    set_arousal(arousal) {
        wasm.edarenderer_set_arousal(this.__wbg_ptr, arousal);
    }
    /**
     * Update display settings
     * @param {DisplaySettings} settings
     */
    set_settings(settings) {
        _assertClass(settings, DisplaySettings);
        var ptr0 = settings.__destroy_into_raw();
        wasm.edarenderer_set_settings(this.__wbg_ptr, ptr0);
    }
    /**
     * Toggle raw signal display
     * @param {boolean} show
     */
    set_show_raw(show) {
        wasm.edarenderer_set_show_raw(this.__wbg_ptr, show);
    }
    /**
     * Push decomposed SCL/SCR values
     *
     * `is_scr_peak` is a bitmask: bit 0 = site 0, bit 1 = site 1, etc.
     * @param {Float32Array} scl
     * @param {Float32Array} scr
     * @param {number} is_scr_peak
     */
    push_decomposed(scl, scr, is_scr_peak) {
        const ptr0 = passArrayF32ToWasm0(scl, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(scr, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.edarenderer_push_decomposed(this.__wbg_ptr, ptr0, len0, ptr1, len1, is_scr_peak);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set sample rate
     * @param {number} rate
     */
    set_sample_rate(rate) {
        wasm.edarenderer_set_sample_rate(this.__wbg_ptr, rate);
    }
    /**
     * Toggle arousal gauge
     * @param {boolean} show
     */
    set_show_arousal(show) {
        wasm.edarenderer_set_show_arousal(this.__wbg_ptr, show);
    }
    /**
     * Toggle decomposition display
     * @param {boolean} show
     */
    set_show_decomposition(show) {
        wasm.edarenderer_set_show_decomposition(this.__wbg_ptr, show);
    }
    /**
     * Create a new EDA renderer
     */
    constructor() {
        const ret = wasm.edarenderer_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        EdaRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Clear all buffers
     */
    clear() {
        wasm.edarenderer_clear(this.__wbg_ptr);
    }
    /**
     * Get the canvas element
     * @returns {HTMLCanvasElement}
     */
    canvas() {
        const ret = wasm.edarenderer_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the EDA visualization
     */
    render() {
        const ret = wasm.edarenderer_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Push raw conductance values (4 sites)
     * @param {Float32Array} conductance
     */
    push_raw(conductance) {
        const ptr0 = passArrayF32ToWasm0(conductance, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.edarenderer_push_raw(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) EdaRenderer.prototype[Symbol.dispose] = EdaRenderer.prototype.free;

/**
 * EMG visualization renderer
 */
export class EmgRenderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EmgRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_emgrenderer_free(ptr, 0);
    }
    /**
     * Update display settings
     * @param {DisplaySettings} settings
     */
    set_settings(settings) {
        _assertClass(settings, DisplaySettings);
        var ptr0 = settings.__destroy_into_raw();
        wasm.emgrenderer_set_settings(this.__wbg_ptr, ptr0);
    }
    /**
     * Toggle bar graph display
     * @param {boolean} show
     */
    set_show_bars(show) {
        wasm.emgrenderer_set_show_bars(this.__wbg_ptr, show);
    }
    /**
     * Set sample rate
     * @param {number} rate
     */
    set_sample_rate(rate) {
        wasm.emgrenderer_set_sample_rate(this.__wbg_ptr, rate);
    }
    /**
     * Toggle valence indicator
     * @param {boolean} show
     */
    set_show_valence(show) {
        wasm.emgrenderer_set_show_valence(this.__wbg_ptr, show);
    }
    /**
     * Toggle waveform display
     * @param {boolean} show
     */
    set_show_waveforms(show) {
        wasm.emgrenderer_set_show_waveforms(this.__wbg_ptr, show);
    }
    /**
     * Update valence and arousal scores
     * @param {number} valence
     * @param {number} arousal
     */
    set_valence_arousal(valence, arousal) {
        wasm.emgrenderer_set_valence_arousal(this.__wbg_ptr, valence, arousal);
    }
    /**
     * Create a new EMG renderer
     */
    constructor() {
        const ret = wasm.emgrenderer_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        EmgRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Clear all buffers
     */
    clear() {
        wasm.emgrenderer_clear(this.__wbg_ptr);
    }
    /**
     * Get the canvas element
     * @returns {HTMLCanvasElement}
     */
    canvas() {
        const ret = wasm.emgrenderer_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the EMG visualization
     */
    render() {
        const ret = wasm.emgrenderer_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Push EMG RMS values (8 channels)
     * @param {Float32Array} channels
     */
    push_rms(channels) {
        const ptr0 = passArrayF32ToWasm0(channels, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.emgrenderer_push_rms(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) EmgRenderer.prototype[Symbol.dispose] = EmgRenderer.prototype.free;

/**
 * Display mode for fNIRS visualization
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const FnirsDisplayMode = Object.freeze({
    /**
     * Show only HbO2 (oxygenated hemoglobin)
     */
    Hbo2Only: 0, "0": "Hbo2Only",
    /**
     * Show only HbR (deoxygenated hemoglobin)
     */
    HbrOnly: 1, "1": "HbrOnly",
    /**
     * Show both side by side
     */
    Both: 2, "2": "Both",
    /**
     * Show total hemoglobin (HbO2 + HbR)
     */
    Total: 3, "3": "Total",
    /**
     * Show oxygenation index (HbO2 / (HbO2 + HbR))
     */
    Oxygenation: 4, "4": "Oxygenation",
});

/**
 * fNIRS heatmap renderer
 */
export class FnirsMapRenderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FnirsMapRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fnirsmaprenderer_free(ptr, 0);
    }
    /**
     * Update with raw HbO2 and HbR values (µM)
     * @param {Float32Array} hbo2
     * @param {Float32Array} hbr
     */
    update_raw(hbo2, hbr) {
        const ptr0 = passArrayF32ToWasm0(hbo2, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(hbr, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.fnirsmaprenderer_update_raw(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set HbR value range
     * @param {number} min
     * @param {number} max
     */
    set_hbr_range(min, max) {
        wasm.fnirsmaprenderer_set_hbr_range(this.__wbg_ptr, min, max);
    }
    /**
     * Get current HbR values
     * @returns {Float32Array}
     */
    get_hbr_values() {
        const ret = wasm.fnirsmaprenderer_get_hbr_values(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Set HbO2 value range
     * @param {number} min
     * @param {number} max
     */
    set_hbo2_range(min, max) {
        wasm.fnirsmaprenderer_set_hbo2_range(this.__wbg_ptr, min, max);
    }
    /**
     * Get current HbO2 values
     * @returns {Float32Array}
     */
    get_hbo2_values() {
        const ret = wasm.fnirsmaprenderer_get_hbo2_values(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Toggle labels
     * @param {boolean} show
     */
    set_show_labels(show) {
        wasm.fnirsmaprenderer_set_show_labels(this.__wbg_ptr, show);
    }
    /**
     * Set display mode
     * @param {FnirsDisplayMode} mode
     */
    set_display_mode(mode) {
        wasm.fnirsmaprenderer_set_display_mode(this.__wbg_ptr, mode);
    }
    /**
     * Set channel positions
     * @param {Float32Array} positions
     */
    set_channel_positions(positions) {
        const ptr0 = passArrayF32ToWasm0(positions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.fnirsmaprenderer_set_channel_positions(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Create a new fNIRS heatmap renderer
     */
    constructor() {
        const ret = wasm.fnirsmaprenderer_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        FnirsMapRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Clear all values
     */
    clear() {
        wasm.fnirsmaprenderer_clear(this.__wbg_ptr);
    }
    /**
     * Get the canvas element
     * @returns {HTMLCanvasElement}
     */
    canvas() {
        const ret = wasm.fnirsmaprenderer_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the fNIRS map
     */
    render() {
        const ret = wasm.fnirsmaprenderer_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) FnirsMapRenderer.prototype[Symbol.dispose] = FnirsMapRenderer.prototype.free;

/**
 * GPU backend type
 * @enum {0 | 1 | 2}
 */
export const GpuBackend = Object.freeze({
    /**
     * WebGPU (preferred)
     */
    WebGpu: 0, "0": "WebGpu",
    /**
     * WebGL2 (fallback)
     */
    WebGl2: 1, "1": "WebGl2",
    /**
     * No GPU available (CPU only)
     */
    CpuOnly: 2, "2": "CpuOnly",
});

/**
 * Multi-device dashboard manager.
 */
export class MultiDeviceDashboard {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MultiDeviceDashboardFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_multidevicedashboard_free(ptr, 0);
    }
    /**
     * Add a device to the dashboard.
     * @param {string} id
     * @param {string} name
     */
    add_device(id, name) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.multidevicedashboard_add_device(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get device IDs as JSON array.
     * @returns {any}
     */
    device_ids() {
        const ret = wasm.multidevicedashboard_device_ids(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get the main canvas element.
     * @returns {HTMLCanvasElement}
     */
    get main_canvas() {
        const ret = wasm.multidevicedashboard_main_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the main visualization area.
     */
    render_main() {
        const ret = wasm.multidevicedashboard_render_main(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get number of devices.
     * @returns {number}
     */
    device_count() {
        const ret = wasm.multidevicedashboard_device_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Focus on a specific device.
     * @param {string} id
     */
    focus_device(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_focus_device(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Push EDA data to a device.
     * @param {string} device_id
     * @param {Float32Array} scl
     * @param {Float32Array} scr
     */
    push_eda_data(device_id, scl, scr) {
        const ptr0 = passStringToWasm0(device_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(scl, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(scr, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_push_eda_data(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
    }
    /**
     * Push EEG data to a device.
     * @param {string} device_id
     * @param {Float32Array} channels
     */
    push_eeg_data(device_id, channels) {
        const ptr0 = passStringToWasm0(device_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(channels, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_push_eeg_data(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Push EMG data to a device.
     * @param {string} device_id
     * @param {Float32Array} channels
     */
    push_emg_data(device_id, channels) {
        const ptr0 = passStringToWasm0(device_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(channels, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_push_emg_data(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Remove a device from the dashboard.
     * @param {string} id
     */
    remove_device(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_remove_device(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set view mode.
     * @param {ViewMode} mode
     */
    set_view_mode(mode) {
        wasm.multidevicedashboard_set_view_mode(this.__wbg_ptr, mode);
    }
    /**
     * Select devices for comparison.
     * @param {string[]} ids
     */
    select_devices(ids) {
        const ptr0 = passArrayJsValueToWasm0(ids, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_select_devices(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get all devices as JSON.
     * @returns {any}
     */
    get_all_devices() {
        const ret = wasm.multidevicedashboard_get_all_devices(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get device info as JSON.
     * @param {string} id
     * @returns {any}
     */
    get_device_info(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.multidevicedashboard_get_device_info(this.__wbg_ptr, ptr0, len0);
        return ret;
    }
    /**
     * Push fNIRS data to a device.
     * @param {string} device_id
     * @param {Float32Array} hbo
     * @param {Float32Array} hbr
     */
    push_fnirs_data(device_id, hbo, hbr) {
        const ptr0 = passStringToWasm0(device_id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(hbo, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(hbr, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_push_fnirs_data(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
    }
    /**
     * Set device name.
     * @param {string} id
     * @param {string} name
     */
    set_device_name(id, name) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_set_device_name(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Set device color.
     * @param {string} id
     * @param {number} r
     * @param {number} g
     * @param {number} b
     */
    set_device_color(id, r, g, b) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_set_device_color(this.__wbg_ptr, ptr0, len0, r, g, b);
    }
    /**
     * Set device muted state.
     * @param {string} id
     * @param {boolean} muted
     */
    set_device_muted(id, muted) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_set_device_muted(this.__wbg_ptr, ptr0, len0, muted);
    }
    /**
     * Get the device bar canvas element.
     * @returns {HTMLCanvasElement}
     */
    get device_bar_canvas() {
        const ret = wasm.multidevicedashboard_device_bar_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the device bar.
     */
    render_device_bar() {
        const ret = wasm.multidevicedashboard_render_device_bar(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Update device stimulation state.
     * @param {string} id
     * @param {boolean} stimulating
     * @param {number} similarity
     */
    update_stim_state(id, stimulating, similarity) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_update_stim_state(this.__wbg_ptr, ptr0, len0, stimulating, similarity);
    }
    /**
     * Update device status.
     * @param {string} id
     * @param {DeviceStatus} status
     */
    update_device_status(id, status) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_update_device_status(this.__wbg_ptr, ptr0, len0, status);
    }
    /**
     * Update device signal quality.
     * @param {string} id
     * @param {number} quality
     */
    update_signal_quality(id, quality) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.multidevicedashboard_update_signal_quality(this.__wbg_ptr, ptr0, len0, quality);
    }
    /**
     * Create a new multi-device dashboard.
     */
    constructor() {
        const ret = wasm.multidevicedashboard_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        MultiDeviceDashboardFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Render the complete dashboard.
     */
    render() {
        const ret = wasm.multidevicedashboard_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) MultiDeviceDashboard.prototype[Symbol.dispose] = MultiDeviceDashboard.prototype.free;

/**
 * Playback controller for recorded data
 */
export class PlaybackController {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PlaybackControllerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_playbackcontroller_free(ptr, 0);
    }
    /**
     * Get recording duration in microseconds
     * @returns {bigint}
     */
    duration_us() {
        const ret = wasm.playbackcontroller_duration_us(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Get next samples for current time
     * @param {bigint} current_time_us
     * @returns {Uint8Array}
     */
    get_samples(current_time_us) {
        const ret = wasm.playbackcontroller_get_samples(this.__wbg_ptr, current_time_us);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Enable/disable looping
     * @param {boolean} looping
     */
    set_looping(looping) {
        wasm.playbackcontroller_set_looping(this.__wbg_ptr, looping);
    }
    /**
     * Load recording from binary data
     * @param {Uint8Array} data
     * @returns {boolean}
     */
    load_recording(data) {
        const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.playbackcontroller_load_recording(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Create new playback controller
     */
    constructor() {
        const ret = wasm.playbackcontroller_new();
        this.__wbg_ptr = ret >>> 0;
        PlaybackControllerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Start playback
     * @param {bigint} current_time_us
     */
    play(current_time_us) {
        wasm.playbackcontroller_play(this.__wbg_ptr, current_time_us);
    }
    /**
     * Stop playback and reset
     */
    stop() {
        wasm.playbackcontroller_stop(this.__wbg_ptr);
    }
    /**
     * Pause playback
     */
    pause() {
        wasm.playbackcontroller_pause(this.__wbg_ptr);
    }
    /**
     * Get playback progress (0.0-1.0)
     * @returns {number}
     */
    progress() {
        const ret = wasm.playbackcontroller_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set playback speed
     * @param {number} speed
     */
    set_speed(speed) {
        wasm.playbackcontroller_set_speed(this.__wbg_ptr, speed);
    }
}
if (Symbol.dispose) PlaybackController.prototype[Symbol.dispose] = PlaybackController.prototype.free;

/**
 * Simulation data generator for testing
 */
export class SimulationGenerator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SimulationGeneratorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_simulationgenerator_free(ptr, 0);
    }
    /**
     * Set number of channels
     * @param {number} channels
     */
    set_channels(channels) {
        wasm.simulationgenerator_set_channels(this.__wbg_ptr, channels);
    }
    /**
     * Set simulation modality
     * @param {string} modality_str
     */
    set_modality(modality_str) {
        const ptr0 = passStringToWasm0(modality_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.simulationgenerator_set_modality(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set base oscillation frequency
     * @param {number} freq_hz
     */
    set_frequency(freq_hz) {
        wasm.simulationgenerator_set_frequency(this.__wbg_ptr, freq_hz);
    }
    /**
     * Create new simulation generator
     */
    constructor() {
        const ret = wasm.simulationgenerator_new();
        this.__wbg_ptr = ret >>> 0;
        SimulationGeneratorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Reset generator state
     */
    reset() {
        wasm.simulationgenerator_reset(this.__wbg_ptr);
    }
    /**
     * Generate samples for a time step
     * @param {bigint} delta_us
     * @returns {Uint8Array}
     */
    generate(delta_us) {
        const ret = wasm.simulationgenerator_generate(this.__wbg_ptr, delta_us);
        var v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Set noise amplitude
     * @param {number} amplitude
     */
    set_noise(amplitude) {
        wasm.simulationgenerator_set_noise(this.__wbg_ptr, amplitude);
    }
}
if (Symbol.dispose) SimulationGenerator.prototype[Symbol.dispose] = SimulationGenerator.prototype.free;

/**
 * WASM-exported SNS visualization application
 */
export class SnsVizApp {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SnsVizAppFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_snsvizapp_free(ptr, 0);
    }
    /**
     * Handle mouse up event
     * @param {number} _x
     * @param {number} _y
     */
    on_mouse_up(_x, _y) {
        wasm.snsvizapp_on_mouse_up(this.__wbg_ptr, _x, _y);
    }
    /**
     * Set colormap
     * @param {string} name
     */
    set_colormap(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.snsvizapp_set_colormap(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Handle mouse down event
     * @param {number} _x
     * @param {number} _y
     * @param {number} _button
     */
    on_mouse_down(_x, _y, _button) {
        wasm.snsvizapp_on_mouse_down(this.__wbg_ptr, _x, _y, _button);
    }
    /**
     * Handle mouse move (for orbit)
     * @param {number} dx
     * @param {number} dy
     * @param {number} buttons
     */
    on_mouse_move(dx, dy, buttons) {
        wasm.snsvizapp_on_mouse_move(this.__wbg_ptr, dx, dy, buttons);
    }
    /**
     * Render heatmap to pixel buffer (returns RGBA bytes)
     * @param {number} width
     * @param {number} height
     * @returns {Uint8Array | undefined}
     */
    render_heatmap(width, height) {
        const ret = wasm.snsvizapp_render_heatmap(this.__wbg_ptr, width, height);
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Get mesh vertex count
     * @returns {number}
     */
    get_vertex_count() {
        const ret = wasm.snsvizapp_get_vertex_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Load retina mesh (visual system)
     *
     * # Arguments
     * * `eye` - "left" or "right"
     * * `view` - "flat" (unrolled), "curved" (anatomical), or "cortex" (V1 mapping)
     * * `detail` - Level of detail (1-10, default 5)
     * @param {string} eye
     * @param {string} view
     * @param {number} detail
     */
    load_retina_mesh(eye, view, detail) {
        const ptr0 = passStringToWasm0(eye, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(view, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.snsvizapp_load_retina_mesh(this.__wbg_ptr, ptr0, len0, ptr1, len1, detail);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Load tongue mesh
     */
    load_tongue_mesh() {
        const ret = wasm.snsvizapp_load_tongue_mesh(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Load cochlea mesh
     * @param {string} ear
     * @param {boolean} unrolled
     */
    load_cochlea_mesh(ear, unrolled) {
        const ptr0 = passStringToWasm0(ear, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snsvizapp_load_cochlea_mesh(this.__wbg_ptr, ptr0, len0, unrolled);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Load tactile mesh for a body region
     * @param {string} region_name
     */
    load_tactile_mesh(region_name) {
        const ptr0 = passStringToWasm0(region_name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snsvizapp_load_tactile_mesh(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get receptor count for selected mesh
     * @returns {number}
     */
    get_receptor_count() {
        const ret = wasm.snsvizapp_get_receptor_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get mesh triangle count
     * @returns {number}
     */
    get_triangle_count() {
        const ret = wasm.snsvizapp_get_triangle_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Update receptor activations (flat array, matches receptor order)
     * @param {Float32Array} activations
     */
    update_activations(activations) {
        const ptr0 = passArrayF32ToWasm0(activations, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.snsvizapp_update_activations(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Load olfactory mesh (smell system)
     *
     * # Arguments
     * * `view` - "epithelium" (receptor neurons), "bulb" (glomeruli), or "combined"
     * * `detail` - Level of detail (1-10, default 5)
     * @param {string} view
     * @param {number} detail
     */
    load_olfactory_mesh(view, detail) {
        const ptr0 = passStringToWasm0(view, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snsvizapp_load_olfactory_mesh(this.__wbg_ptr, ptr0, len0, detail);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set activation range for colormap
     * @param {number} min_val
     * @param {number} max_val
     */
    set_activation_range(min_val, max_val) {
        wasm.snsvizapp_set_activation_range(this.__wbg_ptr, min_val, max_val);
    }
    /**
     * Create a new SNS visualization app
     * @param {number} width
     * @param {number} height
     */
    constructor(width, height) {
        const ret = wasm.snsvizapp_new(width, height);
        this.__wbg_ptr = ret >>> 0;
        SnsVizAppFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Handle mouse wheel (for zoom)
     * @param {number} delta
     */
    on_wheel(delta) {
        wasm.snsvizapp_on_wheel(this.__wbg_ptr, delta);
    }
}
if (Symbol.dispose) SnsVizApp.prototype[Symbol.dispose] = SnsVizApp.prototype.free;

/**
 * Main web application entry point
 */
export class SnsWebApp {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SnsWebAppFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_snswebapp_free(ptr, 0);
    }
    /**
     * Check if running
     * @returns {boolean}
     */
    is_running() {
        const ret = wasm.snswebapp_is_running(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Clear error state
     */
    clear_error() {
        wasm.snswebapp_clear_error(this.__wbg_ptr);
    }
    /**
     * Connect to BCI data stream
     * @param {string} url
     */
    connect_bci(url) {
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snswebapp_connect_bci(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Handle mouse up
     * @param {number} x
     * @param {number} y
     */
    on_mouse_up(x, y) {
        wasm.snswebapp_on_mouse_up(this.__wbg_ptr, x, y);
    }
    /**
     * Set colormap
     * @param {string} colormap
     */
    set_colormap(colormap) {
        const ptr0 = passStringToWasm0(colormap, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.snswebapp_set_colormap(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Handle mouse down
     * @param {number} x
     * @param {number} y
     * @param {number} button
     */
    on_mouse_down(x, y, button) {
        wasm.snswebapp_on_mouse_down(this.__wbg_ptr, x, y, button);
    }
    /**
     * Handle mouse move
     * @param {number} x
     * @param {number} y
     */
    on_mouse_move(x, y) {
        wasm.snswebapp_on_mouse_move(this.__wbg_ptr, x, y);
    }
    /**
     * Disconnect from BCI stream
     */
    disconnect_bci() {
        wasm.snswebapp_disconnect_bci(this.__wbg_ptr);
    }
    /**
     * Get GPU backend name
     * @returns {string}
     */
    get_backend_name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.snswebapp_get_backend_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Check BCI connection status
     * @returns {boolean}
     */
    is_bci_connected() {
        const ret = wasm.bcivizpipeline_is_connected(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Load tongue mesh
     * @returns {boolean}
     */
    load_tongue_mesh() {
        const ret = wasm.snswebapp_load_tongue_mesh(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Load cochlea mesh
     * @param {string} ear
     * @returns {boolean}
     */
    load_cochlea_mesh(ear) {
        const ptr0 = passStringToWasm0(ear, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snswebapp_load_cochlea_mesh(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Load tactile mesh
     * @param {string} region
     * @returns {boolean}
     */
    load_tactile_mesh(region) {
        const ptr0 = passStringToWasm0(region, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.snswebapp_load_tactile_mesh(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Create new web application
     */
    constructor() {
        const ret = wasm.snswebapp_new();
        this.__wbg_ptr = ret >>> 0;
        SnsWebAppFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Initialize with configuration
     * @param {WebDeployConfig} config
     * @returns {GpuBackend}
     */
    init(config) {
        _assertClass(config, WebDeployConfig);
        var ptr0 = config.__destroy_into_raw();
        const ret = wasm.snswebapp_init(this.__wbg_ptr, ptr0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Stop animation loop
     */
    stop() {
        wasm.snswebapp_stop(this.__wbg_ptr);
    }
    /**
     * Start animation loop
     */
    start() {
        const ret = wasm.snswebapp_start(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Update frame (called from animation loop)
     * @param {number} timestamp_ms
     * @returns {boolean}
     */
    update(timestamp_ms) {
        const ret = wasm.snswebapp_update(this.__wbg_ptr, timestamp_ms);
        return ret !== 0;
    }
    /**
     * Get current FPS
     * @returns {number}
     */
    get_fps() {
        const ret = wasm.snswebapp_get_fps(this.__wbg_ptr);
        return ret;
    }
    /**
     * Handle mouse wheel
     * @param {number} delta
     */
    on_wheel(delta) {
        wasm.snswebapp_on_wheel(this.__wbg_ptr, delta);
    }
    /**
     * Get last error message
     * @returns {string | undefined}
     */
    get_error() {
        const ret = wasm.snswebapp_get_error(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
}
if (Symbol.dispose) SnsWebApp.prototype[Symbol.dispose] = SnsWebApp.prototype.free;

/**
 * Stimulation control panel
 */
export class StimControlPanel {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StimControlPanelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_stimcontrolpanel_free(ptr, 0);
    }
    /**
     * Get remaining time in milliseconds
     * @returns {number}
     */
    remaining_ms() {
        const ret = wasm.stimcontrolpanel_remaining_ms(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Set duration in milliseconds
     * @param {number} duration_ms
     */
    set_duration(duration_ms) {
        const ret = wasm.stimcontrolpanel_set_duration(this.__wbg_ptr, duration_ms);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Get parameters as JSON
     * @returns {any}
     */
    get_params_js() {
        const ret = wasm.stimcontrolpanel_get_params_js(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Set amplitude in microamps
     * @param {number} amplitude_ua
     */
    set_amplitude(amplitude_ua) {
        const ret = wasm.stimcontrolpanel_set_amplitude(this.__wbg_ptr, amplitude_ua);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set frequency for tACS (Hz)
     * @param {number} frequency_hz
     */
    set_frequency(frequency_hz) {
        const ret = wasm.stimcontrolpanel_set_frequency(this.__wbg_ptr, frequency_hz);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set parameters from JSON
     * @param {any} params_js
     */
    set_params_js(params_js) {
        const ret = wasm.stimcontrolpanel_set_params_js(this.__wbg_ptr, params_js);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Emergency stop (immediate)
     */
    emergency_stop() {
        wasm.stimcontrolpanel_emergency_stop(this.__wbg_ptr);
    }
    /**
     * Get last error message
     * @returns {string | undefined}
     */
    get_last_error() {
        const ret = wasm.stimcontrolpanel_get_last_error(this.__wbg_ptr);
        let v1;
        if (ret[0] !== 0) {
            v1 = getStringFromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        }
        return v1;
    }
    /**
     * Get current amplitude (accounting for ramp)
     * @returns {number}
     */
    current_amplitude() {
        const ret = wasm.stimcontrolpanel_current_amplitude(this.__wbg_ptr);
        return ret;
    }
    /**
     * Create a new control panel
     */
    constructor() {
        const ret = wasm.stimcontrolpanel_new();
        this.__wbg_ptr = ret >>> 0;
        StimControlPanelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Lock the panel
     */
    lock() {
        wasm.stimcontrolpanel_lock(this.__wbg_ptr);
    }
    /**
     * Stop stimulation
     */
    stop() {
        wasm.stimcontrolpanel_stop(this.__wbg_ptr);
    }
    /**
     * Update elapsed time (call from animation frame)
     * @param {number} delta_ms
     */
    tick(delta_ms) {
        wasm.stimcontrolpanel_tick(this.__wbg_ptr, delta_ms);
    }
    /**
     * Start stimulation
     */
    start() {
        const ret = wasm.stimcontrolpanel_start(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Unlock the safety lock
     * @param {string} confirmation
     * @returns {boolean}
     */
    unlock(confirmation) {
        const ptr0 = passStringToWasm0(confirmation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stimcontrolpanel_unlock(this.__wbg_ptr, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Get elapsed time in milliseconds
     * @returns {number}
     */
    elapsed() {
        const ret = wasm.bcivizpipeline_buffered_samples(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get progress as percentage (0-100)
     * @returns {number}
     */
    progress() {
        const ret = wasm.stimcontrolpanel_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * Set stimulation mode
     * @param {string} mode_str
     */
    set_mode(mode_str) {
        const ptr0 = passStringToWasm0(mode_str, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.stimcontrolpanel_set_mode(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set ramp time in milliseconds
     * @param {number} ramp_ms
     */
    set_ramp(ramp_ms) {
        const ret = wasm.stimcontrolpanel_set_ramp(this.__wbg_ptr, ramp_ms);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Validate all parameters
     */
    validate() {
        const ret = wasm.stimcontrolpanel_validate(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Check if stimulation is active
     * @returns {boolean}
     */
    is_active() {
        const ret = wasm.stimcontrolpanel_is_active(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check if locked
     * @returns {boolean}
     */
    is_locked() {
        const ret = wasm.stimcontrolpanel_is_locked(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) StimControlPanel.prototype[Symbol.dispose] = StimControlPanel.prototype.free;

/**
 * Stimulation preset configurations
 */
export class StimPreset {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(StimPreset.prototype);
        obj.__wbg_ptr = ptr;
        StimPresetFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StimPresetFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_stimpreset_free(ptr, 0);
    }
    /**
     * Alpha tACS preset (10 Hz, 1 mA)
     * @returns {StimPreset}
     */
    static alpha_tacs() {
        const ret = wasm.stimpreset_alpha_tacs();
        return StimPreset.__wrap(ret);
    }
    /**
     * Gamma tACS preset (40 Hz, 1 mA)
     * @returns {StimPreset}
     */
    static gamma_tacs() {
        const ret = wasm.stimpreset_gamma_tacs();
        return StimPreset.__wrap(ret);
    }
    /**
     * Get preset description
     * @returns {string}
     */
    description() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.stimpreset_description(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Low intensity preset for beginners (0.5 mA, 10 min)
     * @returns {StimPreset}
     */
    static low_intensity() {
        const ret = wasm.stimpreset_low_intensity();
        return StimPreset.__wrap(ret);
    }
    /**
     * Standard tDCS anodal preset (1 mA, 20 min)
     * @returns {StimPreset}
     */
    static standard_anodal() {
        const ret = wasm.stimpreset_standard_anodal();
        return StimPreset.__wrap(ret);
    }
    /**
     * Standard tDCS cathodal preset (1 mA, 20 min)
     * @returns {StimPreset}
     */
    static standard_cathodal() {
        const ret = wasm.stimpreset_standard_cathodal();
        return StimPreset.__wrap(ret);
    }
    /**
     * Get preset name
     * @returns {string}
     */
    name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.stimpreset_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) StimPreset.prototype[Symbol.dispose] = StimPreset.prototype.free;

/**
 * JS-accessible stream configuration builder
 */
export class StreamConfigBuilder {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(StreamConfigBuilder.prototype);
        obj.__wbg_ptr = ptr;
        StreamConfigBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        StreamConfigBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_streamconfigbuilder_free(ptr, 0);
    }
    /**
     * Enable/disable EEG
     * @param {boolean} enable
     * @returns {StreamConfigBuilder}
     */
    enable_eeg(enable) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.streamconfigbuilder_enable_eeg(ptr, enable);
        return StreamConfigBuilder.__wrap(ret);
    }
    /**
     * Set fNIRS sample rate
     * @param {number} rate
     * @returns {StreamConfigBuilder}
     */
    fnirs_rate(rate) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.streamconfigbuilder_fnirs_rate(ptr, rate);
        return StreamConfigBuilder.__wrap(ret);
    }
    /**
     * Enable/disable fNIRS
     * @param {boolean} enable
     * @returns {StreamConfigBuilder}
     */
    enable_fnirs(enable) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.streamconfigbuilder_enable_fnirs(ptr, enable);
        return StreamConfigBuilder.__wrap(ret);
    }
    /**
     * Get EEG rate
     * @returns {number}
     */
    get_eeg_rate() {
        const ret = wasm.streamconfigbuilder_get_eeg_rate(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get fNIRS rate
     * @returns {number}
     */
    get_fnirs_rate() {
        const ret = wasm.multidevicedashboard_device_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create a new builder with defaults
     */
    constructor() {
        const ret = wasm.streamconfigbuilder_new();
        this.__wbg_ptr = ret >>> 0;
        StreamConfigBuilderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set WebSocket URL
     * @param {string} url
     * @returns {StreamConfigBuilder}
     */
    url(url) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passStringToWasm0(url, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.streamconfigbuilder_url(ptr, ptr0, len0);
        return StreamConfigBuilder.__wrap(ret);
    }
    /**
     * Get the URL (for display)
     * @returns {string}
     */
    get_url() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.streamconfigbuilder_get_url(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Set EEG sample rate
     * @param {number} rate
     * @returns {StreamConfigBuilder}
     */
    eeg_rate(rate) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.streamconfigbuilder_eeg_rate(ptr, rate);
        return StreamConfigBuilder.__wrap(ret);
    }
}
if (Symbol.dispose) StreamConfigBuilder.prototype[Symbol.dispose] = StreamConfigBuilder.prototype.free;

/**
 * EEG timeseries renderer
 */
export class TimeseriesRenderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TimeseriesRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_timeseriesrenderer_free(ptr, 0);
    }
    /**
     * Update display settings
     * @param {DisplaySettings} settings
     */
    set_settings(settings) {
        _assertClass(settings, DisplaySettings);
        var ptr0 = settings.__destroy_into_raw();
        wasm.timeseriesrenderer_set_settings(this.__wbg_ptr, ptr0);
    }
    /**
     * Set sample rate
     * @param {number} rate
     */
    set_sample_rate(rate) {
        wasm.timeseriesrenderer_set_sample_rate(this.__wbg_ptr, rate);
    }
    /**
     * Get statistics for a channel
     * @param {number} channel
     * @returns {ChannelStats | undefined}
     */
    get_channel_stats(channel) {
        const ret = wasm.timeseriesrenderer_get_channel_stats(this.__wbg_ptr, channel);
        return ret === 0 ? undefined : ChannelStats.__wrap(ret);
    }
    /**
     * Enable/disable a channel
     * @param {number} channel
     * @param {boolean} enabled
     */
    set_channel_enabled(channel, enabled) {
        wasm.timeseriesrenderer_set_channel_enabled(this.__wbg_ptr, channel, enabled);
    }
    /**
     * Create a new timeseries renderer
     */
    constructor() {
        const ret = wasm.timeseriesrenderer_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        TimeseriesRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Clear all buffers
     */
    clear() {
        wasm.timeseriesrenderer_clear(this.__wbg_ptr);
    }
    /**
     * Get the canvas element
     * @returns {HTMLCanvasElement}
     */
    canvas() {
        const ret = wasm.timeseriesrenderer_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the timeseries
     */
    render() {
        const ret = wasm.timeseriesrenderer_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Push raw sample data (8 channel values in µV)
     * @param {Float32Array} channels
     */
    push_raw(channels) {
        const ptr0 = passArrayF32ToWasm0(channels, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.timeseriesrenderer_push_raw(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) TimeseriesRenderer.prototype[Symbol.dispose] = TimeseriesRenderer.prototype.free;

/**
 * Topographic map renderer
 */
export class TopomapRenderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TopomapRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_topomaprenderer_free(ptr, 0);
    }
    /**
     * Update with raw channel values (µV)
     * @param {Float32Array} values
     */
    update_raw(values) {
        const ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.topomaprenderer_update_raw(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Set interpolation resolution
     * @param {number} resolution
     */
    set_resolution(resolution) {
        wasm.topomaprenderer_set_resolution(this.__wbg_ptr, resolution);
    }
    /**
     * Set value range for normalization
     * @param {number} min
     * @param {number} max
     */
    set_value_range(min, max) {
        wasm.topomaprenderer_set_value_range(this.__wbg_ptr, min, max);
    }
    /**
     * Set color scheme
     * @param {ColorScheme} scheme
     */
    set_color_scheme(scheme) {
        wasm.topomaprenderer_set_color_scheme(this.__wbg_ptr, scheme);
    }
    /**
     * Toggle contour lines
     * @param {boolean} show
     */
    set_show_contours(show) {
        wasm.topomaprenderer_set_show_contours(this.__wbg_ptr, show);
    }
    /**
     * Toggle electrode markers
     * @param {boolean} show
     */
    set_show_electrodes(show) {
        wasm.topomaprenderer_set_show_electrodes(this.__wbg_ptr, show);
    }
    /**
     * Create a new topographic map renderer
     */
    constructor() {
        const ret = wasm.topomaprenderer_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        TopomapRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the canvas element
     * @returns {HTMLCanvasElement}
     */
    canvas() {
        const ret = wasm.topomaprenderer_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the topographic map
     */
    render() {
        const ret = wasm.topomaprenderer_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) TopomapRenderer.prototype[Symbol.dispose] = TopomapRenderer.prototype.free;

/**
 * Dashboard view mode.
 * @enum {0 | 1 | 2 | 3}
 */
export const ViewMode = Object.freeze({
    /**
     * Single device full view
     */
    Single: 0, "0": "Single",
    /**
     * Two devices side by side
     */
    SideBySide: 1, "1": "SideBySide",
    /**
     * Multiple signals overlaid
     */
    Overlay: 2, "2": "Overlay",
    /**
     * Grid of all devices
     */
    Grid: 3, "3": "Grid",
});

/**
 * Integrated VR preview panel combining all biosignal modalities
 */
export class VrPreviewRenderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        VrPreviewRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vrpreviewrenderer_free(ptr, 0);
    }
    /**
     * Update stimulation state
     * @param {boolean} active
     * @param {number} progress
     */
    set_stimulation(active, progress) {
        wasm.vrpreviewrenderer_set_stimulation(this.__wbg_ptr, active, progress);
    }
    /**
     * Update EEG band power values
     * @param {number} delta
     * @param {number} theta
     * @param {number} alpha
     * @param {number} beta
     * @param {number} gamma
     */
    set_eeg_band_power(delta, theta, alpha, beta, gamma) {
        wasm.vrpreviewrenderer_set_eeg_band_power(this.__wbg_ptr, delta, theta, alpha, beta, gamma);
    }
    /**
     * Update EEG topography (8 channels)
     * @param {Float32Array} values
     */
    set_eeg_topography(values) {
        const ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.vrpreviewrenderer_set_eeg_topography(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Update neural fingerprint state
     * @param {number} similarity
     * @param {string} target
     * @param {string} modality
     */
    set_fingerprint_state(similarity, target, modality) {
        const ptr0 = passStringToWasm0(target, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(modality, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.vrpreviewrenderer_set_fingerprint_state(this.__wbg_ptr, similarity, ptr0, len0, ptr1, len1);
    }
    /**
     * Create a new VR preview renderer
     */
    constructor() {
        const ret = wasm.vrpreviewrenderer_new();
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        VrPreviewRendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Advance animation time
     * @param {number} dt
     */
    tick(dt) {
        wasm.vrpreviewrenderer_tick(this.__wbg_ptr, dt);
    }
    /**
     * Get the canvas element
     * @returns {HTMLCanvasElement}
     */
    canvas() {
        const ret = wasm.vrpreviewrenderer_canvas(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render the complete VR preview
     */
    render() {
        const ret = wasm.vrpreviewrenderer_render(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Update EDA values
     * @param {Float32Array} scl
     * @param {number} arousal
     */
    set_eda(scl, arousal) {
        const ptr0 = passArrayF32ToWasm0(scl, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.vrpreviewrenderer_set_eda(this.__wbg_ptr, ptr0, len0, arousal);
    }
    /**
     * Update EMG values
     * @param {Float32Array} rms
     * @param {number} valence
     * @param {number} arousal
     */
    set_emg(rms, valence, arousal) {
        const ptr0 = passArrayF32ToWasm0(rms, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.vrpreviewrenderer_set_emg(this.__wbg_ptr, ptr0, len0, valence, arousal);
    }
    /**
     * Update fNIRS hemoglobin values
     * @param {Float32Array} hbo
     * @param {Float32Array} hbr
     */
    set_fnirs(hbo, hbr) {
        const ptr0 = passArrayF32ToWasm0(hbo, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(hbr, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.vrpreviewrenderer_set_fnirs(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
}
if (Symbol.dispose) VrPreviewRenderer.prototype[Symbol.dispose] = VrPreviewRenderer.prototype.free;

/**
 * Web deployment configuration
 */
export class WebDeployConfig {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WebDeployConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_webdeployconfig_free(ptr, 0);
    }
    /**
     * Enable/disable high-DPI rendering
     * @param {boolean} enabled
     */
    set_high_dpi(enabled) {
        wasm.webdeployconfig_set_high_dpi(this.__wbg_ptr, enabled);
    }
    /**
     * Set canvas element ID
     * @param {string} id
     */
    set_canvas_id(id) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.webdeployconfig_set_canvas_id(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set target frame rate
     * @param {number} fps
     */
    set_target_fps(fps) {
        wasm.webdeployconfig_set_target_fps(this.__wbg_ptr, fps);
    }
    /**
     * Enable/disable performance monitor
     * @param {boolean} enabled
     */
    set_perf_monitor(enabled) {
        wasm.webdeployconfig_set_perf_monitor(this.__wbg_ptr, enabled);
    }
    /**
     * Enable/disable WebGPU preference
     * @param {boolean} prefer
     */
    set_prefer_webgpu(prefer) {
        wasm.webdeployconfig_set_prefer_webgpu(this.__wbg_ptr, prefer);
    }
    /**
     * Set background color
     * @param {string} color
     */
    set_background_color(color) {
        const ptr0 = passStringToWasm0(color, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.webdeployconfig_set_background_color(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Create new configuration with defaults
     */
    constructor() {
        const ret = wasm.webdeployconfig_new();
        this.__wbg_ptr = ret >>> 0;
        WebDeployConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set canvas size
     * @param {number} width
     * @param {number} height
     */
    set_size(width, height) {
        wasm.webdeployconfig_set_size(this.__wbg_ptr, width, height);
    }
}
if (Symbol.dispose) WebDeployConfig.prototype[Symbol.dispose] = WebDeployConfig.prototype.free;

/**
 * WASM-accessible demo fixture generator
 * @param {DemoScenario} scenario
 * @param {number} duration_s
 * @returns {string}
 */
export function generate_demo_fixture_json(scenario, duration_s) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.generate_demo_fixture_json(scenario, duration_s);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Initialize the WASM module
 */
export function init() {
    wasm.init();
}

/**
 * Load and parse a demo fixture from JSON
 * @param {string} json
 * @returns {any}
 */
export function parse_demo_fixture(json) {
    const ptr0 = passStringToWasm0(json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.parse_demo_fixture(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return takeFromExternrefTable0(ret[0]);
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_Error_52673b7de5a0ca89 = function(arg0, arg1) {
        const ret = Error(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_Number_2d1dcfcf4ec51736 = function(arg0) {
        const ret = Number(arg0);
        return ret;
    };
    imports.wbg.__wbg_String_8f0eb39a4a4c2f66 = function(arg0, arg1) {
        const ret = String(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_boolean_get_dea25b33882b895b = function(arg0) {
        const v = arg0;
        const ret = typeof(v) === 'boolean' ? v : undefined;
        return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
    };
    imports.wbg.__wbg___wbindgen_debug_string_adfb662ae34724b6 = function(arg0, arg1) {
        const ret = debugString(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_in_0d3e1e8f0c669317 = function(arg0, arg1) {
        const ret = arg0 in arg1;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_null_dfda7d66506c95b5 = function(arg0) {
        const ret = arg0 === null;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_object_ce774f3490692386 = function(arg0) {
        const val = arg0;
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_undefined_f6b95eab589e0269 = function(arg0) {
        const ret = arg0 === undefined;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_jsval_loose_eq_766057600fdd1b0d = function(arg0, arg1) {
        const ret = arg0 == arg1;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_number_get_9619185a74197f95 = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbg___wbindgen_string_get_a2a31e16edf96e42 = function(arg0, arg1) {
        const obj = arg1;
        const ret = typeof(obj) === 'string' ? obj : undefined;
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_arc_c46ca66b5ec2f1ac = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
        arg0.arc(arg1, arg2, arg3, arg4, arg5);
    }, arguments) };
    imports.wbg.__wbg_beginPath_08eae248f93ea32d = function(arg0) {
        arg0.beginPath();
    };
    imports.wbg.__wbg_call_abb4ff46ce38be40 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_close_1db3952de1b5b1cf = function() { return handleError(function (arg0) {
        arg0.close();
    }, arguments) };
    imports.wbg.__wbg_createElement_da4ed2b219560fc6 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.createElement(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_devicePixelRatio_390dee26c70aa30f = function(arg0) {
        const ret = arg0.devicePixelRatio;
        return ret;
    };
    imports.wbg.__wbg_document_5b745e82ba551ca5 = function(arg0) {
        const ret = arg0.document;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_drawImage_36aa227069c6c159 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
        arg0.drawImage(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }, arguments) };
    imports.wbg.__wbg_drawImage_65169eadc6cb661d = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        arg0.drawImage(arg1, arg2, arg3);
    }, arguments) };
    imports.wbg.__wbg_drawImage_f8cc14fd47e6101b = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
        arg0.drawImage(arg1, arg2, arg3, arg4, arg5);
    }, arguments) };
    imports.wbg.__wbg_ellipse_8fe237473fd39db1 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
        arg0.ellipse(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }, arguments) };
    imports.wbg.__wbg_fillRect_84131220403e26a4 = function(arg0, arg1, arg2, arg3, arg4) {
        arg0.fillRect(arg1, arg2, arg3, arg4);
    };
    imports.wbg.__wbg_fillText_56566d8049e84e17 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        arg0.fillText(getStringFromWasm0(arg1, arg2), arg3, arg4);
    }, arguments) };
    imports.wbg.__wbg_fill_dd0f756eea36e037 = function(arg0) {
        arg0.fill();
    };
    imports.wbg.__wbg_getContext_01f42b234e833f0a = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    }, arguments) };
    imports.wbg.__wbg_getElementById_e05488d2143c2b21 = function(arg0, arg1, arg2) {
        const ret = arg0.getElementById(getStringFromWasm0(arg1, arg2));
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_get_af9dab7e9603ea93 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(arg0, arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_get_with_ref_key_1dc361bd10053bfe = function(arg0, arg1) {
        const ret = arg0[arg1];
        return ret;
    };
    imports.wbg.__wbg_height_a07787f693c253d2 = function(arg0) {
        const ret = arg0.height;
        return ret;
    };
    imports.wbg.__wbg_instanceof_ArrayBuffer_f3320d2419cd0355 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof ArrayBuffer;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_CanvasRenderingContext2d_d070139aaac1459f = function(arg0) {
        let result;
        try {
            result = arg0 instanceof CanvasRenderingContext2D;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_HtmlCanvasElement_c4251b1b6a15edcc = function(arg0) {
        let result;
        try {
            result = arg0 instanceof HTMLCanvasElement;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Uint8Array_da54ccc9d3e09434 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Uint8Array;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Window_b5cf7783caa68180 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Window;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isSafeInteger_ae7d3f054d55fa16 = function(arg0) {
        const ret = Number.isSafeInteger(arg0);
        return ret;
    };
    imports.wbg.__wbg_length_22ac23eaec9d8053 = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_lineTo_4b884d8cebfc8c54 = function(arg0, arg1, arg2) {
        arg0.lineTo(arg1, arg2);
    };
    imports.wbg.__wbg_log_1d990106d99dacb7 = function(arg0) {
        console.log(arg0);
    };
    imports.wbg.__wbg_moveTo_36127921f1ca46a5 = function(arg0, arg1, arg2) {
        arg0.moveTo(arg1, arg2);
    };
    imports.wbg.__wbg_new_1ba21ce319a06297 = function() {
        const ret = new Object();
        return ret;
    };
    imports.wbg.__wbg_new_25f239778d6112b9 = function() {
        const ret = new Array();
        return ret;
    };
    imports.wbg.__wbg_new_6421f6084cc5bc5a = function(arg0) {
        const ret = new Uint8Array(arg0);
        return ret;
    };
    imports.wbg.__wbg_new_7c30d1f874652e62 = function() { return handleError(function (arg0, arg1) {
        const ret = new WebSocket(getStringFromWasm0(arg0, arg1));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_new_no_args_cb138f77cf6151ee = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_prototypesetcall_dfe9b766cdc1f1fd = function(arg0, arg1, arg2) {
        Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
    };
    imports.wbg.__wbg_quadraticCurveTo_97e814dec677b49c = function(arg0, arg1, arg2, arg3, arg4) {
        arg0.quadraticCurveTo(arg1, arg2, arg3, arg4);
    };
    imports.wbg.__wbg_restore_6486cb1a7aa3af7b = function(arg0) {
        arg0.restore();
    };
    imports.wbg.__wbg_rotate_4185d7f8614ba2d5 = function() { return handleError(function (arg0, arg1) {
        arg0.rotate(arg1);
    }, arguments) };
    imports.wbg.__wbg_save_b8767cfd2ee7f600 = function(arg0) {
        arg0.save();
    };
    imports.wbg.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
        arg0[arg1] = arg2;
    };
    imports.wbg.__wbg_set_7df433eea03a5c14 = function(arg0, arg1, arg2) {
        arg0[arg1 >>> 0] = arg2;
    };
    imports.wbg.__wbg_set_binaryType_73e8c75df97825f8 = function(arg0, arg1) {
        arg0.binaryType = __wbindgen_enum_BinaryType[arg1];
    };
    imports.wbg.__wbg_set_fillStyle_c9a0550307cd4671 = function(arg0, arg1, arg2) {
        arg0.fillStyle = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_set_fillStyle_ea371e123273908b = function(arg0, arg1) {
        arg0.fillStyle = arg1;
    };
    imports.wbg.__wbg_set_font_37c5ab71d0189314 = function(arg0, arg1, arg2) {
        arg0.font = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_set_globalAlpha_5b9512a71ef816b8 = function(arg0, arg1) {
        arg0.globalAlpha = arg1;
    };
    imports.wbg.__wbg_set_height_6f8f8ef4cb40e496 = function(arg0, arg1) {
        arg0.height = arg1 >>> 0;
    };
    imports.wbg.__wbg_set_lineWidth_feda4b79a15c660b = function(arg0, arg1) {
        arg0.lineWidth = arg1;
    };
    imports.wbg.__wbg_set_strokeStyle_697a576d2d3fbeaa = function(arg0, arg1, arg2) {
        arg0.strokeStyle = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_set_strokeStyle_857faae3a756ddf4 = function(arg0, arg1) {
        arg0.strokeStyle = arg1;
    };
    imports.wbg.__wbg_set_textAlign_5d82eb01e9d2291e = function(arg0, arg1, arg2) {
        arg0.textAlign = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_set_width_7ff7a22c6e9f423e = function(arg0, arg1) {
        arg0.width = arg1 >>> 0;
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_769e6b65d6557335 = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_60cf02db4de8e1c1 = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_08f5a74c69739274 = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_a8924b26aa92d024 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_strokeRect_31a396bc4462b669 = function(arg0, arg1, arg2, arg3, arg4) {
        arg0.strokeRect(arg1, arg2, arg3, arg4);
    };
    imports.wbg.__wbg_stroke_a18b81eb49ff370e = function(arg0) {
        arg0.stroke();
    };
    imports.wbg.__wbg_translate_5c51221dc69f0baa = function() { return handleError(function (arg0, arg1, arg2) {
        arg0.translate(arg1, arg2);
    }, arguments) };
    imports.wbg.__wbg_warn_6e567d0d926ff881 = function(arg0) {
        console.warn(arg0);
    };
    imports.wbg.__wbg_width_dd0cfe94d42f5143 = function(arg0) {
        const ret = arg0.width;
        return ret;
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return ret;
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('rootstar_bci_web_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
