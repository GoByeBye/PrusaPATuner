import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";


function loadLiveMapHooks({ fetchImpl } = {}) {
  const appUrl = new URL(
    "../src/prusa_pa_tuner/static/app.js",
    import.meta.url,
  );
  const source = `${readFileSync(appUrl, "utf8")}

globalThis.__liveMapTestHooks = {
  lmap,
  lmResetLive,
  lmapIngestPos,
  lmapIngestForce,
  lmapIngestAxisZ,
  pollLiveMapPrinterZ,
  renderLiveMapRun,
};
`;

  // app.js is deliberately a classic browser script with no build step.
  // Loading it in a VM keeps this regression tied to the shipped frontend
  // while avoiding a DOM package: its only top-level browser side effect is
  // registering the DOMContentLoaded callback, which the test need not run.
  const elements = new Map();
  const session = new Map();
  const context = vm.createContext({
    console,
    Plotly: { react() {} },
    fetch: fetchImpl || (async () => ({
      ok: true,
      async json() { return { polylines: [] }; },
    })),
    URLSearchParams,
    location: { search: "" },
    sessionStorage: {
      getItem(key) { return session.has(key) ? session.get(key) : null; },
      setItem(key, value) { session.set(key, String(value)); },
      removeItem(key) { session.delete(key); },
    },
    document: {
      addEventListener() {},
      getElementById(id) {
        if (!elements.has(id)) {
          elements.set(id, { textContent: "", value: "", style: {} });
        }
        return elements.get(id);
      },
    },
    setTimeout() { return 0; },
    clearTimeout() {},
    setInterval() { return 0; },
    clearInterval() {},
  });
  context.window = context;
  new vm.Script(source, { filename: appUrl.pathname }).runInContext(context);
  return context.__liveMapTestHooks;
}


function series(start, count, step, valueAt) {
  const t = [];
  const v = [];
  for (let i = 0; i < count; i++) {
    t.push(start + i * step);
    v.push(valueAt(i));
  }
  return { t, v };
}


function ingestPositions(hooks, t, x, y, z) {
  hooks.lmapIngestPos("x", t, x);
  hooks.lmapIngestPos("y", t, y);
  hooks.lmapIngestPos("z", t, z);
}


test("INDX Live Map excludes startup wipe and learns the first-layer Z offset", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;

  lmap.summary = {
    n_layers: 5,
    layer_zs: [0.2, 0.4, 0.6, 0.8, 1.0],
    bbox: [99, 111, 99, 101],
  };
  // Reset without starting the asynchronous geometry fetch. The test installs
  // the parsed first-layer segment directly, then switches to live mode.
  lmap.mode = "review";
  hooks.lmResetLive();
  lmap.mode = "live";
  lmap.followLive = false;
  lmap.zTrack.anchorSegments = [[100, 100, 110, 100]];

  // Merely uploading a preview must not consume telemetry from a different
  // print that was already running on the printer.
  const unrelated = series(-2, 20, 0.02, (i) => i);
  ingestPositions(
    hooks,
    unrelated.t,
    unrelated.v.map(() => 100),
    unrelated.v.map(() => 100),
    unrelated.v.map(() => 0.32),
  );
  hooks.lmapIngestForce(unrelated.t, unrelated.v.map(() => 777));
  assert.equal(lmap.posX.t.length, 0);
  assert.equal(lmap.liveSeen, 0);

  lmap.captureActive = true;

  // INDX pickup/cleaner motion visits Z values that are valid layers in a
  // taller print. It must still be ignored because it is nowhere near the
  // parsed first-layer extrusion path. Include firmware-style 168 mm spikes.
  const wipe = series(0, 60, 0.02, (i) => i);
  const wipeX = wipe.v.map(() => 242);
  const wipeY = wipe.v.map((i) => 205 + (i % 4) * 0.2);
  const wipeZ = wipe.v.map((i) => i % 7 === 0 ? 168 : (i < 30 ? 0.8 : 1.0));
  ingestPositions(hooks, wipe.t, wipeX, wipeY, wipeZ);
  hooks.lmapIngestForce(wipe.t, wipe.v.map(() => 900));

  assert.equal(lmap.zTrack.armed, false);
  assert.equal(lmap.liveSeen, 0);
  assert.deepEqual(Object.keys(lmap.livePaths), []);

  // The real first layer follows the parsed path at raw Z=0.32: slicer Z=0.20
  // plus a constant +0.12 mm tool/mesh-frame offset. A minority of 168 mm
  // samples must not poison the rolling median or the learned offset.
  const first = series(2, 70, 0.02, (i) => i);
  const firstX = first.v.map((i) => 100 + i * 0.08);
  const firstY = first.v.map(() => 100);
  const firstZ = first.v.map((i) => i % 5 === 0 ? 168 : 0.32);
  ingestPositions(hooks, first.t, firstX, firstY, firstZ);
  hooks.lmapIngestForce(first.t, first.v.map(() => 42));

  assert.equal(lmap.zTrack.armed, true);
  assert.ok(Math.abs(lmap.zTrack.offset - 0.12) < 1e-9);
  assert.equal(lmap.liveLayer, 0);
  assert.ok(lmap.liveSeen > 0);
  assert.ok(lmap.livePaths[0].x.every((x) => x >= 99 && x <= 111));
  assert.ok(lmap.livePaths[0].y.every((y) => y >= 99 && y <= 101));
  assert.ok(!lmap.livePaths[0].x.includes(242));

  const firstLayerSeen = lmap.livePaths[0].seen;

  // The same +0.12 mm offset on slicer layer Z=0.40 must advance exactly one
  // layer despite another minority of sentinel spikes.
  const second = series(4, 130, 0.02, (i) => i);
  const secondX = second.v.map((i) => 100 + i * 0.05);
  const secondY = second.v.map(() => 100);
  const secondZ = second.v.map((i) => i % 5 === 0 ? 168 : 0.52);
  ingestPositions(hooks, second.t, secondX, secondY, secondZ);
  hooks.lmapIngestForce(second.t, second.v.map(() => 55));

  assert.equal(lmap.liveLayer, 1);
  assert.equal(lmap.livePaths[0].seen, firstLayerSeen);
  assert.ok(lmap.livePaths[1].seen > 0);
});


test("INDX +1.2 mm raw-Z offset maps physical 1.0 mm to layer 5, not 11", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;

  lmap.summary = {
    n_layers: 12,
    layer_zs: Array.from({ length: 12 }, (_, i) => (i + 1) * 0.2),
    bbox: [99, 121, 99, 101],
  };
  lmap.mode = "review";
  hooks.lmResetLive();
  lmap.mode = "live";
  lmap.captureActive = true;
  lmap.followLive = false;
  lmap.zTrack.anchorSegments = [[100, 100, 120, 100]];

  const ingestLayer = (start, plannedZ, x0 = 100) => {
    const samples = series(start, 130, 0.02, (i) => i);
    const x = samples.v.map((i) => x0 + i * 0.08);
    const y = samples.v.map(() => 100);
    const rawZ = samples.v.map((i) => i % 7 === 0 ? 168 : plannedZ + 1.2);
    ingestPositions(hooks, samples.t, x, y, rawZ);
    hooks.lmapIngestForce(samples.t, samples.v.map(() => 50));
    return start + 3.0;
  };

  // A sustained cleaner pass at raw Z=2.2 is exactly what the old code
  // interpreted as index 10 / display layer 11. It is outside the part and
  // must not establish a layer or leak any force samples into the map.
  const wipe = series(0, 80, 0.02, (i) => i);
  ingestPositions(
    hooks,
    wipe.t,
    wipe.v.map(() => 242),
    wipe.v.map(() => 205),
    wipe.v.map(() => 2.2),
  );
  hooks.lmapIngestForce(wipe.t, wipe.v.map(() => 900));
  assert.equal(lmap.zTrack.armed, false);
  assert.equal(lmap.liveSeen, 0);

  let t = ingestLayer(2, 0.2);
  assert.ok(Math.abs(lmap.zTrack.offset - 1.2) < 1e-9);
  assert.equal(lmap.liveLayer, 0);

  for (const plannedZ of [0.4, 0.6, 0.8, 1.0]) {
    t = ingestLayer(t, plannedZ, 101);
  }

  assert.equal(lmap.liveLayer, 4);
  assert.equal(lmap.liveLayer + 1, 5);
  assert.equal(lmap.summary.layer_zs[lmap.liveLayer], 1.0);
  assert.equal(lmap.livePaths[10], undefined);
});


test("INDX in-part Z hop does not promote the live layer", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;

  lmap.summary = {
    n_layers: 4,
    layer_zs: [0.2, 0.4, 0.6, 0.8],
    bbox: [99, 121, 99, 101],
  };
  lmap.mode = "review";
  hooks.lmResetLive();
  lmap.mode = "live";
  lmap.captureActive = true;
  lmap.followLive = false;
  lmap.zTrack.anchorSegments = [[100, 100, 120, 100]];

  const first = series(0, 90, 0.02, (i) => i);
  ingestPositions(
    hooks,
    first.t,
    first.v.map((i) => 100 + i * 0.08),
    first.v.map(() => 100),
    first.v.map(() => 0.32),
  );
  hooks.lmapIngestForce(first.t, first.v.map(() => 40));
  assert.equal(lmap.zTrack.armed, true);
  assert.equal(lmap.liveLayer, 0);

  // A 1.2 s +0.20 mm hop is long enough to beat the old 20-sample gate,
  // but its rolling-median candidate is not sustained long enough to mean a
  // real layer transition.
  const hop = series(2, 60, 0.02, (i) => i);
  ingestPositions(
    hooks,
    hop.t,
    hop.v.map((i) => 101 + (i % 40) * 0.1),
    hop.v.map(() => 100),
    hop.v.map(() => 0.52),
  );
  assert.equal(lmap.liveLayer, 0);

  const returned = series(3.4, 70, 0.02, (i) => i);
  ingestPositions(
    hooks,
    returned.t,
    returned.v.map((i) => 102 + (i % 40) * 0.1),
    returned.v.map(() => 100),
    returned.v.map(() => 0.32),
  );
  assert.equal(lmap.liveLayer, 0);

  const second = series(5, 120, 0.02, (i) => i);
  ingestPositions(
    hooks,
    second.t,
    second.v.map((i) => 103 + (i % 40) * 0.1),
    second.v.map(() => 100),
    second.v.map(() => 0.52),
  );
  assert.equal(lmap.liveLayer, 1);
});


test("printer axis Z rejects a hop and corrects a pos_z layer lead", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;
  lmap.summary = {
    n_layers: 4,
    layer_zs: [0.2, 0.4, 0.6, 0.8],
    bbox: [99, 121, 99, 101],
  };
  lmap.mode = "live";
  lmap.captureActive = true;
  lmap.zTrack.armed = true;
  lmap.zTrack.offset = 0.46;
  lmap.zTrack.pitch = 0.2;
  lmap.liveLayer = 1;

  // A +0.26 mm hop is not close enough to the logical layer staircase.
  hooks.lmapIngestAxisZ(0.66, 1.0);
  hooks.lmapIngestAxisZ(0.66, 3.0);
  assert.equal(lmap.liveLayer, 1);

  // If compensated pos_z already put samples one layer ahead, two sustained
  // logical-Z reads correct the layer downward and discard that bad bucket.
  lmap.liveLayer = 2;
  lmap.livePaths[2] = { x: [100], y: [100], force: [1], seen: 1, stride: 1 };
  hooks.lmapIngestAxisZ(0.4, 4.0);
  hooks.lmapIngestAxisZ(0.4, 5.1);
  assert.equal(lmap.liveLayer, 1);
  assert.equal(lmap.livePaths[2], undefined);

  hooks.lmapIngestAxisZ(0.6, 6.0);
  hooks.lmapIngestAxisZ(0.6, 7.1);
  assert.equal(lmap.liveLayer, 2);
});


test("INDX mid-print attach seeds logical Z 6.2 / raw Z 7.4 as layer 31", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;

  lmap.summary = {
    n_layers: 40,
    layer_zs: Array.from({ length: 40 }, (_, i) => (i + 1) * 0.2),
    bbox: [99, 121, 99, 101],
  };
  lmap.mode = "review";

  hooks.renderLiveMapRun({
    state: "running",
    message: "Attached",
    progress_pct: 55,
    attached_existing: true,
    attach_axis_z: 6.2,
    attach_layer_index: 30,
  });

  assert.equal(lmap.captureActive, true);
  assert.equal(lmap.captureBlocked, false);
  assert.equal(lmap.zTrack.armed, false);

  // A stable raw height outside the part is a dock/wipe candidate, not a seed.
  const outside = series(0, 60, 0.02, (i) => i);
  ingestPositions(
    hooks,
    outside.t,
    outside.v.map(() => 242),
    outside.v.map(() => 205),
    outside.v.map(() => 7.4),
  );
  hooks.lmapIngestForce(outside.t, outside.v.map(() => 900));
  assert.equal(lmap.zTrack.armed, false);
  assert.equal(lmap.liveSeen, 0);

  const print = series(2, 70, 0.02, (i) => i);
  ingestPositions(
    hooks,
    print.t,
    print.v.map((i) => 100 + i * 0.08),
    print.v.map(() => 100),
    print.v.map((i) => i % 7 === 0 ? 168 : 7.4),
  );
  hooks.lmapIngestForce(print.t, print.v.map(() => 50));

  assert.equal(lmap.zTrack.armed, true);
  assert.ok(Math.abs(lmap.zTrack.offset - 1.2) < 1e-9);
  assert.equal(lmap.liveLayer, 30);  // zero-based index, display layer 31
  assert.equal(lmap.curLayer, 30);
  assert.ok(lmap.livePaths[30].seen > 0);
  assert.equal(lmap.livePaths[36], undefined);  // raw 7.4 must not be guessed
});


test("new tab opened mid-print seeds from printer Z and starts live capture", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;

  lmap.summary = {
    n_layers: 40,
    layer_zs: Array.from({ length: 40 }, (_, i) => (i + 1) * 0.2),
    bbox: [99, 121, 99, 101],
  };
  lmap.mode = "review";
  hooks.renderLiveMapRun({
    state: "running",
    progress_pct: 55,
    attached_existing: false,
    attach_axis_z: null,
    attach_layer_index: null,
  });

  assert.equal(lmap.captureActive, true);
  assert.equal(lmap.captureBlocked, false);
  assert.equal(lmap.axisTrack.seedPending, true);
  hooks.lmapIngestAxisZ(6.2, 10.0);
  hooks.lmapIngestAxisZ(6.2, 11.1);
  assert.equal(lmap.zTrack.armed, true);
  assert.equal(lmap.axisTrack.seedPending, false);
  assert.equal(lmap.liveLayer, 30);
  assert.equal(lmap.curLayer, 30);

  const samples = series(0, 70, 0.02, (i) => i);
  ingestPositions(
    hooks,
    samples.t,
    samples.v.map(() => 100),
    samples.v.map(() => 100),
    samples.v.map(() => 7.4),
  );
  hooks.lmapIngestForce(samples.t, samples.v.map(() => 50));
  assert.equal(lmap.posZ.t.length, 70);
  assert.ok(lmap.liveSeen > 0);
  assert.ok(lmap.livePaths[30].seen > 0);
});


test("mid-print printer-Z poll only seeds the matching active run", async () => {
  let runStartedAt = 123;
  const hooks = loadLiveMapHooks({
    fetchImpl: async (url) => {
      assert.equal(url, "/api/livemap/printer-z");
      return {
        ok: true,
        async json() {
          return {
            axis_z: 6.2,
            state: "PRINTING",
            online: true,
            stale_sec: 0,
            run_started_at: runStartedAt,
          };
        },
      };
    },
  });
  const { lmap } = hooks;
  lmap.summary = {
    n_layers: 40,
    layer_zs: Array.from({ length: 40 }, (_, i) => (i + 1) * 0.2),
    bbox: [99, 121, 99, 101],
  };
  lmap.mode = "live";
  lmap.captureActive = true;
  lmap.runStartedAt = 123;
  lmap.axisTrack.seedPending = true;

  await hooks.pollLiveMapPrinterZ(10.0);
  assert.equal(lmap.zTrack.armed, false);
  await hooks.pollLiveMapPrinterZ(11.1);
  assert.equal(lmap.zTrack.armed, true);
  assert.equal(lmap.liveLayer, 30);

  // A response tagged with a different backend run must not move this tab.
  runStartedAt = 456;
  lmap.axisTrack.seedPending = true;
  lmap.zTrack.armed = false;
  lmap.axisTrack.cand = null;
  lmap.axisTrack.candSince = null;
  await hooks.pollLiveMapPrinterZ(12.2);
  assert.equal(lmap.zTrack.armed, false);
  assert.equal(lmap.axisTrack.cand, null);
});


test("INDX attach during 0% setup falls back to first-layer calibration", () => {
  const hooks = loadLiveMapHooks();
  const { lmap } = hooks;

  lmap.summary = {
    n_layers: 4,
    layer_zs: [0.2, 0.4, 0.6, 0.8],
    bbox: [99, 111, 99, 101],
  };
  lmap.mode = "review";
  hooks.renderLiveMapRun({
    state: "running",
    progress_pct: 0,
    attached_existing: true,
    attach_axis_z: null,
    attach_layer_index: null,
  });

  assert.equal(lmap.captureActive, true);
  assert.equal(lmap.captureBlocked, false);
  assert.equal(lmap.zTrack.attachLayer, null);
});


test("starting from stale review loads matching live summary and anchor geometry", async () => {
  const requests = [];
  const hooks = loadLiveMapHooks({
    fetchImpl: async (url) => {
      requests.push(String(url));
      if (url === "/api/livemap/pending") {
        return {
          ok: true,
          async json() {
            return {
              name: "new-print.gcode",
              n_layers: 2,
              layer_zs: [0.2, 0.4],
              bbox: [99, 111, 99, 101],
            };
          },
        };
      }
      if (url === "/api/livemap/pending/layer/0") {
        return {
          ok: true,
          async json() {
            return { polylines: [{ x: [100, 110], y: [100, 100] }] };
          },
        };
      }
      return { ok: false, async json() { return {}; } };
    },
  });
  const { lmap } = hooks;
  lmap.summary = {
    name: "old-review.gcode",
    n_layers: 1,
    layer_zs: [9.9],
    bbox: [0, 1, 0, 1],
  };
  lmap.mode = "review";

  hooks.renderLiveMapRun({
    name: "new-print.gcode",
    state: "preparing",
    progress_pct: 0,
    started_at: 123,
    attached_existing: false,
  });
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(lmap.mode, "live");
  assert.equal(lmap.summary.name, "new-print.gcode");
  assert.deepEqual(lmap.summary.layer_zs, [0.2, 0.4]);
  assert.equal(lmap.zTrack.anchorSegments.length, 1);
  assert.ok(requests.includes("/api/livemap/pending"));
  assert.ok(requests.includes("/api/livemap/pending/layer/0"));
});
