import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ProjectionPoint } from "../../types";
import { clamp, fmtFixed, fmtInt } from "./format";
import { readPalette, type VisualPalette } from "./palette";

export interface ParticleCloud3DProps {
  points: ProjectionPoint[];
  activeStep: number | null;
  reducer?: string;
  onSelect?(p: ProjectionPoint): void;
  height?: number;
}

interface CloudEntry {
  object: THREE.Points;
  data: ProjectionPoint[];
}

interface CloudScene {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  group: THREE.Group;
  raycaster: THREE.Raycaster;
  pointer: THREE.Vector2;
  entries: CloudEntry[];
  trail: THREE.Line | null;
  data: ProjectionPoint[];
  dirty: boolean;
  reset: () => void;
}

const DEFAULT_HEIGHT = 340;
const MIN_ZOOM = 2.4;
const MAX_ZOOM = 8.5;

function toWorld(point: ProjectionPoint, scale: number): [number, number, number] {
  // The projection's z becomes the vertical axis of the observatory floor.
  return [point.x * scale, point.z * scale, point.y * scale];
}

function disposePointsObject(entry: CloudEntry): void {
  entry.object.geometry.dispose();
  const material = entry.object.material;
  if (Array.isArray(material)) material.forEach((item) => item.dispose());
  else material.dispose();
}

function pickPoint(scene: CloudScene, clientX: number, clientY: number): ProjectionPoint | null {
  const rect = scene.renderer.domElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  scene.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  scene.pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  scene.raycaster.setFromCamera(scene.pointer, scene.camera);
  const objects = scene.entries.map((entry) => entry.object);
  if (objects.length === 0) return null;
  const hits = scene.raycaster.intersectObjects(objects, false);
  const hit = hits[0];
  if (!hit || typeof hit.index !== "number") return null;
  const entry = scene.entries.find((candidate) => candidate.object === hit.object);
  if (!entry) return null;
  return entry.data[hit.index] ?? null;
}

function makePoints(
  data: ProjectionPoint[],
  scale: number,
  color: string,
  size: number,
  opacity: number,
): CloudEntry {
  const positions = new Float32Array(data.length * 3);
  data.forEach((point, index) => {
    const [x, y, z] = toWorld(point, scale);
    positions[index * 3] = x;
    positions[index * 3 + 1] = y;
    positions[index * 3 + 2] = z;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: new THREE.Color(color),
    size,
    sizeAttenuation: true,
    transparent: opacity < 1,
    opacity,
    depthWrite: true,
  });
  return { object: new THREE.Points(geometry, material), data };
}

function kindLabel(kind: ProjectionPoint["kind"]): string {
  return kind === "prompt" ? "提示" : "生成";
}

export function ParticleCloud3D({
  points,
  activeStep,
  reducer = "UMAP / PCA",
  onSelect,
  height = DEFAULT_HEIGHT,
}: ParticleCloud3DProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<CloudScene | null>(null);
  const onSelectRef = useRef(onSelect);
  const hoveredIdRef = useRef<string | null>(null);
  const [hovered, setHovered] = useState<ProjectionPoint | null>(null);
  const [listOpen, setListOpen] = useState(false);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || typeof window === "undefined") return undefined;

    const palette: VisualPalette = readPalette(host);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 60);
    camera.position.set(0, 1.2, 4.4);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "low-power",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    const canvas = renderer.domElement;
    canvas.className = "lx-cloud__canvas";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "3D 语义投影点云；键盘用户可使用上方的列表按钮");
    host.appendChild(canvas);

    const group = new THREE.Group();
    scene.add(group);

    // A flat observatory floor: 1px warm lines on a dark plane, no glow.
    const gridPositions: number[] = [];
    const floor = 1.8;
    const floorY = -1.55;
    for (let step = -floor; step <= floor + 1e-6; step += 0.3) {
      gridPositions.push(-floor, floorY, step, floor, floorY, step);
      gridPositions.push(step, floorY, -floor, step, floorY, floor);
    }
    const gridGeometry = new THREE.BufferGeometry();
    gridGeometry.setAttribute("position", new THREE.Float32BufferAttribute(gridPositions, 3));
    const gridMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(palette.line),
      transparent: true,
      opacity: 0.75,
    });
    const grid = new THREE.LineSegments(gridGeometry, gridMaterial);
    scene.add(grid);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.1;

    const cloud: CloudScene = {
      renderer,
      scene,
      camera,
      group,
      raycaster,
      pointer: new THREE.Vector2(),
      entries: [],
      trail: null,
      data: [],
      dirty: true,
      reset: () => undefined,
    };
    sceneRef.current = cloud;

    let dragging = false;
    let moved = false;
    let interacted = false;
    let pointerInside = false;
    let pendingPick = false;
    let hoverX = 0;
    let hoverY = 0;
    let lastPointer = { x: 0, y: 0 };

    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    let reduceMotion = motionQuery.matches;
    const onMotionChange = () => {
      reduceMotion = motionQuery.matches;
      cloud.dirty = true;
    };
    motionQuery.addEventListener("change", onMotionChange);

    const markDirty = () => {
      cloud.dirty = true;
    };

    const onPointerDown = (event: PointerEvent) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      dragging = true;
      moved = false;
      interacted = true;
      lastPointer = { x: event.clientX, y: event.clientY };
      if (typeof canvas.setPointerCapture === "function") {
        try {
          canvas.setPointerCapture(event.pointerId);
        } catch {
          // Pointer capture is a nicety; sandboxed webviews may refuse it.
        }
      }
    };

    const onPointerMove = (event: PointerEvent) => {
      pointerInside = true;
      hoverX = event.clientX;
      hoverY = event.clientY;
      pendingPick = true;
      if (!dragging) return;
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      lastPointer = { x: event.clientX, y: event.clientY };
      cloud.group.rotation.y += dx * 0.008;
      cloud.group.rotation.x = clamp(cloud.group.rotation.x + dy * 0.006, -1.1, 1.1);
      markDirty();
    };

    const onPointerUp = (event: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      if (typeof canvas.hasPointerCapture === "function" && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      if (!moved) {
        const hit = pickPoint(cloud, event.clientX, event.clientY);
        if (hit) onSelectRef.current?.(hit);
      }
      pendingPick = true;
    };

    const onPointerLeave = () => {
      pointerInside = false;
      pendingPick = true;
      dragging = false;
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      camera.position.z = clamp(camera.position.z + event.deltaY * 0.0016, MIN_ZOOM, MAX_ZOOM);
      markDirty();
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    let resizeFrame = 0;
    const resize = () => {
      const width = host.clientWidth;
      const heightPx = host.clientHeight;
      if (width <= 0 || heightPx <= 0) return;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(width, heightPx, false);
      camera.aspect = width / heightPx;
      camera.updateProjectionMatrix();
      markDirty();
    };
    const observer = new ResizeObserver(() => {
      if (resizeFrame !== 0) return;
      resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        resize();
      });
    });
    observer.observe(host);
    resize();

    cloud.reset = () => {
      cloud.group.rotation.set(0, 0, 0);
      camera.position.set(0, 1.2, 4.4);
      camera.lookAt(0, 0, 0);
      cloud.dirty = true;
    };

    let animationFrame = 0;
    let lastRender = 0;
    const loop = (time: number) => {
      animationFrame = window.requestAnimationFrame(loop);
      const current = sceneRef.current;
      if (!current) return;

      const autoRotate = !reduceMotion && !interacted && current.data.length > 0;
      if (autoRotate) {
        current.group.rotation.y += 0.0035;
        current.dirty = true;
      }

      if (pendingPick) {
        pendingPick = false;
        const hit = pointerInside && !dragging ? pickPoint(current, hoverX, hoverY) : null;
        const nextId = hit?.id ?? null;
        if (nextId !== hoveredIdRef.current) {
          hoveredIdRef.current = nextId;
          setHovered(hit);
        }
      }

      if (!current.dirty) return;
      if (time - lastRender < 32) return; // cap idle / interaction rendering at ~30 fps
      lastRender = time;
      renderer.render(scene, camera);
      current.dirty = false;
    };
    animationFrame = window.requestAnimationFrame(loop);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      window.cancelAnimationFrame(resizeFrame);
      observer.disconnect();
      motionQuery.removeEventListener("change", onMotionChange);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
      canvas.removeEventListener("pointerleave", onPointerLeave);
      canvas.removeEventListener("wheel", onWheel);
      for (const entry of cloud.entries) disposePointsObject(entry);
      if (cloud.trail) {
        cloud.trail.geometry.dispose();
        const material = cloud.trail.material;
        if (Array.isArray(material)) material.forEach((item) => item.dispose());
        else material.dispose();
      }
      scene.traverse((object) => {
        const candidate = object as THREE.Mesh & {
          geometry?: THREE.BufferGeometry;
          material?: THREE.Material | THREE.Material[];
        };
        if (candidate.geometry) candidate.geometry.dispose();
        if (candidate.material) {
          if (Array.isArray(candidate.material)) candidate.material.forEach((item) => item.dispose());
          else candidate.material.dispose();
        }
      });
      renderer.dispose();
      renderer.forceContextLoss();
      if (canvas.parentElement === host) host.removeChild(canvas);
      sceneRef.current = null;
      hoveredIdRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cloud = sceneRef.current;
    if (!cloud) return;
    const host = hostRef.current;
    if (!host) return;

    for (const entry of cloud.entries) {
      cloud.group.remove(entry.object);
      disposePointsObject(entry);
    }
    cloud.entries = [];
    if (cloud.trail) {
      cloud.group.remove(cloud.trail);
      cloud.trail.geometry.dispose();
      const material = cloud.trail.material;
      if (Array.isArray(material)) material.forEach((item) => item.dispose());
      else material.dispose();
      cloud.trail = null;
    }

    const palette = readPalette(host);
    const valid = points.filter(
      (point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z),
    );
    cloud.data = valid;
    const maxAbs = valid.reduce(
      (max, point) => Math.max(max, Math.abs(point.x), Math.abs(point.y), Math.abs(point.z)),
      0,
    );
    const scale = maxAbs > 0 ? 1.45 / maxAbs : 1;

    const promptData = valid.filter((point) => point.kind === "prompt");
    const generatedData = valid.filter((point) => point.kind !== "prompt");
    const activeData = activeStep == null ? [] : generatedData.filter((point) => point.step === activeStep);
    const restGenerated =
      activeStep == null ? generatedData : generatedData.filter((point) => point.step !== activeStep);

    if (promptData.length > 0) {
      cloud.entries.push(makePoints(promptData, scale, palette.reasoning, 0.07, 0.95));
    }
    if (restGenerated.length > 0) {
      cloud.entries.push(makePoints(restGenerated, scale, palette.accent, 0.095, 1));
    }
    if (activeData.length > 0) {
      cloud.entries.push(makePoints(activeData, scale, palette.inkStrong, 0.15, 1));
    }
    for (const entry of cloud.entries) cloud.group.add(entry.object);

    // Trajectory: consecutive generated positions in decoding order.
    const trailData = [...generatedData].sort(
      (a, b) => (a.step ?? a.index) - (b.step ?? b.index),
    );
    if (trailData.length >= 2) {
      const positions = new Float32Array(trailData.length * 3);
      trailData.forEach((point, index) => {
        const [x, y, z] = toWorld(point, scale);
        positions[index * 3] = x;
        positions[index * 3 + 1] = y;
        positions[index * 3 + 2] = z;
      });
      const trailGeometry = new THREE.BufferGeometry();
      trailGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const trailMaterial = new THREE.LineBasicMaterial({
        color: new THREE.Color(palette.accentDeep),
        transparent: true,
        opacity: 0.6,
      });
      const trail = new THREE.Line(trailGeometry, trailMaterial);
      cloud.group.add(trail);
      cloud.trail = trail;
    }

    if (hoveredIdRef.current != null && !valid.some((point) => point.id === hoveredIdRef.current)) {
      hoveredIdRef.current = null;
      setHovered(null);
    }
    cloud.dirty = true;
  }, [points, activeStep]);

  const promptCount = points.filter((point) => point.kind === "prompt").length;
  const generatedCount = points.length - promptCount;
  const activeCount =
    activeStep == null ? 0 : points.filter((point) => point.kind === "generated" && point.step === activeStep).length;
  const listPoints = points.slice(0, 200);

  return (
    <section className="lx-panel lx-cloud" aria-label="3D 语义投影">
      <header className="lx-panel__head">
        <div className="lx-panel__heading">
          <h3 className="lx-panel__title">语义投影</h3>
          <p className="lx-panel__sub">UMAP / PCA 投影 · 替代透镜</p>
        </div>
        <div className="lx-panel__controls">
          <span className="lx-chip num">{reducer}</span>
          <button
            type="button"
            className="lx-button"
            onClick={() => sceneRef.current?.reset()}
            aria-label="重置 3D 视角"
          >
            重置视角
          </button>
          <button
            type="button"
            className="lx-button"
            aria-expanded={listOpen}
            onClick={() => setListOpen((open) => !open)}
            aria-label="以列表方式浏览投影点"
          >
            列表
          </button>
        </div>
      </header>

      <div className="lx-panel__meta">
        <span className="lx-chip">提示 {fmtInt(promptCount)}</span>
        <span className="lx-chip lx-chip--accent">生成 {fmtInt(generatedCount)}</span>
        <span className="lx-chip num">当前步 {activeStep == null ? "—" : fmtInt(activeStep)}</span>
        {activeCount > 0 ? <span className="lx-chip num">高亮 {fmtInt(activeCount)}</span> : null}
        <span className="lx-panel__note">拖动旋转 · 滚轮缩放 · 单击选中</span>
      </div>

      <div className="lx-cloud__stage" ref={hostRef} style={{ height: `${height}px` }}>
        <div className="lx-cloud__readout" aria-live="polite">
          {hovered ? (
            <>
              <span className="lx-cloud__readout-token">{hovered.token || "∅"}</span>
              <span className={`lx-cloud__readout-kind lx-cloud__readout-kind--${hovered.kind}`}>
                {kindLabel(hovered.kind)}
              </span>
              <span className="num">
                x {fmtFixed(hovered.x, 3)} · y {fmtFixed(hovered.y, 3)} · z {fmtFixed(hovered.z, 3)}
              </span>
              <span className="num muted">步 {hovered.step == null ? "—" : hovered.step} · 索引 {hovered.index}</span>
            </>
          ) : (
            <span className="muted">悬停点查看 Token 与坐标</span>
          )}
        </div>

        {points.length === 0 ? (
          <div className="lx-empty lx-cloud__empty">
            <p className="lx-empty__title">尚无投影点</p>
            <p className="lx-empty__hint">透镜在累计至少 8 个 Token 后计算投影；此前只有文本与概率。</p>
            <p className="lx-empty__hint">投影只来自透镜模型，不会伪装成主模型的内部张量。</p>
          </div>
        ) : null}

        {listOpen ? (
          <div className="lx-cloud__list" role="group" aria-label="投影点列表">
            <div className="lx-cloud__list-head">
              <span>投影点</span>
              <button type="button" className="lx-button" onClick={() => setListOpen(false)}>
                关闭
              </button>
            </div>
            <ul className="lx-cloud__list-items">
              {listPoints.map((point) => (
                <li key={point.id}>
                  <button
                    type="button"
                    className={`lx-cloud__list-item${
                      hovered?.id === point.id ? " is-active" : ""
                    }${point.kind === "generated" ? " is-generated" : ""}`}
                    onMouseEnter={() => {
                      hoveredIdRef.current = point.id;
                      setHovered(point);
                    }}
                    onFocus={() => {
                      hoveredIdRef.current = point.id;
                      setHovered(point);
                    }}
                    onClick={() => onSelect?.(point)}
                  >
                    <span className="lx-cloud__list-token">{point.token || "∅"}</span>
                    <span className="lx-cloud__list-kind">{kindLabel(point.kind)}</span>
                    <span className="lx-cloud__list-coords num">
                      {fmtFixed(point.x, 2)}, {fmtFixed(point.y, 2)}, {fmtFixed(point.z, 2)}
                    </span>
                    <span className="lx-cloud__list-index num">#{point.index}</span>
                  </button>
                </li>
              ))}
            </ul>
            {points.length > listPoints.length ? (
              <p className="lx-cloud__list-note">仅列出前 {listPoints.length} 个点</p>
            ) : null}
          </div>
        ) : null}
      </div>

      <footer className="lx-cloud__foot">
        <p className="lx-panel__note">
          坐标来自透镜模型隐状态的降维结果（最多 256 点），不是模型原始的 4096 维坐标。
        </p>
        <p className="lx-panel__note">
          点颜色：提示 Token 为灰米色，生成 Token 为铜色，当前步为奶油色高亮；细线连接相邻生成位置。
        </p>
      </footer>
    </section>
  );
}

export default ParticleCloud3D;
